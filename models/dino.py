import torch
import torch.nn as nn
import pytorch_lightning as pl

from loss.dino_loss import DINOLoss
from models.models import DINOHead, MultiCropWrapper, VisionTransformer
from utils import bool_flag, utils

import models as vits
from torchvision import models as torchvision_models


class DINO(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.init_backbones()

        # Create student and teacher networks
        self.student_head = DINOHead(
            self.student_backbone.embed_dim,
            self.hparams.out_dim,
        )
        self.teacher_head = DINOHead(
            self.teacher_backbone.embed_dim,
            self.hparams.out_dim,
            norm_last_layer=True,
        )

        self.student = MultiCropWrapper(self.student_backbone, self.student_head)
        self.teacher = MultiCropWrapper(self.teacher_backbone, self.teacher_head)

        # Initialize teacher with student weights and stop its gradients 
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False
            
        # Loss
        self.loss_fn = DINOLoss(**vars(self.hparams))


    def init_backbones(self):
        # Get from folder
        if self.hparams.arch in vits.__dict__.keys():
            self.student_backbone = vits.__dict__[self.hparams.arch](
                patch_size=self.hparams.patch_size,
                drop_path_rate=self.hparams.drop_path_rate,
            )
            self.teacher_backbone = vits.__dict__[self.hparams.arch](patch_size=self.hparams.patch_size)
            self.embed_dim = self.student_backbone.embed_dim

        # Get from url 
        elif self.hparams.arch in torch.hub.list("facebookresearch/xcit:main"):
            self.student_backbone = torch.hub.load('facebookresearch/xcit:main', self.hparams.arch, pretrained=False, drop_path_rate=self.hparams.drop_path_rate)
            self.teacher_backbone = torch.hub.load('facebookresearch/xcit:main', self.hparams.arch, pretrained=False)
            self.embed_dim = self.student_backbone.embed_dim

        # Fet from torchvision models packages
        elif self.hparams.arch in torchvision_models.__dict__.keys():
            self.student_backbone = torchvision_models.__dict__[self.hparams.arch]()
            self.teacher_backbone = torchvision_models.__dict__[self.hparams.arch]()
            self.embed_dim = self.student_backbone.fc.weight.shape[1]
        else:
            print(f"Unknow architecture: {self.hparams.arch}")


    def training_step(self, batch, batch_idx):
        images, _ = batch
        
        teacher_output = self.teacher(images[:self.hparams.n_global_crops])
        student_output = self.student(images)
        
        loss = self.loss_fn(student_output, teacher_output, self.current_epoch)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # EMA 
        with torch.no_grad():
            m = self.trainer.lr_schedulers[2]['scheduler'][self.global_step]
            
            for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def configure_optimizers(self):
        if self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)  
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=0, momentum=0.9)
        elif self.hparams.optimizer == "lars":
            optimizer = utils.LARS(self.parameters()) 

        lr_schedule = utils.cosine_scheduler(
            self.hparams.lr * (self.hparams.batch_size_per_gpu * utils.get_world_size()) / 256.,  
            self.hparams.min_lr,
            self.hparams.epochs, 
            len(self.trainer.datamodule),
            warmup_epochs=self.hparams.warmup_epochs,
        )
        wd_schedule = utils.cosine_scheduler(
            self.hparams.weight_decay,
            self.hparams.weight_decay_end,
            self.hparams.epochs, 
            len(self.trainer.datamodule),
        )

        momentum_schedule = utils.cosine_scheduler(self.hparams.momentum_teacher, 1, self.hparams.epochs, len(self.trainer.datamodule))
            
        return [optimizer], [lr_schedule, wd_schedule, momentum_schedule]
    

    @staticmethod
    def add_specific_args(parser):  
        parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
        parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
        parser.add_argument("--warmup_epochs", default=10, type=int, help="Number of epochs for the linear learning-rate warm up.")

        parser.add_argument('--batch_size_per_gpu', default=64, type=int, help='Per-GPU batch-size : number of distinct images loaded on one GPU.')

        parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of linear warmup (highest LR used during training). The learning rate is linearly scaled with the batch size, and specified here for a reference batch size of 256.""")
        parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.""")
        parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')

        parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.""")
        parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the weight decay. We use a cosine schedule for WD and using a larger decay by the end of training improves performance for ViTs.""")
        
        parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
        parser.add_argument('--use_fp16', type=bool_flag, default=True, help="""Whether or not to use half precision for training. Improves training time and memory requirements, but can provoke instability and slight decay of performance. We recommend disabling mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")

        parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures. 0 for disabling.""")
        parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs during which we keep the output layer fixed. Typically doing so during the first epoch helps training. Try increasing this value if the loss does not decrease.""")
        parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
        parser.add_argument('--seed', default=0, type=int, help='Random seed.')

        parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set updistributed training; see https://pytorch.org/docs/stable/distributed.html""")
        parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
        
        parser.add_argument('--n_global_crops', default=2, type=int, help='Number of global views (crops) to generate.')
        parser.add_argument('--n_local_crops', default=8, type=int, help='Number of local views (crops) to generate.')
       
        parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA parameter for teacher update. The value is increased to 1 during training with cosine schedule. We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
        parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int, help='Number of warmup epochs for the teacher temperature (Default: 30).')
        
        return parser