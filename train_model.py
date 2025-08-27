import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data.augmentation import DataAugmentationDINO
from data.datamodule import DINODataModule
from loss.dino_loss import DINOLoss
from models.dino import DINO
from models.models import VisionTransformer

parser = argparse.ArgumentParser()
parser = DINO.add_specific_args(parser)
parser = DINOLoss.add_specific_args(parser)
parser = DataAugmentationDINO.add_specific_args(parser)
parser = VisionTransformer.add_specific_args(parser)
args = parser.parse_args()

pl.seed_everything(123)


#################################
#####        DATASET        #####
#################################

args.data_path = 'dataset/imagenet-mini'
datamodule = DINODataModule(
    data_path=args.data_path,
    batch_size=args.batch_size_per_gpu,
)

#################################
#####         MODEL         #####
#################################

model = DINO(**vars(args))


#################################
#####       TRAINING        #####
#################################

checkpoint_callback = ModelCheckpoint(
    dirpath='./checkpoints',
    filename='dino-{epoch:02d}-{train_loss:.2f}',
    save_top_k=1,
    monitor='val_loss',
    mode='min'
)

trainer = pl.Trainer(
    max_epochs=args.epochs,
    accelerator='auto',
    callbacks=[checkpoint_callback],
)

trainer.fit(model, datamodule)

