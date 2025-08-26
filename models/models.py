import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import bool_flag
from config import torchvision_archs

class DINOHead(nn.Module):
    """Placeholder for the DINO projection head."""

    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, bottleneck_dim)
        
        layers = [
            nn.Linear(bottleneck_dim, out_dim, bias=False),
            nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False), name="weight")
        ]
        self.last_layer = layers[1] 


    def forward(self, x):
        x = F.gelu(self.bn1(self.fc1(x)))
        x = F.gelu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
    
    @staticmethod
    def add_specific_args(parser):
        parser.add_argument('--norm_last_layer', default=True, type=bool_flag,help="""Whether or not to weight normalize the last layer of the DINO head. Not normalizing leads to better performance but can make the training unstable. In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
        parser.add_argument('--use_bn_in_head', default=False, type=bool_flag, help="Whether to use batch normalizations in projection head.")
        return parser
    
class MultiCropWrapper(nn.Module):
    """
    Wraps a model to process multiple crops of an image.
    This is a simplified version. The original is in `utils.py`.
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        n_crops = len(x)
        concatenated = torch.cat(x, dim=0)
        cls_tokens = self.backbone(concatenated)
        logits = self.head(cls_tokens)

        chunks = logits.chunk(n_crops, dim=0)
        return chunks
    
class VisionTransformer(nn.Module):
    """Placeholder for the Vision Transformer model."""
    def __init__(self, patch_size=16, embed_dim=384, num_heads=6, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # A simple conv layer to simulate patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # A dummy transformer block
        self.blocks = nn.Sequential(*[nn.Linear(embed_dim, embed_dim) for _ in range(12)])

        self.norm = nn.LayerNorm(embed_dim)

        # The real ViT has a more complex head
        self.fc = nn.Identity() 
        print("--- Using Placeholder VisionTransformer ---")

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.blocks(x)
        x = self.norm(x)
        
        # Return CLS token
        return x[:, 0]
    
    @staticmethod
    def add_specific_args(parser):
        parser.add_argument('--arch', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"), help="""Name of architecture to train. For quick experiments with ViTs,we recommend using vit_tiny or vit_small.""")
        parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels of input square patches - default 16 (for 16x16 patches). Using smaller values leads to better performance but requires more memory. Applies only for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling mixed precision training (--use_fp16 false) to avoid unstabilities.""")
        parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of the DINO head output. For complex and large datasets large values (like 65k) work well.""")
        return parser
    

class MultiCropWrapper(nn.Module):
    """
    Wraps a model to process multiple crops of an image.
    This is a simplified version. The original is in `utils.py`.
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        n_crops = len(x)
        concatenated = torch.cat(x, dim=0)
        cls_tokens = self.backbone(concatenated)
        logits = self.head(cls_tokens)
        # Split logits back per crop
        chunks = logits.chunk(n_crops, dim=0)
        return chunks
    
    @staticmethod
    def add_specific_args(parser):
        parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.), help="""Scale range of the cropped image before resizing, relatively to the origin image. Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
        parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small local views to generate. Set this parameter to 0 to disable multi-crop training. When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
        parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4), help="""Scale range of the cropped image before resizing, relatively to the origin image. Used for small local view cropping of multi-crop.""")
        return parser