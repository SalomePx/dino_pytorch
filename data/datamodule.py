import os

from torch.utils.data import DataLoader
import torchvision
import pytorch_lightning as pl

from data.augmentation import DataAugmentationDINO
import requests
import zipfile


class DINODataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for DINO.
    Handles dataset creation and data loaders.
    """
    def __init__(self, data_path: str, batch_size:int):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

        if not os.path.exists(self.data_path):
            raise ModuleNotFoundError(f"Data path {self.data_path} does not exist. You might first download ImageNet-Mini to this link: https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000.")
        
        # Define DINO augmentations
        self.transform = DataAugmentationDINO(
            global_crops_scale=(0.4, 1.0),
            local_crops_scale=(0.05, 0.4),
            n_local_crops=8
        )

    def __len__(self):
        return len(self.train_dataloader())

    def setup(self, stage=None):
        self.train_dataset = torchvision.datasets.ImageFolder(os.path.join(self.data_path, 'train'), transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )
    
    @staticmethod
    def add_specific_args(parser):  
        parser.add_argument('--batch_size_per_gpu', default=64, type=int, help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
        parser.add_argument('--epochs', type=str, required=True, help='Maximum number of epochs.')
        parser.add_argument('--num_workers', type=str, required=True, help='Num Workers for traning.')
        parser.add_argument('--data_path', type=str, required=True, help='Path to the ImageNet training data.')
        return parser