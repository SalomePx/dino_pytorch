import os
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data.augmentation import DataAugmentationDINO
from data.datamodule import DINODataModule
from models.dino import DINO
from models.models import MultiCropWrapper
from utils import utils

argparser = argparse.ArgumentParser(description="DINO Training Script")
parser = argparse.ArgumentParser()
parser = MultiCropWrapper.add_specific_args(parser)
parser = DINO.add_specific_args(parser)
parser = DataAugmentationDINO.add_specific_args(parser)
args = parser.parse_args()

pl.seed_everything(123)
utils.init_distributed_mode(args)


#################################
#####        DATASET        #####
#################################

datamodule = DINODataModule(
    data_path=args.data_path
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
    gpus=args.gpus,
    callbacks=[checkpoint_callback],
    precision=16,
)

trainer.fit(model, datamodule)

