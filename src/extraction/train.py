"""Module d'apprentissage"""
import yaml
import sys
import os
import albumentations as album
from pathlib import Path
import torch
import gc
import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from tablenet import MarmotDataModule, TableNetModule
from utils import get_root_path
from optimizers import optimizers
from schedulers import schedulers


def main(config_path):
    # Parameters
    with open(get_root_path() / config_path, "r") as stream:
        config = yaml.safe_load(stream)
    gpus = config["gpus"]
    batch_size = config["batch_size"]
    max_epochs = config["max_epochs"]
    num_sanity_val_steps = config["num_sanity_val_steps"]
    patience = config["patience"]
    batch_norm = config["batch_norm"]
    fp_data = config["fp_data"]

    optimizer_params = config["optimizer_params"]
    optimizer = optimizer_params.pop("optimizer")
    optimizer = optimizers[optimizer]

    scheduler_params = config["scheduler_params"]
    scheduler = scheduler_params.pop("scheduler")
    scheduler_interval = scheduler_params.pop("interval")
    scheduler = schedulers[scheduler]
    strategy = "ddp" if gpus > 1 else None
    print(f'---------------------------------{optimizer_params}')
    torch.cuda.empty_cache()
    gc.collect()

    # Choix du nombre de coeur du CPU pour l'entrainement
    if gpus == 0:
        cores = os.cpu_count()
        torch.set_num_threads(cores)

    image_size = (896, 896)
    transforms_augmentation = album.Compose([])
    transforms_augmentation = album.Compose(
        [
            album.Resize(1024, 1024, always_apply=True),
            album.RandomResizedCrop(
                *image_size, scale=(0.7, 1.0), ratio=(0.7, 1)
            ),
            album.HorizontalFlip(),
            album.VerticalFlip(),
            album.Normalize(),
            ToTensorV2(),
        ]
    )

    transforms_preprocessing = album.Compose(
        [
            album.Resize(*image_size, always_apply=True),
            album.Normalize(),
            ToTensorV2(),
        ]
    )

    # Data for the training pipeline
    # Clean up this code
    
    data_dir = "extraction-comptes-sociaux/data/marmot_data"
    siren_test = [
        "305756413",
        "324084698",
        "326300159",
        "331154765",
        "333916385",
        "334303823",
        "344066733",
        "393525852",
        "393712286",
        "411787567",
        "414728337",
        "552065187",
        "552081317",
        "702012956",
        "797080850",
    ]
    test_data = [
        Path(data_dir).joinpath(siren + ".bmp") for siren in siren_test
    ]

    train_data = [
        path
        for path in (
            list(Path(data_dir).glob("*.png"))
            + list(Path(data_dir).glob("*.bmp"))
        )
        if path not in test_data
    ]
    
    if not fp_data:
        train_data = [path for path in train_data if len(path.name) > 13]

    # Data module
    data_module = MarmotDataModule(
        train_data=train_data,
        test_data=test_data,
        transforms_preprocessing=transforms_preprocessing,
        transforms_augmentation=transforms_augmentation,
        batch_size=batch_size,
        num_workers=0,
    )  # type: ignore

    model = TableNetModule(
        batch_norm=batch_norm,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        scheduler=scheduler,
        scheduler_params=scheduler_params,
        scheduler_interval=scheduler_interval
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss", save_top_k=1, save_last=True, mode="min"
    )
    early_stop_callback = EarlyStopping(
        monitor="validation_loss", mode="min", patience=patience
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
            callbacks=[lr_monitor, checkpoint_callback, early_stop_callback],
            max_epochs=max_epochs,
            gpus=gpus,
            num_sanity_val_steps=num_sanity_val_steps,
            strategy=strategy
        )
    print(f' ------------------------------------------------------------------------------------- {data_module.data}------------------------------------------------------------------')
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)

if __name__ == "__main__":
    # MLFlow param
    config_path = "config/tablenet_config/tablenet_config_cycle.yaml"

    main(config_path)
