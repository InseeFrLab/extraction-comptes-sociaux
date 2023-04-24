"""Module d'apprentissage"""
import yaml
import sys
from time import sleep
from typing import Tuple
import albumentations as album
from pathlib import Path
import torch
from torch import nn
import gc
import mlflow
import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from .tablenet import MarmotDataModule, TableNetModule, DiceLoss
from .utils import get_root_path
from .optimizers import optimizers
from .schedulers import schedulers


def main(remote_server_uri, experiment_name, run_name, config_path):
    # Parameters
    with open(get_root_path() / config_path, "r") as stream:
        config = yaml.safe_load(stream)
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

    torch.cuda.empty_cache()
    gc.collect()

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
    data_dir = "./data/marmot_data"
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
        transforms_augmentation=transforms_preprocessing,
        batch_size=batch_size,
        num_workers=0,
    )  # type: ignore

    model = TableNetModule(
        batch_norm=batch_norm,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        scheduler=scheduler,
        scheduler_params=scheduler_params,
        scheduler_interval=scheduler_interval,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss", save_top_k=1, save_last=True, mode="min"
    )
    early_stop_callback = EarlyStopping(
        monitor="validation_loss", mode="min", patience=patience
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    batch_size = get_batch_size(
        model=model.model,
        device=model.device,
        input_shape=(3, *image_size),
        output_shape=image_size,
        dataset_size=len(data_module.dataset_train),
    )

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog(registered_model_name="extraction")
    with mlflow.start_run(run_name=run_name):
        trainer = pl.Trainer(
            accelerator="auto",
            strategy="auto",
            callbacks=[lr_monitor, checkpoint_callback, early_stop_callback],
            max_epochs=max_epochs,
            num_sanity_val_steps=num_sanity_val_steps,
        )
        trainer.fit(model, datamodule=data_module)
        trainer.test(datamodule=data_module)


def get_batch_size(
    model: nn.Module,
    device: torch.device,
    input_shape: Tuple[int, int, int],
    output_shape: Tuple[int, int],
    dataset_size: int,
    max_batch_size: int = None,
    num_iterations: int = 5,
) -> int:
    model.to(device)
    model.train(True)
    optimizer = torch.optim.Adam(model.parameters())

    print("Test batch size")
    batch_size = 2
    while True:
        if max_batch_size is not None and batch_size >= max_batch_size:
            batch_size = max_batch_size
            break
        if batch_size >= dataset_size:
            batch_size = batch_size // 2
            break
        try:
            for _ in range(num_iterations):
                # dummy inputs and targets
                inputs = torch.rand(*(batch_size, *input_shape), device=device)
                table_targets = torch.rand(
                    *(batch_size, *output_shape), device=device
                )
                column_targets = torch.rand(
                    *(batch_size, *output_shape), device=device
                )

                output_table, output_column = model(inputs)

                dice_loss = DiceLoss()
                loss_table = dice_loss(output_table, table_targets)
                loss_column = dice_loss(output_column, column_targets)

                loss = loss_table + loss_column
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            batch_size *= 2
            print(f"\tTesting batch size {batch_size}")
            sleep(3)
        except RuntimeError:
            print(f"\tOOM at batch size {batch_size}")
            batch_size //= 2
            break
    del model, optimizer
    torch.cuda.empty_cache()
    print(f"Final batch size {batch_size}")
    return batch_size


if __name__ == "__main__":
    # MLFlow params
    remote_server_uri = sys.argv[1]
    experiment_name = sys.argv[2]
    run_name = sys.argv[3]
    config_path = sys.argv[4]

    main(remote_server_uri, experiment_name, run_name, config_path)
