"""TableNet Module."""
from typing import Dict, Union
import os
import mlflow
import pytorch_lightning as pl
import torch
from torch import optim
import numpy as np
from PIL import Image
from torch import nn, optim
from torch.nn import functional as F
from torchvision.models import vgg19, vgg19_bn

EPSILON = 1e-15


class TableNetModule(pl.LightningModule):
    """
    Pytorch Lightning Module for TableNet.
    """

    def __init__(
        self,
        optimizer: Union[optim.SGD, optim.Adam],
        optimizer_params: Dict,
        scheduler: Union[optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.ReduceLROnPlateau],
        scheduler_params: Dict,
        scheduler_interval: str,
        num_class: int = 1,
        batch_norm: bool = False
    ):
        """
        Initialize TableNet Module.

        Args:
            optimizer
            optimizer_params
            scheduler
            scheduler_params
            scheduler_interval (str):
            num_class (int): Number of classes per point.
            batch_norm (bool): Select VGG with or without batch normalization.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = TableNet(num_class, batch_norm)
        self.num_class = num_class
        self.dice_loss = DiceLoss()

        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.scheduler_interval = scheduler_interval

    def forward(self, batch):
        """
        Perform forward-pass.

        Args:
            batch (tensor): Batch of images to perform forward-pass.

        Returns (Tuple[tensor, tensor]): Table, Column prediction.
        """
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.

        Returns: Tensor
        """
        samples, labels_table, labels_column = batch
        output_table, output_column = self.forward(samples)

        loss_table = self.dice_loss(output_table, labels_table)
        loss_column = self.dice_loss(output_column, labels_column)

        self.log("train_loss_table", loss_table)
        self.log("train_loss_column", loss_column)
        self.log("train_loss", loss_column + loss_table)
        return loss_table + loss_column

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.

        Returns: Tensor
        """
        samples, labels_table, labels_column = batch
        output_table, output_column = self.forward(samples)

        loss_table = self.dice_loss(output_table, labels_table)
        loss_column = self.dice_loss(output_column, labels_column)

        self.log("valid_loss_table", loss_table, on_epoch=True)
        self.log("valid_loss_column", loss_column, on_epoch=True)
        self.log("validation_loss", loss_column + loss_table, on_epoch=True)
        self.log(
            "validation_iou_table",
            binary_mean_iou(output_table, labels_table),
            on_epoch=True,
        )
        self.log(
            "validation_iou_column",
            binary_mean_iou(output_column, labels_column),
            on_epoch=True,
        )
        return loss_table + loss_column

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.

        Returns: Tensor
        """
        samples, labels_table, labels_column = batch
        output_table, output_column = self.forward(samples)

        loss_table = self.dice_loss(output_table, labels_table)
        loss_column = self.dice_loss(output_column, labels_column)

        if batch_idx == 0:
            self._log_images(
                1,
                labels_table,
                labels_column,
                output_table,
                output_column,
            )

        self.log("test_loss_table", loss_table, on_epoch=True)
        self.log("test_loss_column", loss_column, on_epoch=True)
        self.log("test_loss", loss_column + loss_table, on_epoch=True)
        self.log(
            "test_iou_table",
            binary_mean_iou(output_table, labels_table),
            on_epoch=True,
        )
        self.log(
            "test_iou_column",
            binary_mean_iou(output_column, labels_column),
            on_epoch=True,
        )
        return loss_table + loss_column

    def configure_optimizers(self):
        """
        Configure optimizer for pytorch lighting.

        Returns: optimizer and scheduler for pytorch lighting.

        """
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)
        scheduler = self.scheduler(optimizer, **self.scheduler_params)
        scheduler = {
            "scheduler": scheduler,
            "monitor": "validation_loss",
            "interval": self.scheduler_interval
        }

        return [optimizer], [scheduler]

    def _log_images(
        self,
        n_images: int,
        labels_table: torch.Tensor,
        labels_column: torch.Tensor,
        output_table: torch.Tensor,
        output_column: torch.Tensor
    ):
        """
        Log images on to logger.

        Args:
            n_images (int): Number of images to log.
            labels_table (torch.Tensor): Batch of table labels.
            labels_column (torch.Tensor): Batch of column labels.
            output_table (torch.Tensor): Batch of table model outputs.
            output_column (torch.Tensor): Batch of column model outputs.
        """
        # TODO: clean up
        for i in range(min(len(labels_table), n_images)):
            image = Image.fromarray(255 * labels_table[i].squeeze().cpu().numpy().astype(np.uint8))
            file_name = f"table_label_{i}.png"
            image.save(file_name)
            mlflow.log_artifact(file_name, artifact_path="images")
            os.remove(file_name)

            image = Image.fromarray(255 * labels_column[i].squeeze().cpu().numpy().astype(np.uint8))
            file_name = f"column_label_{i}.png"
            image.save(file_name)
            mlflow.log_artifact(file_name, artifact_path="images")
            os.remove(file_name)

            image = Image.fromarray(255 * output_table[i].squeeze().cpu().numpy().astype(np.uint8))
            file_name = f"table_output_{i}.png"
            image.save(file_name)
            mlflow.log_artifact(file_name, artifact_path="images")
            os.remove(file_name)

            image = Image.fromarray(255 * output_column[i].squeeze().cpu().numpy().astype(np.uint8))
            file_name = f"column_output_{i}.png"
            image.save(file_name)
            mlflow.log_artifact(file_name, artifact_path="images")
            os.remove(file_name)


class TableNet(nn.Module):
    """
    TableNet.
    """

    def __init__(self, num_class: int, batch_norm: bool = False):
        """
        Initialize TableNet.

        Args:
            num_class (int): Number of classes per point.
            batch_norm (bool): Select VGG with or without batch normalization.
        """
        super().__init__()

        self.vgg = (
            vgg19(pretrained=True).features
            if not batch_norm
            else vgg19_bn(pretrained=True).features
        )
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.layers = [18, 27] if not batch_norm else [26, 39]
        self.model = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
        )
        self.table_decoder = TableDecoder(num_class)
        self.column_decoder = ColumnDecoder(num_class)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (tensor): Batch of images to perform forward-pass.

        Returns (Tuple[tensor, tensor]): Table, Column prediction.
        """
        results = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                results.append(x)
        x_table = self.table_decoder(x, results)
        x_column = self.column_decoder(x, results)
        return torch.sigmoid(x_table), torch.sigmoid(x_column)


class ColumnDecoder(nn.Module):
    """
    Column Decoder.
    """

    def __init__(self, num_classes: int):
        """
        Initialize Column Decoder.

        Args:
            num_classes (int): Number of classes per point.
        """
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.layer = nn.ConvTranspose2d(
            1280, num_classes, kernel_size=2, stride=2, dilation=1
        )

    def forward(self, x, pools):
        """
        Forward pass.

        Args:
            x (tensor): Batch of images to perform forward-pass.
            pools (Tuple[tensor, tensor]): The 3 and 4 pooling layer
                from VGG-19.

        Returns (tensor): Forward-pass result tensor.

        """
        pool_3, pool_4 = pools
        x = self.decoder(x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, pool_4], dim=1)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, pool_3], dim=1)
        x = F.interpolate(x, scale_factor=2)
        x = F.interpolate(x, scale_factor=2)
        return self.layer(x)


class TableDecoder(ColumnDecoder):
    """
    Table Decoder.
    """

    def __init__(self, num_classes):
        """
        Initialize Table decoder.

        Args:
            num_classes (int): Number of classes per point.
        """
        super().__init__(num_classes)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
        )


class DiceLoss(nn.Module):
    """
    Dice loss.
    """

    def __init__(self):
        """
        Dice Loss.
        """
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Calculate loss.

        Args:
            inputs (tensor): Output from the forward pass.
            targets (tensor): Labels.
            smooth (float): Value to smooth the loss.

        Returns (tensor): Dice loss.

        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (
            inputs.sum() + targets.sum() + smooth
        )

        return 1 - dice


def binary_mean_iou(inputs, targets):
    """Calculate binary mean intersection over union.

    Args:
        inputs (tensor): Output from the forward pass.
        targets (tensor): Labels.

    Returns (tensor): Intersection over union value.
    """
    output = (inputs > 0).int()
    if output.shape != targets.shape:
        targets = torch.squeeze(targets, 1)
    intersection = (targets * output).sum()
    union = targets.sum() + output.sum() - intersection
    result = (intersection + EPSILON) / (union + EPSILON)

    return result
