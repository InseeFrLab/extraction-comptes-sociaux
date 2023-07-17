"""Marmot Dataset Module."""

from pathlib import Path
from typing import List

import numpy as np
import pytorch_lightning as pl
from albumentations import Compose
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class MarmotDataset(Dataset):
    """Marmot Dataset."""

    def __init__(self, data: List[Path], transforms: Compose = None) -> None:
        """
        Marmot Dataset initialization.

        Args:
            data (List[Path]): A list of Path.
            transforms (Optional[Compose]): Compose object from albumentations.
        """
        self.data = data
        self.transforms = transforms

    def __len__(self):
        """
        Dataset Length.
        """
        return len(self.data)

    def __getitem__(self, item):
        """
        Get sample data.

        Args:
            item (int): sample id.

        Returns (Tuple[tensor, tensor, tensor]): Image, Table Mask, Column Mask
        """
        sample_id = self.data[item].stem

        image_path = self.data[item]
        table_path = self.data[item].parent.parent.joinpath(
            "table_mask", sample_id + ".bmp"
        )
        column_path = self.data[item].parent.parent.joinpath(
            "column_mask", sample_id + ".bmp"
        )

        image = np.array(Image.open(image_path))
        table_mask = np.expand_dims(np.array(Image.open(table_path)), axis=2)
        column_mask = np.expand_dims(np.array(Image.open(column_path)), axis=2)
        # Fix bug temporaire
        table_mask[table_mask > 0] = 255
        column_mask[column_mask > 0] = 255

        mask = np.concatenate([table_mask, column_mask], axis=2) / 255

        sample = {"image": image, "mask": mask}
        if self.transforms:
            sample = self.transforms(image=image, mask=mask)

        image = sample["image"]
        mask_table = sample["mask"][:, :, 0].unsqueeze(0)
        mask_column = sample["mask"][:, :, 1].unsqueeze(0)

        return image, mask_table, mask_column


class MarmotDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning Data Module for Marmot.
    """

    def __init__(
        self,
        train_data: List[Path],
        test_data: List[Path],
        transforms_preprocessing: Compose = None,
        transforms_augmentation: Compose = None,
        batch_size: int = 8,
        num_workers: int = 4,
    ):
        """
        Marmot Data Module initialization.

        Args:
            train_data (List[Path]): List of paths to training (and validation)
                images
            test_data (List[Path]): List of paths to test images
            transforms_preprocessing (Optional[Compose]): Compose object
                from albumentations applied on validation an test dataset.
            transforms_augmentation (Optional[Compose]): Compose object
                from albumentations applied on training dataset.
            batch_size (int): Define batch size.
            num_workers (int): Define number of workers to process data.
        """
        super().__init__()

        self.data = []
        # Removing horizontal images
        for path in train_data:
            image = np.array(Image.open(path))
            ratio = image.shape[0] / image.shape[1]
            if ratio > 1.2 and ratio < 1.6:
                self.data.append(path)

        self.test_data = test_data
        self.transforms_preprocessing = transforms_preprocessing
        self.transforms_augmentation = transforms_augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.setup()

    def setup(self, stage: str = None) -> None:
        """
        Start training, validation and test datasets.

        Args:
            stage (Optional[str]): Used to separate setup logic
                for trainer.fit and trainer.test.
        """
        n_samples = len(self.data)
        self.data.sort()
        train_slice = slice(0, int(n_samples * 0.8))
        val_slice = slice(int(n_samples * 0.8), n_samples)

        self.dataset_train = MarmotDataset(
            self.data[train_slice], transforms=self.transforms_augmentation
        )
        self.dataset_val = MarmotDataset(
            self.data[val_slice], transforms=self.transforms_preprocessing
        )
        self.dataset_test = MarmotDataset(
            self.test_data, transforms=self.transforms_preprocessing
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        """
        Create Dataloader.

        Returns: DataLoader
        """
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        """
        Create Dataloader.

        Returns: DataLoader
        """
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        """Create Dataloader.

        Returns: DataLoader
        """
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
