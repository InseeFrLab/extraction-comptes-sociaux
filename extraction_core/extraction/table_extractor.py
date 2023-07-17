"""
Table extractor.
"""
from __future__ import annotations
from collections import OrderedDict
from typing import Union, List, Dict, Tuple
import sys
import yaml

import torch
import numpy as np
import pandas as pd
import albumentations as album
from albumentations.pytorch.transforms import ToTensorV2
from matplotlib import pyplot as plt
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, convex_hull_image
import os
from cv2 import minAreaRect
from operator import itemgetter
from .column_extractor import ColumnExtractor
from .column_assembler import ColumnAssembler
from .utils import get_root_path
from .optimizers import optimizers
from .schedulers import schedulers
from .data import fs
from .tablenet import TableNetModule, LegacyTableNetModule

np.set_printoptions(threshold=sys.maxsize)


default_transforms = album.Compose(
    [
        album.Resize(896, 896, always_apply=True),
        album.Normalize(),
        ToTensorV2(),
    ]
)


class TableExtractor:
    """
    Table extractor which creates pd.DataFrame objects
    containing table information from images.
    """

    def __init__(
        self,
        model: TableNetModule,
        transforms: album.Compose = default_transforms,
        threshold: float = 0.5,
        per: float = 0.005,
        further_process_table_masks: bool = False,
        table_closing_width: int = 1,
        column_closing_width: int = 2,
        column_threshold: float = 0.4,
        column_extractor: ColumnExtractor = ColumnExtractor(),
        column_assembler: ColumnAssembler = ColumnAssembler(),
        weights_path: str = "./weights.ckpt",
    ):
        """
        Constructor for the TableExtractor class.

        Args:
            model (TableNetModule): TableNet model used to predict table
                and column masks.
            transforms (album.Compose): transformations for test data.
            threshold (float): Threshold above which a pixel is considered
                part of the table or column mask.
            per (float): Percentage of pixels above which a closed set of
                pixels is kept in the table or column mask.
        """
        self.transforms = transforms
        self.threshold = threshold
        self.per = per
        self.further_process_table_masks = further_process_table_masks
        self.table_closing_width = table_closing_width
        self.column_closing_width = column_closing_width
        self.column_threshold = column_threshold

        self.column_extractor = column_extractor
        self.column_assembler = column_assembler

        self.model = model
        print(model)
        self.model.eval()
        self.model.requires_grad_(False)

    @staticmethod
    def from_checkpoint(
        checkpoint_path: str,
        transforms: album.Compose = default_transforms,
        threshold: float = 0.5,
        per: float = 0.005,
        further_process_table_masks: bool = False,
        table_closing_width: int = 1,
        column_closing_width: int = 2,
        column_threshold: float = 0.4,
        column_extractor: ColumnExtractor = ColumnExtractor(),
        column_assembler: ColumnAssembler = ColumnAssembler(),
        on_s3=True,
        weights_path: str = "./weights.ckpt",
    ) -> TableExtractor:
        """
        Constructor for the TableExtractor class.

        Args:
            checkpoint_path (str): Path to the checkpoint file of the TableNet
                model used to predict table and column masks.
            transforms (album.Compose): transformations for test data.
            threshold (float): Threshold above which a pixel is considered
                part of the table or column mask.
            per (float): Percentage of pixels above which a closed set of
                pixels is kept in the table or column mask.
        """
        if on_s3:
            fs.get(checkpoint_path, "./weights.ckpt")
            # model = TableNetModule.load_from_checkpoint("./weights.ckpt")
            model = LegacyTableNetModule.load_from_checkpoint("./weights.ckpt")
            os.remove("./weights.ckpt")
        else:
            model = TableNetModule.load_from_checkpoint(checkpoint_path)

        return TableExtractor(
            model,
            transforms,
            threshold,
            per,
            further_process_table_masks,
            table_closing_width,
            column_closing_width,
            column_threshold,
            column_extractor,
            column_assembler,
            weights_path,
        )

    def get_raw_masks(self, image: Image) -> Tuple[np.array, np.array]:
        """
        Takes an image as input and returns the raw table and column masks.

        Args:
            image: image from which to get masks.
        """
        processed_image = self.transforms(image=np.array(image))["image"]
        table_mask, column_mask = self.model.forward(
            processed_image.unsqueeze(0)
        )
        table_mask = self.apply_threshold(table_mask)
        column_mask = self.apply_threshold(column_mask)
        return table_mask, column_mask

    def extract(
        self,
        image: Image,
        invert_dark_areas: bool = False,
        on_bw: bool = False,
        post_processing: bool = True,
    ) -> Dict[str, Union[List[pd.DataFrame], np.array, np.array]]:
        """
        Takes an image as input and returns a dictionary with keys
        tables (containing a list of pd.DataFrame objects, one for
        each table on the image), table_mask (containing the table
        mask) and column_mask (containing the column mask).

        Args:
            image (Image): image for which to extract tables.
            invert_dark_area (bool): if True, inverts the colors
                of the largest "dark" area before using Tesseract to
            extract text.
            on_bw (bool): if True, text extraction is performed on
                a black and white image.
            post_processing (bool): if False, stops after model outputs
                masks and returns them.
        """
        raw_table_mask, raw_column_mask = self.get_raw_masks(image)

        if not post_processing:
            return {
                "tables": [],
                "table_mask": raw_table_mask,
                "column_mask": raw_column_mask,
            }

        # First plot
        # plt.rcParams["figure.figsize"] = (20, 10)
        # plt.plot()
        # plt.imshow(image.resize((896, 896)), interpolation="none")
        # plt.imshow(raw_column_mask, interpolation="none", alpha=0.5)
        # plt.savefig(
        #     os.path.join(get_root_path(), "output/raw_column_masks.png")
        # )
        # plt.clf()

        segmented_tables = self.process_tables(
            self.segment_table_mask(raw_table_mask)
        )

        tables = []
        for table in segmented_tables:
            segmented_columns = self.process_columns(
                self.segment_column_mask(raw_column_mask, table)
            )
            if segmented_columns:
                cols_array = np.zeros_like(raw_column_mask)
                for i, (idx, column) in enumerate(segmented_columns.items()):
                    cols_array += column
                # Plot post-treated masks
                # plt.rcParams["figure.figsize"] = (20, 10)
                # plt.plot()
                # plt.imshow(image.resize((896, 896)), interpolation="none")
                # plt.imshow(cols_array, interpolation="none", alpha=0.5)
                # plt.colorbar(orientation="vertical")
                # plt.savefig(
                #     os.path.join(get_root_path(), "output/column_masks.png")
                # )
                # plt.clf()

                if invert_dark_areas:
                    ocr_image = self.invert_dark_areas(image)
                else:
                    ocr_image = image
                if on_bw:
                    ocr_image = self.to_bw(ocr_image)
                cols = self.column_extractor.extract_columns(
                    segmented_columns, ocr_image
                )
                tables.append(self.column_assembler.assemble(cols))
        return_dict = {
            "tables": tables,
            "table_mask": raw_table_mask,
            "column_mask": raw_column_mask,
        }
        return return_dict

    def apply_threshold(self, mask: torch.Tensor) -> np.array:
        """
        Takes a mask Tensor containing probabilities as input
        and returns a binary mask.

        Args:
            mask (torch.Tensor): probabilities mask output of the model.
        """
        mask = mask.squeeze(0).squeeze(0).numpy() > self.threshold
        return mask.astype(int)

    def process_tables(self, segmented_tables: np.array) -> List[np.array]:
        """
        Takes a segmented table mask as input and return a list of
        post-processed table masks.

        Args:
            segmented_tables (np.array): probabilities mask output of
                the model.
        """
        width, height = segmented_tables.shape
        tables = []
        for i in np.unique(segmented_tables)[1:]:
            table = np.where(segmented_tables == i, 1, 0)
            if table.sum() > height * width * self.per:
                if self.further_process_table_masks:
                    tables.append(self.further_process_table(table))
                else:
                    tables.append(convex_hull_image(table))

        return tables

    @staticmethod
    def further_process_table(table: np.array) -> np.array:
        """
        Takes a table mask as input and returns a post-processed
        table mask. Post-processing here consists in returning a
        rectangle mask which covers the original mask.

        Args:
            table (np.array): table mask.
        """
        coords = np.column_stack(np.where(table))
        # table_width, table_height = minAreaRect(coords)[1]
        x_min, x_max, y_min, y_max = (
            min(coords, key=itemgetter(0))[0],
            max(coords, key=itemgetter(0))[0],
            min(coords, key=itemgetter(1))[1],
            max(coords, key=itemgetter(1))[1],
        )
        processed_table = np.zeros_like(table)
        processed_table[x_min:x_max, y_min:y_max] = 1
        return processed_table

    def process_columns(
        self, segmented_columns: np.array
    ) -> Dict[int, np.array]:
        """
        Takes a segmented column mask as input and return a dictionary
        of post-processed column masks.
        TODO: return type to adjust ?

        Args:
            segmented_columns (np.array): probabilities mask output
                of the model.
        """
        table_width = np.where(segmented_columns > 0, 1, 0).sum(axis=1).max()
        table_height = np.where(segmented_columns > 0, 1, 0).sum(axis=0).max()
        cols = {}

        for j in np.unique(segmented_columns)[1:]:
            column = np.where(segmented_columns == j, 1, 0)
            column = column.astype(int)
            if column.sum() > table_width * table_height * self.per:
                position = regionprops(column)[0].centroid[1]
                cols[position] = column
        return OrderedDict(sorted(cols.items()))

    def segment_table_mask(self, table_mask: np.array) -> np.array:
        """
        Takes a binary table mask as input and returns a post-processed
        segmented mask (a mask with values in 0..n) where each closed set
        has a different value.

        Args:
            table_mask (np.array): binary mask.
        """
        thresh = threshold_otsu(table_mask)
        bw = closing(table_mask > thresh, square(self.table_closing_width))
        cleared = clear_border(bw)
        segmented_mask = label(cleared)
        return segmented_mask

    def segment_column_mask(
        self, column_mask: np.array, table_mask: np.array
    ) -> np.array:
        """
        Takes a binary column mask and a table mask as inputs
        and returns a post-processed segmented mask (a mask with
        values in 0..n) where each closed set has a different value.

        Args:
            column_mask (np.array): binary column mask.
            table_mask (np.array): binary table mask.
        """
        thresh = threshold_otsu(column_mask * table_mask)
        bw = closing(
            column_mask * table_mask > thresh,
            square(self.column_closing_width),
        )
        cleared = clear_border(bw)

        table_height = cleared.sum(axis=0).max()
        new_cols = []
        for column in cleared.T:
            height = column.sum()
            if height / table_height < self.column_threshold:
                new_cols.append(np.zeros_like(column))
            else:
                # new_cols.append(column)
                new_cols.append(np.ones_like(column))

        sep_cleared = np.stack(new_cols, axis=1)
        reconnected_sep_cleared = sep_cleared * table_mask

        return label(reconnected_sep_cleared)
