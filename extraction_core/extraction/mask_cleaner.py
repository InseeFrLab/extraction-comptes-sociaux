"""
Code to clean masks.
"""
from typing import List, Dict
import numpy as np
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, convex_hull_image
from operator import itemgetter


class TableMaskCleaner:
    """
    """

    def __init__(
        self,
        table_closing_width: int = 1,
        min_surface_ratio_to_image: float = 0.005,
        rectangle_tables: bool = True,
    ):
        """
        """
        self.table_closing_width = table_closing_width
        self.min_surface_ratio_to_image = min_surface_ratio_to_image
        self.rectangle_tables = rectangle_tables

    def clean(self, mask: np.array) -> List[np.array]:
        """

        Args:
            mask (np.array): _description_

        Returns:
            np.array: _description_
        """
        segmented_mask = self.segment_table_mask(mask)
        mask_list = self.clean_segmented_mask(segmented_mask)
        return mask_list

    def segment_table_mask(self, mask: np.array) -> np.array:
        """
        Takes a binary table mask as input and returns a post-processed
        segmented mask (a mask with values in 0..n) where each closed set
        has a different value.

        Args:
            mask (np.array): binary mask.
        """
        thresh = threshold_otsu(mask)
        bw = closing(mask > thresh, square(self.table_closing_width))
        cleared = clear_border(bw)
        segmented_mask = label(cleared)
        return segmented_mask

    def clean_segmented_mask(self, segmented_mask: np.array) -> List[np.array]:
        """

        Args:
            segmented_mask (np.array): _description_

        Returns:
            np.array: _description_
        """
        width, height = segmented_mask.shape
        single_table_masks = []
        for i in np.unique(segmented_mask)[1:]:
            single_table_mask = np.where(segmented_mask == i, 1, 0)
            if single_table_mask.sum() > height * width * self.min_surface_ratio_to_image:
                if self.rectangle_tables:
                    single_table_masks.append(self.rectangularize_single_table_mask(single_table_mask))
                else:
                    single_table_masks.append(convex_hull_image(single_table_mask))

        return single_table_masks

    @staticmethod
    def rectangularize_single_table_mask(single_table_mask: np.array) -> np.array:
        """

        Args:
            self (_type_): _description_

        Returns:
            _type_: _description_
        """
        coords = np.column_stack(np.where(single_table_mask))
        x_min, x_max, y_min, y_max = (
            min(coords, key=itemgetter(0))[0],
            max(coords, key=itemgetter(0))[0],
            min(coords, key=itemgetter(1))[1],
            max(coords, key=itemgetter(1))[1],
        )

        single_rectangular_table_mask = np.zeros_like(single_table_mask)
        single_rectangular_table_mask[x_min:x_max, y_min:y_max] = 1
        return single_rectangular_table_mask

class ColumnMaskCleaner:
    """
    """

    def __init__(
        self,
        column_closing_width: int = 2,
        min_surface_ratio_to_table: float = 0.005,
        min_height_ratio_to_table: float = 0.4
    ):
        """
        """
        self.column_closing_width = column_closing_width
        self.min_surface_ratio_to_table = min_surface_ratio_to_table
        self.min_height_ratio_to_table = min_height_ratio_to_table

    def clean(self, mask: np.array, single_table_masks: List[np.array]) -> List[Dict]:
        """

        Args:
            mask (np.array): _description_

        Returns:
            np.array: _description_
        """
        segmented_columns_list = []
        for single_table_mask in single_table_masks:
            segmented_column_mask = self.segment_column_mask(mask, single_table_mask)
            segmented_columns = self.process_columns(
                segmented_column_mask
            )
            segmented_columns_list.append(segmented_columns)
        return segmented_columns_list

    def segment_column_mask(
        self, column_mask: np.array, single_table_mask: np.array
    ) -> np.array:
        """
        Takes a binary column mask and a table mask as inputs
        and returns a post-processed segmented mask (a mask with
        values in 0..n) where each closed set has a different value.

        Args:
            column_mask (np.array): binary column mask.
            single_table_mask (np.array): binary table mask.
        """
        intersection_mask = column_mask * single_table_mask
        thresh = threshold_otsu(intersection_mask)
        bw = closing(
            intersection_mask > thresh,
            square(self.column_closing_width),
        )

        cleared = clear_border(bw)

        table_height = cleared.sum(axis=0).max()
        new_cols = []
        for column in cleared.T:
            height = column.sum()
            if height / table_height < self.min_height_ratio_to_table:
                new_cols.append(np.zeros_like(column))
            else:
                # new_cols.append(column)
                new_cols.append(np.ones_like(column))

        sep_cleared = np.stack(new_cols, axis=1)
        reconnected_sep_cleared = sep_cleared * single_table_mask

        return label(reconnected_sep_cleared)

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
            if column.sum() > table_width * table_height * self.min_surface_ratio_to_table:
                position = regionprops(column)[0].centroid[1]
                cols[position] = column

        cols = OrderedDict(sorted(cols.items()))