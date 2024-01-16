from typing import Dict, List
from PIL import Image
import pandas as pd
import numpy as np
import os
from skimage.morphology import square, convex_hull_image, binary_dilation
from skimage.transform import resize
from skimage.util import invert
from pytesseract import image_to_string, image_to_data
from matplotlib import pyplot as plt
from .utils import get_root_path


class ColumnExtractor:
    """
    Extractor of the content of columns.
    """

    def __init__(
        self,
        with_loc: str = True,
        readjust: str = True,
        readjust_dilation_diameter: int = 10,
        final_dilation_diameter: int = 10,
        config: str = "--oem 1 --psm 4",
    ):
        """
        Constructor for ColumnExtractor.
        """
        self.with_loc = with_loc
        self.readjust = readjust
        self.readjust_dilation_diameter = readjust_dilation_diameter
        self.final_dilation_diameter = final_dilation_diameter
        self.config = config

    def extract_columns(
        self, column_masks: Dict, image: Image
    ) -> List[pd.DataFrame]:
        """
        Extract multiple columns.

        Args:
            column_masks (Dict): dictionary of column masks.
            image (Image): image.

        Returns:
            List[pd.DataFrame]: list of pd.DataFrame objects
                containing column contents.
        """
        extractions = []

        if self.readjust:
            readjusted_column_masks = self.readjust_column_masks(
                column_masks, image
            )
        else:
            readjusted_column_masks = column_masks

        cols_array = np.zeros_like(list(readjusted_column_masks.values())[0])
        for i, (idx, column) in enumerate(readjusted_column_masks.items()):
            cols_array += (i + 1) * column
        # Plot post-treated masks
        plt.rcParams["figure.figsize"] = (20, 10)
        plt.plot()
        plt.imshow(image.resize((896, 896)), interpolation="none")
        plt.imshow(cols_array, interpolation="none", alpha=0.5)
        plt.colorbar(orientation="vertical")
        plt.savefig(
            os.path.join(get_root_path(), "output/readj_column_masks.png")
        )
        plt.clf()

        for colname, column_mask in readjusted_column_masks.items():
            extractions.append(self.extract_column(column_mask, image))
        return extractions

    def extract_column(
        self, column_mask: np.array, image: Image
    ) -> pd.DataFrame:
        """
        Extracts a single column.

        Args:
            column_mask (np.array): column mask.
            image (Image): image.
        """
        if self.with_loc:
            return self.column_to_dataframe_with_loc(column_mask, image)
        return self.column_to_dataframe(column_mask, image)

    def readjust_column_masks(self, column_masks: Dict, image: Image) -> Dict:
        """
        Readjusts column masks, so that there is no space between columns.

        Args:
            column_mask (np.array): dictionary of column masks.
            image (Image): image.
        """
        readjusted_masks = {}

        min_ys = []
        max_ys = []
        min_xs = []
        max_xs = []

        width, height = image.size
        for colname, column_mask in column_masks.items():
            hull = convex_hull_image(column_mask)
            scaled_hull = binary_dilation(
                hull, square(self.readjust_dilation_diameter)
            )

            [rows, columns] = np.where(scaled_hull)
            min_ys.append(min(rows))
            max_ys.append(max(rows))
            min_xs.append(min(columns))
            max_xs.append(max(columns))

        min_y = min(min_ys)
        max_y = max(max_ys)

        for index, (colname, column_mask) in enumerate(column_masks.items()):
            if index != 0:
                min_x = int((min_xs[index] + max_xs[index - 1]) / 2)
            else:
                min_x = min_xs[index]

            if index != len(min_ys) - 1:
                max_x = int((max_xs[index] + min_xs[index + 1]) / 2)
            else:
                max_x = max_xs[index]

            mask = np.zeros_like(column_mask)
            mask[min_y:max_y, min_x:max_x] = 1
            readjusted_masks[colname] = mask

        return readjusted_masks

    def column_to_dataframe(
        self, column_mask: np.array, image: Image
    ) -> pd.DataFrame:
        """
        Takes a column mask and an image as inputs and returns a
        pd.DataFrame with OCRd text from the masked image.

        Args:
            column_mask (np.array): column mask.
            image (Image): image from where to extract text.
        """
        width, height = image.size
        column_mask = (
            resize(
                np.expand_dims(column_mask, axis=2),
                (height, width),
                preserve_range=True,
            )
            > 0.01
        )

        crop = column_mask * image
        white = np.ones(column_mask.shape) * invert(column_mask) * 255
        crop = crop + white
        ocr = image_to_string(
            Image.fromarray(crop.astype(np.uint8)), lang="fra"
        )
        return pd.DataFrame(
            {"col": [value for value in ocr.split("\n") if len(value) > 0]}
        )

    def column_to_dataframe_with_loc_legacy(
        self, column_mask: np.array, image: Image
    ) -> pd.DataFrame:
        """
        Extract column information to a DataFrame also containing coordinates
        for text in the column.

        Args:
            column_mask (np.array): column mask.
            image (Image): image from where to extract text.
        """
        width, height = image.size

        # Final post-processing to adjust
        hull = convex_hull_image(column_mask)
        scaled_hull = binary_dilation(
            hull, square(self.final_dilation_diameter)
        )

        column = (
            resize(
                np.expand_dims(scaled_hull, axis=2),
                (height, width),
                preserve_range=True,
            )
            > 0.01
        )
        [rows, columns, channels] = np.where(column)
        row1 = min(rows)
        row2 = max(rows)
        col1 = min(columns)
        col2 = max(columns)

        crop = column * image
        white = np.ones(column.shape, dtype="uint8") * invert(column) * 255
        crop = crop + white
        cropped_image = crop[row1:row2, col1:col2]

        # Plot
        plt.rcParams["figure.figsize"] = (20, 10)
        plt.plot()
        plt.imshow(cropped_image, interpolation="none")
        plt.savefig(os.path.join(get_root_path(), "output/crop.png"))
        plt.clf()

        ocr_string = image_to_string(
            Image.fromarray(cropped_image.astype(np.uint8)),
            lang="fra",
            config=self.config,
        )
        ocr_string_df = pd.DataFrame(
            {
                "col": [
                    value
                    for value in ocr_string.split("\n")
                    if len(value) > 0 and value not in ["\x0c", "", " "]
                ]
            }
        )
        ocr_data_df = image_to_data(
            Image.fromarray(cropped_image.astype(np.uint8)),
            lang="fra",
            config=self.config,
            output_type="data.frame",
        )

        tops = []
        for row in ocr_string_df.col:
            first_word = row.split(" ", 1)[0]
            index = ocr_data_df.text.eq(first_word).idxmax()
            tops.append(ocr_data_df.iloc[index].top + row1)
            ocr_data_df = ocr_data_df.iloc[index + 1 :].reset_index(drop=True)

        ocr_string_df["top"] = tops
        return ocr_string_df

    def column_to_dataframe_with_loc(
        self, column_mask: np.array, image: Image
    ) -> pd.DataFrame:
        """
        Extract column information to a DataFrame also containing coordinates
        for text in the column.

        Args:
            column_mask (np.array): column mask.
            image (Image): image from where to extract text.
        """
        width, height = image.size

        # Final post-processing to adjust
        hull = convex_hull_image(column_mask)
        scaled_hull = binary_dilation(
            hull, square(self.final_dilation_diameter)
        )

        column = (
            resize(
                np.expand_dims(scaled_hull, axis=2),
                (height, width),
                preserve_range=True,
            )
            > 0.01
        )
        [rows, columns, channels] = np.where(column)
        col1 = min(rows)
        col2 = max(rows)
        row1 = min(columns)
        row2 = max(columns)

        # OCR of entire page
        # Should not be done n times
        ocr_data_df = image_to_data(
            image,
            lang="fra",
            config=self.config,
            output_type="data.frame",
        )
        ocr_data_df["pct_bb_column"] = ocr_data_df.apply(
            lambda row: self.get_iou(
                {
                    "x1": row["left"],
                    "x2": row["left"] + row["width"],
                    "y1": row["top"],
                    "y2": row["top"] + row["height"],
                },
                {"x1": row1, "x2": row2, "y1": col1, "y2": col2},
            ),
            axis=1,
        )

        # First we want to filter bounding boxes
        ocr_data_df["text"] = ocr_data_df["text"].replace("", np.nan)
        ocr_data_df = ocr_data_df.dropna(subset=["text"])
        filtered_df = ocr_data_df[
            ocr_data_df["pct_bb_column"] > 0.5
        ]  # Tunable parameter

        # Then we want to recompose lines
        filtered_df_indexed = filtered_df.copy()
        filtered_df_indexed["line"] = (
            filtered_df["block_num"] - 1
        ) * filtered_df["line_num"] + filtered_df["line_num"]
        filtered_df_indexed = filtered_df_indexed.groupby("line").agg(
            {"top": "min", "text": lambda x: " ".join(x)}
        )
        filtered_df_indexed = filtered_df_indexed.rename(
            columns={"text": "col"}
        )[["col", "top"]].sort_values("top")
        return filtered_df_indexed

    @staticmethod
    def get_iou(bb1: Dict, bb2: Dict):
        """
        Calculate the percentage of area of bb1 contained in bb2.

        Args:
            bb1 (Dict):
                Keys: {'x1', 'x2', 'y1', 'y2'}
                The (x1, y1) position is at the top left corner,
                the (x2, y2) position is at the bottom right corner
            bb2 (Dict):
                Keys: {'x1', 'x2', 'y1', 'y2'}
                The (x, y) position is at the top left corner,
                the (x2, y2) position is at the bottom right corner

        Returns:
            float in [0, 1]
        """
        assert bb1["x1"] < bb1["x2"]
        assert bb1["y1"] < bb1["y2"]
        assert bb2["x1"] < bb2["x2"]
        assert bb2["y1"] < bb2["y2"]

        # Determine the coordinates of the intersection rectangle
        x_left = max(bb1["x1"], bb2["x1"])
        y_top = max(bb1["y1"], bb2["y1"])
        x_right = min(bb1["x2"], bb2["x2"])
        y_bottom = min(bb1["y2"], bb2["y2"])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Compute the area of bb1
        bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])

        # Compute what we are interested in
        output = intersection_area / bb1_area
        assert output >= 0.0
        assert output <= 1.0
        return output
