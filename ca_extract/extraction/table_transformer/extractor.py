"""
Extractor classes.
"""
from PIL import Image
from typing import List, Optional, Tuple, Dict
import abc
import numpy as np
import pandas as pd
from transformers import DetrFeatureExtractor, TableTransformerForObjectDetection
from .utils import get_plot
import torch
from .table_transformer_utils import (
    results_to_objects,
    objects_to_structures,
    structure_to_cells,
    cells_to_csv,
    structure_class_thresholds,
    str_class_idx2name,
)
from io import StringIO
from doctr.models import ocr_predictor


class Extractor(abc.ABC):
    """
    Extraction model base class.
    """

    def __init__(self, weights_path: Optional[str] = None):
        """
        Constructor.

        Args:
            weights_path (str): Path to model weights.
        """
        self.weights_path = weights_path
        self.model = self.load_model()

    @abc.abstractmethod
    def load_model(self):
        """
        Load model from weights.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def extract(self, crop: Image) -> pd.DataFrame:
        """
        Run inference on a pdf from a path.

        Args:
            crop (Image): Table crop.
        Returns:
            pd.DataFrame: Extracted table.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def ocr(self, image: Image) -> Tuple[List[Dict], List[List[int]]]:
        """
        OCR on an image.

        Args:
            image (Image): Image.

        Returns:
            Tuple[List[Dict], List[List[int]]]: Tokens and boxes
        """
        raise NotImplementedError


class TableTransformerExtractor(Extractor):

    def __init__(
        self,
        weights_path: Optional[str] = None,
    ):
        super(TableTransformerExtractor, self).__init__(weights_path)
        self.ocr_predictor = ocr_predictor(
            det_arch='db_resnet50',
            reco_arch='crnn_vgg16_bn',
            pretrained=True
        )

    def load_model(self):
        """
        Load model from weights.
        """
        self.feature_extractor = DetrFeatureExtractor()
        return TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition-v1.1-all"
        )

    def extract(self, crop: Image) -> pd.DataFrame:
        """
        Run inference on a pdf from a path.

        Args:
            crop (Image): Table crop.
        Returns:
            pd.DataFrame: Extracted table.
        """
        # Encoding for structure recognition
        encoding = self.feature_extractor(crop, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**encoding)

        target_sizes = [crop.size[::-1]]
        results = self.feature_extractor.post_process_object_detection(
            outputs, threshold=0.6, target_sizes=target_sizes
        )[0]

        # OCR on upright image
        tokens, boxes = self.ocr(crop)

        # Plot detected table
        fig = get_plot(crop, boxes)

        # 'tokens' is a list of tokens
        # Need to be in a relative reading order
        # If no order is provided, use current order
        for idx, token in enumerate(tokens):
            if "span_num" not in token:
                token["span_num"] = idx
            if "line_num" not in token:
                token["line_num"] = 0
            if "block_num" not in token:
                token["block_num"] = 0

        # Post-process detected objects, assign class labels
        objects = results_to_objects(
            results, crop.size, str_class_idx2name
        )

        # Further process the detected objects so they correspond
        # to a consistent table
        tables_structure = objects_to_structures(
            objects, tokens, structure_class_thresholds
        )

        # Enumerate all table cells: grid cells and
        # spanning cells
        table_cells = [
            structure_to_cells(structure, tokens)[0]
            for structure in tables_structure
        ]

        # Convert cells to CSV, including flattening multi-row column
        # headers to a single row
        table_csvs = [cells_to_csv(cells) for cells in table_cells]

        io = StringIO(table_csvs[0])
        df = pd.read_csv(io, sep=",")
        df = df.T.reset_index().T.reset_index(drop=True)

        # Case when no column headers were detected
        # Replace "Unnamed: x" by ""
        df.iloc[0] = df.iloc[0].str.replace(r"Unnamed: \d+", "")

        return df, fig

    def ocr(self, image: Image) -> Tuple[List[Dict], List[List[int]]]:
        """
        OCR on an image.

        Args:
            image (Image): Image.

        Returns:
            Tuple[List[Dict], List[List[int]]]: Tokens and boxes.
        """
        width, height = image.size
        tokens = []
        image_array = np.asarray(image)
        ocr_result = self.ocr_predictor([image_array])
        ocr_output = ocr_result.export()

        for block in ocr_output["pages"][0]["blocks"]:
            for line in block["lines"]:
                for word in line["words"]:
                    xmin, ymin = word["geometry"][0]
                    xmax, ymax = word["geometry"][1]
                    xmin = int(xmin * width)
                    ymin = int(ymin * height)
                    xmax = int(xmax * width)
                    ymax = int(ymax * height)
                    text = word["value"]
                    if text.replace(" ", "") in ["", "|"]:
                        continue
                    tokens.append(
                        {"bbox": [xmin, ymin, xmax, ymax], "text": text}
                    )

        boxes = [token["bbox"] for token in tokens]
        return tokens, boxes
