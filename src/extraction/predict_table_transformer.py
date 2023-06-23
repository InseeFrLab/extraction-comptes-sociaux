"""Module de prédiction."""
from pathlib import Path
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import DetrFeatureExtractor
from transformers import TableTransformerForObjectDetection
import torch
import pytesseract
from pytesseract import Output
from .table_transformer_utils import *
from io import StringIO
import os
import pandas as pd
from .utils import get_root_path


def main():
    """
    Main method.
    """
    TEST_DATA = [
        # "305756413",
        # "324084698",
        # "326300159",
        # "331154765",
        # "333916385",
        # "334303823",
        # "344066733",
        # "393525852",
        # "393712286",
        # "411787567",
        # "414728337",
        # "552065187",
        # "552081317",
        # "702012956",
        # "797080850",
    ]
    TEST_DATA = [
        "342360005",
        "343009866",
        "343406732",
        "344066733",
        "347951238",
        "350693529",
        "351279641",
        "379134166",
        "379196181",
        "383474814",
        "380129866",
    ]
    data_dir = "./data/marmot_data"
    test_images = [
        Path(data_dir).joinpath(path + ".bmp") for path in TEST_DATA
    ]

    feature_extractor = DetrFeatureExtractor()
    detection_model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection"
    )
    structure_model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-structure-recognition"
    )

    # Paddings
    left_padding = 20
    top_padding = 50
    right_padding = 20
    bottom_padding = 50

    for siren, image_path in zip(TEST_DATA, test_images):
        image = Image.open(image_path)
        width, height = image.size

        # Encoding for table detection
        encoding = feature_extractor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = detection_model(**encoding)
        results = feature_extractor.post_process_object_detection(
            outputs, threshold=0.7, target_sizes=[(height, width)]
        )[0]
        table_boxes = results["boxes"].tolist()

        for table_idx, table_box in enumerate(table_boxes):
            xmin, ymin, xmax, ymax = table_box
            # Cropped image (only detected table)
            resized_image = image.crop(
                (
                    xmin - right_padding,
                    ymin - top_padding,
                    xmax + right_padding,
                    ymax + bottom_padding,
                )
            )

            # Encoding for structure recognition
            encoding = feature_extractor(resized_image, return_tensors="pt")
            with torch.no_grad():
                outputs = structure_model(**encoding)

            target_sizes = [resized_image.size[::-1]]
            results = feature_extractor.post_process_object_detection(
                outputs, threshold=0.6, target_sizes=target_sizes
            )[0]

            # Getting tokens
            d = pytesseract.image_to_data(
                resized_image, output_type=Output.DICT
            )
            tokens = []
            n_boxes = len(d["level"])
            for i in range(n_boxes):
                (xmin, ymin, w, h) = (
                    d["left"][i],
                    d["top"][i],
                    d["width"][i],
                    d["height"][i],
                )
                xmax = xmin + w
                ymax = ymin + h
                text = d["text"][i]
                tokens.append({"bbox": [xmin, ymin, xmax, ymax], "text": text})
            # 'tokens' is a list of tokens
            # Need to be in a relative reading order
            # If no order is provided, use current order
            for idx, token in enumerate(tokens):
                if not "span_num" in token:
                    token["span_num"] = idx
                if not "line_num" in token:
                    token["line_num"] = 0
                if not "block_num" in token:
                    token["block_num"] = 0

            # Post-process detected objects, assign class labels
            objects = results_to_objects(
                results, resized_image.size, str_class_idx2name
            )

            # Further process the detected objects so they correspond to a consistent table
            tables_structure = objects_to_structures(
                objects, tokens, structure_class_thresholds
            )

            # Enumerate all table cells: grid cells and spanning cells
            table_cells = [
                structure_to_cells(structure, tokens)[0]
                for structure in tables_structure
            ]

            # Convert cells to CSV, including flattening multi-row column headers to a single row
            table_csvs = [cells_to_csv(cells) for cells in table_cells]

            io = StringIO(table_csvs[0])
            df = pd.read_csv(io, sep=",")
            save_path = os.path.join(
                get_root_path(),
                "output/tt_" + siren + "_table_" + str(table_idx) + ".csv",
            )
            df.to_csv(save_path, sep=";", index=False)


if __name__ == "__main__":
    main()
