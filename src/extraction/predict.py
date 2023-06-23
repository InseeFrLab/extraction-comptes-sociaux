"""Module de prédiction."""

import albumentations as album
from albumentations.pytorch.transforms import ToTensorV2
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import mlflow
import torch
import os

from .data import fs
from .table_extractor import TableExtractor
from .utils import get_root_path
import numpy as np


def main(args):
    """
    Main method.
    """
    model_name = "extraction"
    version = args.version
    clf = mlflow.pytorch.load_model(
        f"models:/{model_name}/{version}", map_location=torch.device("cpu")
    )

    table_extractor = TableExtractor(model=clf)

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

    for siren, image_path in zip(TEST_DATA, test_images):
        image = Image.open(image_path)
        out = table_extractor.extract(image)

        plt.rcParams["figure.figsize"] = (20, 10)
        plt.plot()
        plt.imshow(out["table_mask"], interpolation="none")
        plt.colorbar(orientation="vertical")
        plt.savefig(
            os.path.join(
                get_root_path(), "output/" + siren + "_table_mask.png"
            )
        )

        plt.rcParams["figure.figsize"] = (20, 10)
        plt.plot()
        plt.imshow(out["column_mask"], interpolation="none")
        plt.colorbar(orientation="vertical")
        plt.savefig(
            os.path.join(
                get_root_path(), "output/" + siren + "_column_mask.png"
            )
        )

        for i, df in enumerate(out["tables"]):
            save_path = os.path.join(
                get_root_path(),
                "output/" + siren + "_table_" + str(i) + ".csv",
            )
            df.to_csv(
                save_path,
                sep=";",
                header=False,
                index=False,
                index_label=False,
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--version", default=38)
    args = parser.parse_args()
    main(args)
