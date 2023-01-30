"""Module de prédiction."""

import albumentations as album
from albumentations.pytorch.transforms import ToTensorV2
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from argparse import ArgumentParser

from .data import fs
from .table_extractor import TableExtractor


def main(args):
    """
    Main method.
    """
    # Transformations applied to test data.
    transforms = album.Compose(
        [
            album.Resize(896, 896, always_apply=True),
            album.Normalize(),
            ToTensorV2(),
        ]
    )

    checkpoint_path = (
        "projet-extraction-tableaux/logs/TableNetModule/"
        + "version_"
        + "00"
        + "/checkpoints/"
    )
    table_extractor = None
    for checkpoints in fs.ls(checkpoint_path):
        if (
            Path(checkpoints).suffix == ".ckpt"
            and Path(checkpoints).name != "last.ckpt"
        ):
            table_extractor = TableExtractor(
                checkpoint_path=checkpoints, transforms=transforms
            )
            break

    TEST_DATA = [
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
    TEST_DATA = ["301940219"]
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
        plt.savefig("table_mask.png")

        plt.rcParams["figure.figsize"] = (20, 10)
        plt.plot()
        plt.imshow(out["column_mask"], interpolation="none")
        plt.colorbar(orientation="vertical")
        plt.savefig("column_mask.png")

        for i, df in enumerate(out["tables"]):
            save_path = (
                "projet-extraction-tableaux/logs/TableNetModule/"
                + "version_"
                + str(args.version)
                + "/test_output/"
                + siren
                + "_table_"
                + str(i)
                + ".csv"
            )
            with fs.open(save_path, "w") as f:
                df.to_csv(f, index=False, index_label=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--version", default=4)
    args = parser.parse_args()
    main(args)
