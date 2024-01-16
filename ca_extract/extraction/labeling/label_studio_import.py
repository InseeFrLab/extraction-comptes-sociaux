import xml.etree.ElementTree as ET
from PIL import Image
import os
import numpy as np
from ..data import fs


def get_image(path: str) -> Image:
    """
    Get image on s3 corresponding to path and load it.

    Args:
        path (str): Image path.

    Returns:
        Image: Image.
    """
    fs.get(path, "tmp.bmp")
    image = Image.open("tmp.bmp")
    os.remove("tmp.bmp")
    return image


def read_label_studio_export(xml_file: str):
    """
    Read a Pascal VOC format XML file and return table
    and column masks.

    Args:
        xml_file (str): xml file path.

    Returns:
        _type_: _description_
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    labels = []

    for boxes in root.iter("object"):
        name = root.find("filename").text
        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        label = boxes.find("name").text

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        labels.append({"label": label, "bb": list_with_single_boxes})

    table_labels = [label for label in labels if label["label"] == "table"]
    column_labels = [label for label in labels if label["label"] == "column"]

    path = "projet-extraction-tableaux/app_data/bmp/" + name
    image = get_image(path)
    shape = image.size
    width, height = shape

    table_mask = np.zeros((height, width))
    column_mask = np.zeros((height, width))
    for label in table_labels:
        bb = label["bb"]
        table_mask[bb[1] : bb[3], bb[0] : bb[2]] = 1
    for label in column_labels:
        bb = label["bb"]
        column_mask[bb[1] : bb[3], bb[0] : bb[2]] = 1

    return table_mask, column_mask


if __name__ == "__main__":
    table_mask, column_mask = read_label_studio_export(
        "/home/onyxia/work/322804147_2020.xml"
    )
    print(table_mask)
    print(column_mask)
