"""Module de création des masques"""

import json
import os
import cv2
import numpy as np

from data import fs


def make_mask(path, mask, name_png):
    dim = mask["dim"]
    if path == "data/column_mask/":
        contour_mask = mask["columns"]
    else:
        contour_mask = mask["tableaux"]

    white_color = (255, 255, 255)

    img = np.zeros(dim, np.uint8)

    for i in range(0, len(contour_mask), 2):
        cv2.rectangle(
            img, contour_mask[i], contour_mask[i + 1], white_color, cv2.FILLED
        )

    cv2.imwrite(path + name_png, img)


if __name__ == "__main__":
    CONFIG_FILE = "config/config.yaml"
    with open(CONFIG_FILE, "r") as stream:
        config = yaml.safe_load(stream)

    list_json = [
        f for f in fs.listdir(config.get("data_path")) if f.endswith(".json")
    ]
    for temp_json in list_json:
        path_column = config.get("column_mask_path")
        list_mask_img_column = [
            f
            for f in fs.listdir(path_column)
            if f.endswith(".png") or f.endswith(".bmp") or f.endswith(".jpg")
        ]

        if temp_json[:-5] + ".bmp" not in list_mask_img_column:
            with fs.open(config.get("data_path") + temp_json) as f:
                data_mask_column = json.load(f)
            make_mask(path_column, data_mask_column, temp_json[:-5] + ".bmp")

        path_table = config.get("table_mask_path")
        list_mask_img_table = [
            f
            for f in fs.listdir(path_table)
            if f.endswith(".png") or f.endswith(".bmp") or f.endswith(".jpg")
        ]
        if temp_json[:-5] + ".bmp" not in list_mask_img_table:
            with fs.open(config.get("data_path") + temp_json) as f:
                data_mask_table = json.load(f)
            make_mask(path_table, data_mask_table, temp_json[:-5] + ".bmp")
