"""
Module d'annotation d'images pour l'apprentissage.
"""
import cv2
import json
import os
import tkinter
import yaml

from data import fs


class BoundingBoxWidget(object):
    def __init__(self, img):
        self.img = img
        self.img2 = self.img.copy()

        self.dim = self.img.shape

        self.image_coordinates = []
        self.image_coordinates_column = []

        self.column = False
        self.drawing = False
        self.last_coord = []

        self.color = (36, 255, 12)

    def extract_coordinates(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last_coord = (x, y)
            if not self.column:
                self.image_coordinates.append((x, y))
                cv2.rectangle(
                    self.img2, self.last_coord, (x, y), self.color, 2
                )
            else:
                self.image_coordinates_column.append((x, y))
                cv2.rectangle(
                    self.img2, self.last_coord, (x, y), self.color, 2
                )

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                a, b = x, y
                if a != x & b != y:
                    self.img = self.img2.copy()
                    cv2.rectangle(
                        self.img, self.last_coord, (a, b), self.color, 2
                    )

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if not self.column:
                self.image_coordinates.append((x, y))
                temp_coord = self.image_coordinates
            else:
                self.image_coordinates_column.append((x, y))
                temp_coord = self.image_coordinates_column

            print(
                "top left: {}, bottom right: {}".format(
                    temp_coord[-2], temp_coord[-1]
                )
            )

            cv2.rectangle(
                self.img2, temp_coord[-2], temp_coord[-1], self.color, 2
            )
            self.img2 = self.img.copy()

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.color = (0, 0, 205)
            self.column = True
            self.img2 = self.img.copy()


if __name__ == "__main__":
    CONFIG_FILE = "config/config.yaml"
    with open(CONFIG_FILE, "r") as stream:
        config = yaml.safe_load(stream)

    root = tkinter.Tk()
    screen_height = int(root.winfo_screenheight() * 0.9)

    with fs.open(file, "rb") as f:
        reader = pq.ParquetFile(f)
        iterator = reader.iter_batches(batch_size)
        for record_batch in iterator:
            yield record_batch.to_pandas()

    img_files = [
        f
        for f in fs.listdir(config.get("data_path"))
        if f.endswith(".png") or f.endswith(".bmp") or f.endswith(".jpg")
    ]
    for file in img_files:
        list_json = [
            f
            for f in fs.listdir(config.get("data_path"))
            if f.endswith(".json")
        ]
        list_img_tab = [
            f
            for f in fs.listdir(config.get("table_mask_path"))
            if f.endswith(".png") or f.endswith(".bmp") or f.endswith(".jpg")
        ]
        list_img_col = [
            f
            for f in fs.listdir(config.get("column_mask_path"))
            if f.endswith(".png") or f.endswith(".bmp") or f.endswith(".jpg")
        ]
        if (
            file[:-4] + ".json" not in list_json
            and file[:-4] + ".bmp" not in list_img_tab
            and file[:-4] + ".png" not in list_img_col
            and file[:-4] + ".bmp" not in list_img_col
            and file[:-4] + ".png" not in list_img_tab
        ):
            with fs.open(config.get("data_path") + tab, "r") as f:
                img_base = f.imread()
            ratio = img_base.shape[0] / img_base.shape[1]
            img_base = cv2.resize(
                img_base, (int(screen_height / ratio), screen_height)
            )
            bounding_box_widget = BoundingBoxWidget(img_base)
            cv2.namedWindow("Tables")
            cv2.setMouseCallback(
                "Tables", bounding_box_widget.extract_coordinates
            )
            stop_loop = True
            while stop_loop:
                cv2.imshow("Tables", bounding_box_widget.img)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    stop_loop = False
                    sav = {
                        "dim": bounding_box_widget.dim,
                        "tableaux": bounding_box_widget.image_coordinates,
                        "columns": bounding_box_widget.image_coordinates_column,
                    }
                    print(sav["dim"])
                    print(sav["tableaux"])
                    print(sav["columns"])
                    with fs.open(
                        config.get("data_path") + tab[:-4] + ".json", "w"
                    ) as a_file:
                        json.dump(sav, a_file)
                    cv2.destroyAllWindows()
