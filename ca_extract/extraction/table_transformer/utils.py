"""
Plot utils.
"""
from typing import List
from PIL import Image, ImageDraw, ImageOps
from matplotlib import pyplot as plt
from fitz import Rect
import fitz
import os
import s3fs


fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": "https://" + os.environ["AWS_S3_ENDPOINT"]},
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
)


def get_plot(image: Image, boxes: List[List[int]]):
    """
    Plot an image with bounding boxes.

    Args:
        image (Image): Image.
        boxes (List[List[int]]): Boxes.
    """
    image_copy = image.copy()
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        c1 = (xmin, ymin)
        c2 = (xmax, ymax)
        draw = ImageDraw.Draw(
            image_copy
        )
        draw.rectangle(
            (c1, c2),
            fill=None,
            outline="red",
            width=3
        )

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image_copy)

    return plt.gcf()


def pad_image(image: Image, dilation_factor: float = 0.02) -> Image:
    """
    Pad image.
    """
    color = "white"
    width, height = image.size
    border = (
        int(dilation_factor * height),
        int(dilation_factor * width),
        int(dilation_factor * height),
        int(dilation_factor * width),
    )

    return ImageOps.expand(image, border=border, fill=color)


def iob(bbox1, bbox2):
    """
    Compute the intersection area over box area, for bbox1.
    """
    intersection = Rect(bbox1).intersect(bbox2)

    bbox1_area = Rect(bbox1).get_area()
    if bbox1_area > 0:
        return intersection.get_area() / bbox1_area

    return 0


def load_pdf(pdf_path: str, s3: bool = True) -> fitz.Document:
    """
    Load pdf file from path.

    Args:
        pdf_path (str): Path to PDF file.
        s3 (bool): True if file is on s3.
    """
    if s3:
        # TODO : clean up
        fs.get(pdf_path, "tmp_pdf.pdf")
        doc = fitz.open("tmp_pdf.pdf")
        os.remove("tmp_pdf.pdf")
    else:
        doc = fitz.open(pdf_path)

    return doc
