"""
Plot utils.
"""
from typing import List
from PIL import Image, ImageDraw, ImageOps
from matplotlib import pyplot as plt
from fitz import Rect


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
