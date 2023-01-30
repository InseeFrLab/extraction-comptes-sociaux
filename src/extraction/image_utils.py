"""
Utilitary functions to deal with images
"""
from PIL import Image
import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.measure import label


def to_bw(image: Image) -> Image:
    """
    Returns a black and white image from an input image.

    Args:
        image (Image): image considered.
    """
    thresh = threshold_otsu(
        np.expand_dims(np.array(image.convert("L")), axis=-1)
    )
    bw = np.expand_dims(np.array(image.convert("L")), axis=-1) > thresh
    bw_rgb = cv2.cvtColor(np.float32(bw), cv2.COLOR_GRAY2RGB)
    return Image.fromarray(np.uint8(bw_rgb * 255))


def invert_dark_areas(image: Image, threshold_area: float = 1e-3) -> Image:
    """
    Selects the dark areas (with an area of more than threshold_area
    of `image`) within `image` and inverts the colors there.

    Args:
        image (Image): image considered.
        threshold_area (float): area over which colors are inverted
    """
    thresh = threshold_otsu(
        np.expand_dims(np.array(image.convert("L")), axis=-1)
    )
    bw = np.expand_dims(np.array(image.convert("L")), axis=-1) > thresh
    # `cv2.findContours` detects white contours on a black background
    cs, _ = cv2.findContours(
        np.invert(bw).astype("uint8"),
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )

    # set up filled_image
    filled_image = np.zeros(bw.shape[0:2]).astype("uint8")
    image_area = bw.shape[0] * bw.shape[1]

    dark_areas_dummy = False
    for i, c in enumerate(cs):
        m = cv2.moments(c)
        area = m["m00"]

        if area / image_area > threshold_area:
            selected_contour = cv2.drawContours(
                filled_image, cs, i, color=255, thickness=-1
            )
            dark_areas_dummy = True

    if not dark_areas_dummy:
        return image
    mask = selected_contour == 255
    segmented_mask = label(mask)
    filtered_mask = np.zeros_like(mask)
    for i in np.unique(segmented_mask)[1:]:
        contour = np.where(segmented_mask == i, 1, 0)
        crop = bw[np.ix_(contour.any(1), contour.any(0))]
        if (
            len(crop[crop == 0])
            / (len(crop[crop == 0]) + len(crop[crop == 1]))
        ) > 0.5:
            filtered_mask = np.maximum(filtered_mask, contour)

    # modified_image = bw * (1 - np.expand_dims(filtered_mask, -1)) \
    #   + np.invert(bw) * np.expand_dims(filtered_mask, -1)
    modified_image = image * (
        1 - np.expand_dims(filtered_mask, -1)
    ) + np.invert(image) * np.expand_dims(filtered_mask, -1)
    return Image.fromarray(np.uint8(modified_image)).convert("RGB")
