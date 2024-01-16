"""
Transforms
"""
from PIL import Image


class MaxResize():
    def __init__(self, max_size: int = 800):
        self.max_size = max_size

    def __call__(self, image: Image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))

        return resized_image
