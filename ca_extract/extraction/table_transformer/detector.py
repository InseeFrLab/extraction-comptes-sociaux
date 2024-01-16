"""
Detector classes.
"""
from PIL import Image, ImageOps
from typing import List, Optional, Tuple
import abc
from .transforms import MaxResize
from torchvision import transforms
from transformers import AutoModelForObjectDetection
from .utils import pad_image, iob
import torch
from utils import load_pdf
import fitz


class Detector(abc.ABC):
    """
    Extraction model base class.
    """

    def __init__(self, weights_path: Optional[str] = None):
        """
        Constructor.

        Args:
            weights_path (str): Path to model weights.
        """
        self.weights_path = weights_path
        self.model = self.load_model()

    @abc.abstractmethod
    def load_model(self):
        """
        Load model from weights.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def detect(self, pdf_path: str, s3: bool) -> List:
        """
        Run inference on a pdf from a path.

        Args:
            weights_path (str): Path to pdf.
            s3 (bool): Is the pdf on https://minio.lab.sspcloud.fr ?
        Returns:
            List: List of cropped table images.
        """
        raise NotImplementedError()


class TableTransformerDetector(Detector):
    """
    TableTransformerDetector.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        padding_factor: float = 1.0,
        crop_padding_factor: float = 1.0,
    ):
        super(TableTransformerDetector, self).__init__(weights_path)
        self.padding_factor = padding_factor
        self.crop_padding_factor = crop_padding_factor

    def load_model(self):
        """
        Load model from weights.
        """
        self.detection_transform = transforms.Compose(
            [
                MaxResize(800),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        return AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection", revision="no_timm"
        )

    @staticmethod
    def box_cxcywh_to_xyxy(x: Tuple[int]) -> torch.Tensor:
        """
        Reformat bounding boxes.

        Args:
            x (Tuple[int]): Original bounding boxes coordinates.

        Returns:
            torch.Tensor: Reformated coordinates.
        """
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox: Tuple[int], size: Tuple[int]) -> torch.Tensor:
        """
        Rescale bounding box to change format.

        Args:
            out_bbox (Tuple[int]): Bounding box with original format.
            size (Tuple[int]): Image size.

        Returns:
            torch.Tensor: Rescaled box.
        """
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def outputs_to_objects(self, outputs, img_size, id2label):
        """
        First post-processing of the detection model output to
        obtain dict objects containing label, score and bbox keys.

        Args:
            outputs: Detection model output.
            img_size (Tuple[int]): Image size.
            id2label (Dict): Id to label.

        Returns:
            List: List of "objects".
        """
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
        pred_bboxes = [
            elem.tolist() for elem in self.rescale_bboxes(pred_bboxes, img_size)
        ]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = id2label[int(label)]
            if not class_label == "no object":
                objects.append(
                    {
                        "label": class_label,
                        "score": float(score),
                        "bbox": [float(elem) for elem in bbox],
                    }
                )

        return objects

    @staticmethod
    def objects_to_crops(img, tokens, objects, class_thresholds, padding):
        """
        Process the bounding boxes dictionary objects produced by the table
        detection model into cropped table images and cropped tokens.
        """
        table_crops = []
        for obj in objects:
            if obj["score"] < class_thresholds[obj["label"]]:
                continue

            cropped_table = {}

            bbox = obj["bbox"]
            bbox = [
                bbox[0] - padding,
                bbox[1] - padding,
                bbox[2] + padding,
                bbox[3] + padding,
            ]

            cropped_img = img.crop(bbox)

            table_tokens = [
                token for token in tokens if iob(token["bbox"], bbox) >= 0.5
            ]
            for token in table_tokens:
                token["bbox"] = [
                    token["bbox"][0] - bbox[0],
                    token["bbox"][1] - bbox[1],
                    token["bbox"][2] - bbox[0],
                    token["bbox"][3] - bbox[1],
                ]

            # If table is predicted to be rotated, rotate cropped image and tokens/words:
            if obj["label"] == "table rotated":
                cropped_img = cropped_img.rotate(270, expand=True)
                for token in table_tokens:
                    bbox = token["bbox"]
                    bbox = [
                        cropped_img.size[0] - bbox[3] - 1,
                        bbox[0],
                        cropped_img.size[0] - bbox[1] - 1,
                        bbox[2],
                    ]
                    token["bbox"] = bbox

            cropped_table["image"] = cropped_img
            cropped_table["tokens"] = table_tokens

            table_crops.append(cropped_table)

        return table_crops

    def detect(self, document: fitz.Document) -> List:
        """
        Run inference on a pdf from a path.

        Args:
            document (fitz.Document): Document.
        Returns:
            List: List of cropped table images.
        """
        all_crops = []
        for page_idx, page in enumerate(document):
            pix = page.get_pixmap(dpi=300)
            mode = "RGBA" if pix.alpha else "RGB"
            image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            # Grayscale
            image = ImageOps.grayscale(image).convert("RGB")
            image = pad_image(image, dilation_factor=self.padding_factor - 1.0)

            pixel_values = self.detection_transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(pixel_values)

            # update id2label to include "no object"
            id2label = self.model.config.id2label
            id2label[len(self.model.config.id2label)] = "no object"
            objects = self.outputs_to_objects(outputs, image.size, id2label)

            tokens = []
            detection_class_thresholds = {
                "table": 0.5,
                "table rotated": 0.5,
                "no object": 10,
            }
            # TODO: tune ?
            crop_padding = int(
                (self.crop_padding_factor - 1.0) * ((image.size[0] + image.size[1]) / 2)
            )

            page_crops = self.objects_to_crops(
                image, tokens, objects, detection_class_thresholds, padding=crop_padding
            )
            all_crops += [crop["image"].convert("RGB") for crop in page_crops]
        return all_crops

    def detect_from_path(self, pdf_path: str, s3: bool) -> List:
        """
        Run inference on a pdf from a path.

        Args:
            weights_path (str): Path to pdf.
            s3 (bool): Is the pdf on https://minio.lab.sspcloud.fr ?
        Returns:
            List: List of cropped table images.
        """
        # Read PDF page
        document = load_pdf(pdf_path, s3)
        return self.detect(document)
