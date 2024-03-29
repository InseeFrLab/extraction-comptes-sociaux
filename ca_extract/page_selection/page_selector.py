"""
Page selector.
"""
from PIL import Image
import fitz
from .utils import (
    clean_page_content,
    extract_content_from_file,
    extract_document_content,
    load_pdf,
)
import mlflow


class PageSelector:
    """
    Page selector.
    """

    def __init__(
        self,
        clf: mlflow.pyfunc.PythonModel,
        threshold: float = 0.5,
        resolution: int = 200,
        parallel: bool = True,
        maxthreads: int = 10,
    ):
        """
        Constructor.

        Args:
            clf (mlflow.pyfunc.PythonModel): Classifier.
            threshold: Threshold under which a page is not classified as
                containing the table.
            resolution (int): Resolution used for the OCR.
            parallel (bool): True if OCR should be in parallel in multiple
                threads.
            maxthreads (int): Max. number of threads for parallel processing.
        """
        self.clf = clf
        self.threshold = threshold
        self.resolution = resolution
        self.parallel = parallel
        self.maxthreads = maxthreads

    def get_page_from_file(
        self,
        pdf_path: str,
        s3: bool = True,
        dpi: int = 300,
    ) -> Image:
        """
        Returns the page of the pdf stored in pdf_path with the fp table
        as an image.

        Args:
            pdf_path (str): Path to pdf file.
            s3 (bool): True if file is on s3.
            dpi (int): Resolution of picture output.
        """
        page_number = self.get_page_number_from_file(pdf_path, s3)

        doc = load_pdf(pdf_path, s3)
        pix = doc[page_number].get_pixmap(dpi=dpi)
        mode = "RGBA" if pix.alpha else "RGB"
        image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        return image

    def get_page_number_from_file(self, pdf_path: str, s3: bool = True) -> int:
        """
        Returns the number of the page containing the fp table.

        Args:
            pdf_path (str): Path to pdf file.
            s3 (bool): Indicates if s3 file system should be used.

        Returns:
            int: Number of page with the "filiales et participations" table.
        """
        page_list = extract_content_from_file(
            pdf_path, s3, self.resolution, self.parallel, self.maxthreads
        )

        clean_page_list = []
        for page in page_list:
            clean_page_list.append(clean_page_content(page))

        model_output = self.clf.predict(clean_page_list)
        predictions = model_output["predictions"]
        probas = model_output["probas"]
        std_probas = probas.copy()
        for idx, prediction in enumerate(predictions):
            if str(prediction) == "0":
                std_probas[idx] = 1 - probas[idx]

        if std_probas.max() < self.threshold:
            raise ValueError(
                "Pas de tableau filiales et participations détecté."
            )
        else:
            return std_probas.idxmax()

    def get_page_number(self, document: fitz.Document) -> int:
        """
        Returns the number of the page containing the fp table.

        Args:
            document (fitz.Document): PDF document.

        Returns:
            int: Number of page with the "filiales et participations" table.
        """
        page_list = extract_document_content(
            document,
            self.resolution,
            self.parallel,
            self.maxthreads
        )

        clean_page_list = []
        for page in page_list:
            clean_page_list.append(clean_page_content(page))

        model_output = self.clf.predict(clean_page_list)
        predictions = model_output["predictions"]
        probas = model_output["probas"]
        std_probas = probas.copy()
        for idx, prediction in enumerate(predictions):
            if str(prediction) == "0":
                std_probas[idx] = 1 - probas[idx]

        if std_probas.max() < self.threshold:
            raise ValueError(
                "Pas de tableau filiales et participations détecté."
            )
        else:
            return std_probas.idxmax()
