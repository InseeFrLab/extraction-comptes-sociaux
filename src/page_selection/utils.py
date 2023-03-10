"""
Utils functions.
"""
from typing import List, Dict, Tuple
from time import time
import os
import s3fs
import fitz
import re
import unidecode
import pandas as pd
from nltk.corpus import stopwords as ntlk_stopwords
from nltk.stem.snowball import SnowballStemmer
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from pytesseract import image_to_data


fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": "https://" + os.environ["AWS_S3_ENDPOINT"]},
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"]
)


def extract_document_content(
    pdf_path: str, s3: bool = True, resolution: int = 300
) -> List[pd.DataFrame]:
    """
    From a path to a pdf file, extract content as a list of
    strings each containing the text in a page.

    Args:
        pdf_path (str): Path to PDF file.
        s3 (bool): True if file is on s3.
        resolution (int): Resolution.
    """
    doc = load_pdf(pdf_path, s3)

    if is_scan(doc):
        return extract_document_content_ocr(doc, resolution=resolution)
    else:
        return extract_document_content_fitz(doc)


def is_scan(doc: fitz.Document) -> bool:
    """
    Return True if doc is a scan and False otherwise.

    Args:
        doc (fitz.Document): PDF document.
    """
    page_lengths = []
    for page_index, page in enumerate(doc):
        if page_index > 5:
            break
        page_content = page.get_text()
        page_lengths.append(len(page_content))

    # TODO: better heuristic to find ?
    if (sum(page_lengths) / len(page_lengths)) < 20:
        return True
    return False


def extract_document_content_fitz(doc: fitz.Document) -> List[pd.DataFrame]:
    """
    From a fitz.Document object, extract content as a list of
    strings each containing the text in a page.

    Args:
        doc (fitz.Document): PDF document.
    """
    page_list = []

    for page in doc:
        page_content = page.get_text()
        if not page_content:
            page_list.append("vide")
        else:
            page_list.append(page_content)

    return page_list


def extract_document_content_ocr(
    doc: fitz.Document, resolution: int = 200
) -> List[pd.DataFrame]:
    """
    From a fitz.Document object, extract content as a list of
    strings each containing the text in a page using Tesseract OCR.

    Args:
        doc (fitz.Document): PDF document.
        resolution (int): Resolution.
    """
    page_list = []

    for page in doc:
        pix = page.get_pixmap(dpi=resolution)
        mode = "RGBA" if pix.alpha else "RGB"
        image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        ocr = image_to_data(
            image, lang="fra", config="--psm 1", output_type="data.frame"
        )
        cleaned_ocr = ocr["text"].dropna()
        if cleaned_ocr.empty:
            page_list.append("vide")
        else:
            page_list.append(cleaned_ocr.str.cat(sep=" "))

    return page_list


def clean_page_content(page_content: str) -> str:
    """
    From a raw page content input as a string, return
    a clean string.

    Args:
        page_content (str): Content of a page.
    """
    # Remove line breaks
    content = page_content.replace("\r", "").replace("\n", "")
    # Remove punctuation
    content = re.sub(r"[^\w\s]", "", content)

    words = content.split()
    # Convert to lower case
    words = [word.lower() for word in words]
    # Remove numbers
    words_no_numbers = [word for word in words if not word.isdigit()]
    # Remove stopwords and stem
    stopwords = tuple(ntlk_stopwords.words("french"))
    stemmer = SnowballStemmer(language="french")
    words_no_numbers = [
        stemmer.stem(word)
        for word in words_no_numbers
        if word not in stopwords
    ]
    # Remove accents
    clean_content = " ".join(
        [unidecode.unidecode(word) for word in words_no_numbers]
    )

    return clean_content


def create_document_term_matrix(
    clean_page_content: str, vectorizer: TfidfVectorizer
) -> List[int]:
    """
    Create document-term matrix associated with clean page content.

    Args:
        clean_page_content (str): Clean page content.
        vectorizer (TfidfVectorizer): TfidfVectorizer.
    """
    return vectorizer.transform(clean_page_content)


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


def fit_transform_vectorizer(
    flat_corpus: List[str],
) -> Tuple[TfidfVectorizer, List[List[int]]]:
    """
    Function to fit a TfidfVectorizer on a corpus and to
    vectorize this corpus.

    Args:
        flat_corpus (List[str]): Corpus to vectorize.
    """
    vectorizer = TfidfVectorizer(input="content")
    vectorized_corpus = vectorizer.fit_transform(flat_corpus)
    return (vectorizer, vectorized_corpus)


def train_random_forest(
    params: Dict, X_train: List[List[int]], y_train: List[int]
):
    """
    Train a random forest classifier.

    Args:
        params (Dict): Parameters of the RF classifier.
        X_train (List[List[int]]): Training features.
        y_train (List[int]): Training outputs.
    """
    clf = RandomForestClassifier(**params)

    # Training time
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0

    # Classifier name
    clf_descr = clf.__class__.__name__
    return clf, clf_descr, train_time
