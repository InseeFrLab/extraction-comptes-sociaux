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
import json
import pickle
import threading
from tqdm import tqdm
from nltk.corpus import stopwords as ntlk_stopwords
from nltk.stem.snowball import SnowballStemmer
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from pytesseract import image_to_data


fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": "https://" + os.environ["AWS_S3_ENDPOINT"]},
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
)


def extract_document_content(
    pdf_path: str,
    s3: bool = True,
    resolution: int = 200,
    parallel: bool = True,
    maxthreads: int = 10,
) -> List[pd.DataFrame]:
    """
    From a path to a pdf file, extract content as a list of
    strings each containing the text in a page.

    Args:
        pdf_path (str): Path to PDF file.
        s3 (bool): True if file is on s3.
        resolution (int): Resolution.
        parallel (bool): True if OCR should be in parallel in multiple threads.
        maxthreads (int): Max. number of threads for parallel processing.
    """
    doc = load_pdf(pdf_path, s3)

    if is_scan(doc):
        return extract_document_content_ocr(
            doc,
            resolution=resolution,
            parallel=parallel,
            maxthreads=maxthreads,
        )
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
    doc: fitz.Document,
    resolution: int = 200,
    parallel: bool = True,
    maxthreads: int = 10,
) -> List[str]:
    """
    From a fitz.Document object, extract content as a list of
    strings each containing the text in a page using Tesseract OCR.

    Args:
        doc (fitz.Document): PDF document.
        resolution (int): Resolution.
        parallel (bool): True if OCR should be in parallel in multiple threads.
        maxthreads (int): Max. number of threads for parallel processing.
    """
    page_list = []
    if not parallel:
        for page in doc:
            page_list.append(extract_ocr_page(page, resolution))
    # Parallel workflow
    else:
        # List of threads
        threads = []
        lock = threading.Lock()
        semaphore = threading.BoundedSemaphore(value=maxthreads)

        page_dict = {}
        for page_number, page in enumerate(doc):
            t = threading.Thread(
                group=None,
                target=ocr_page_to_dict,
                args=(
                    page_number,
                    page,
                    page_dict,
                    resolution,
                    lock,
                    semaphore,
                ),
            )
            threads.append(t)
            t.start()

        # Wait for threads to complete
        for t in threads:
            t.join()

        # Transform output dict
        for key, value in sorted(page_dict.items()):
            page_list.append(value)

    return page_list


def ocr_page_to_dict(
    page_number: int,
    page: fitz.Page,
    page_dict: Dict,
    resolution: int,
    lock: threading.Lock,
    semaphore: threading.BoundedSemaphore,
):
    """
    OCR page with given page number and puts result in a dictionary.

    Args:
        page_number (int): Page number.
        page (fitz.Page): Page.
        page_dict (Dict): Dictionary to update.
        resolution (int): Resolution.
        lock (threading.Lock): Lock.
        semaphore (threading.BoundedSemaphore): Semaphore.
    """
    with semaphore:
        page_content = extract_ocr_page(page, resolution)
        with lock:
            page_dict[page_number] = page_content
        return


def extract_ocr_page(page: fitz.Page, resolution: int = 200):
    """
    Extract text content from PDF fitz Page using Tesseract OCR.

    Args:
        page (fitz.Page): Page.
        resolution (int): Resolution.
    """
    pix = page.get_pixmap(dpi=resolution)
    mode = "RGBA" if pix.alpha else "RGB"
    image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    ocr = image_to_data(
        image, lang="fra", config="--psm 1", output_type="data.frame"
    )
    cleaned_ocr = ocr["text"].dropna()
    if cleaned_ocr.empty:
        return "vide"
    else:
        return cleaned_ocr.str.cat(sep=" ")


def clean_page_content(page_content: str) -> str:
    """
    From a raw page content input as a string, return
    a clean string.

    Args:
        page_content (str): Content of a page.
    """
    # Remove line breaks
    content = page_content.replace("\r", " ").replace("\n", " ")
    # Remove punctuation
    content = re.sub(r"[^\w\s]", " ", content)

    words = content.split()
    # Convert to lower case
    words = [word.lower() for word in words]
    # Remove stopwords and stem
    stopwords = tuple(ntlk_stopwords.words("french"))
    stemmer = SnowballStemmer(language="french")
    words_no_numbers = [
        stemmer.stem(word) for word in words if word not in stopwords
    ]
    # Remove accents
    clean_content = " ".join(
        [unidecode.unidecode(word) for word in words_no_numbers]
    )

    return clean_content


def remove_number(page_content: str) -> str:
    """
    Remove all numbers from a string.

    Args:
        page_content (str): Content of a page.
    """
    content = re.sub("[0-9]+", "", page_content)
    content = re.sub(r"\s+", " ", content).strip()

    return content


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


def load_extra_labeled_data():
    """ """

    with fs.open(
        "s3://projet-extraction-tableaux/data/df_trainrf.pickle", "rb"
    ) as f:
        df = pickle.load(f)

    flat_corpus = list(df.text)
    flat_corpus_with_number = [
        clean_page_content(page) for page in flat_corpus
    ]
    flat_corpus = [remove_number(page) for page in flat_corpus_with_number]
    valid_labels = list(df.tableau_f_et_p)

    num_rates = []
    num_rates = [
        get_numeric_char_rate(content) for content in flat_corpus_with_number
    ]

    return flat_corpus, valid_labels, num_rates


def load_labeled_data():
    """
    Load data labeled manually on a selection of
    pdf documents. Keep only those with at least
    1 page with the relevant table.
    """
    with fs.open(
        "s3://projet-extraction-tableaux/updated_labels_filtered.json", "rb"
    ) as f:
        labels = json.load(f)

    labeled_file_names = []
    valid_labels = []
    num_rates = []

    i = 0
    for file_name, file_labels in labels.items():
        # Keep documents with at least 1 table
        table_count = sum(file_labels)
        if table_count > 0:
            i += 1
            labeled_file_names.append(file_name)
            for label in file_labels:
                valid_labels.append(label)

    corpus = []
    labeled_file_names = [
        "projet-extraction-tableaux/raw-comptes/CS_extrait/" + file_name
        for file_name in labeled_file_names
    ]
    for file_name in tqdm(labeled_file_names):
        clean_document_content = []
        page_list = extract_document_content(file_name, resolution=50)
        for page in page_list:
            clean_content = clean_page_content(page)
            clean_document_content.append(clean_content)
        corpus.append(clean_document_content)

    flat_corpus_with_number = [item for sublist in corpus for item in sublist]
    flat_corpus = [remove_number(page) for page in flat_corpus_with_number]
    num_rates = [
        get_numeric_char_rate(content) for content in flat_corpus_with_number
    ]

    return flat_corpus, valid_labels, num_rates


def get_numeric_char_rate(page_content: str):
    """
    Compute rate of numeric characters in `page_content`.

    Args:
        page_content (str): Page content.
    """
    try:
        return float(len("".join(re.findall("\d", page_content)))) / float(
            len(re.sub(r"\s+", "", page_content))
        )
    except ZeroDivisionError:
        return 0.0
