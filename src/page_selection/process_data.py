"""
Training a random forest model.
"""
import tempfile
from datetime import datetime
import pickle
import json
import mlflow
import os
import sys
from tqdm import tqdm
from time import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
from .utils import (
    fs,
    clean_page_content,
    extract_document_content,
    fit_transform_vectorizer,
    train_random_forest,
    load_labeled_data,
    load_extra_labeled_data,
    get_numeric_char_rate,
    load_extra_labeled_data_checked,
)
from .model_wrapper import RandomForestWrapper
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy import sparse

import shutil


def main(test_size, starting_data, extra_data):
    """
    Main method.
    """
    if (not bool(starting_data)) & (not bool(extra_data)):
        raise ValueError("No data.")
    print(f"Using starting data: {bool(starting_data)}")
    print(f"Using extra data: {bool(extra_data)}")

    if bool(starting_data):
        flat_corpus, valid_labels, num_rates = load_labeled_data()
        if bool(extra_data):
            (
                flat_corpus_extra,
                valid_labels_extra,
                num_rates_extra,
            ) = load_extra_labeled_data_checked()
            flat_corpus += flat_corpus_extra
            valid_labels += valid_labels_extra
            num_rates += num_rates_extra
    else:
        (
            flat_corpus,
            valid_labels,
            num_rates,
        ) = load_extra_labeled_data_checked()

    # Split : random
    random_state = 42
    test_size = 0.2
    (
        train_corpus,
        test_corpus,
        y_train,
        y_test,
        train_num_rates,
        test_num_rates,
    ) = train_test_split(
        flat_corpus,
        valid_labels,
        num_rates,
        test_size=test_size,
        random_state=random_state,
    )

    vectorizer, X_train = fit_transform_vectorizer(train_corpus)
    X_train = sparse.hstack((X_train, np.array(train_num_rates)[:, None]))

    X_test = vectorizer.transform(test_corpus)
    X_test = sparse.hstack((X_test, np.array(test_num_rates)[:, None]))

    # Save vectorizer, X_train, y_train, X_test and y_test
    tag = str(datetime.now())
    s3_dir = (
        "/projet-extraction-tableaux/data/page_selection_data/" + tag + "/"
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        with open(tmpdirname + "/tokenizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        sparse.save_npz(tmpdirname + "/X_train.npz", X_train)
        sparse.save_npz(tmpdirname + "/X_test.npz", X_test)
        with open(tmpdirname + "/y_train.pkl", "wb") as f:
            pickle.dump(y_train, f)
        with open(tmpdirname + "/y_test.pkl", "wb") as f:
            pickle.dump(y_test, f)
        fs.put(tmpdirname, s3_dir, recursive=True)


if __name__ == "__main__":
    test_size = sys.argv[1]
    starting_data = sys.argv[2]
    extra_data = sys.argv[3]

    main(test_size, int(starting_data), int(extra_data))
