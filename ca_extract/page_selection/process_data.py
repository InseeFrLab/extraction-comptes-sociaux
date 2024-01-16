"""
Training a random forest model.
"""
from typing import List
import tempfile
from datetime import datetime
import pickle
import sys
from sklearn.model_selection import train_test_split
from .utils import (
    fs,
    fit_transform_vectorizer,
    load_labeled_data,
    load_extra_labeled_data_checked,
)
import numpy as np
from scipy import sparse


def process_data(test_size: float, starting_data: int, extra_data: int):
    """
    Process data until before tokenizing.

    Args:
        test_size (float): Test size.
        starting_data (int): Starting data.
        extra_data (int): Extra data.
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

    return (
        train_corpus,
        test_corpus,
        y_train,
        y_test,
        train_num_rates,
        test_num_rates,
    )


def tokenize_processed_data(
    train_corpus: List[str],
    test_corpus: List[str],
    train_num_rates: List[float],
    test_num_rates: List[float],
):
    """
    Tokenize processed data.

    Args:
        train_corpus (List[str]): Train corpus.
        test_corpus (List[str]): Test corpus.
        train_num_rates (List[float]): Train num rates.
        test_num_rates (List[float]): Test num rates.
    """
    vectorizer, X_train = fit_transform_vectorizer(train_corpus)
    X_train = sparse.hstack((X_train, np.array(train_num_rates)[:, None]))

    X_test = vectorizer.transform(test_corpus)
    X_test = sparse.hstack((X_test, np.array(test_num_rates)[:, None]))

    return vectorizer, X_train, X_test


def main(test_size: float, starting_data: int, extra_data: int, tokenize: int):
    """
    Main method.

    Args:
        test_size (float): Test size.
        starting_data (int): Starting data.
        extra_data (int): Extra data.
        tokenize (int) : Should texts be tokenized ?
    """
    (
        train_corpus,
        test_corpus,
        y_train,
        y_test,
        train_num_rates,
        test_num_rates,
    ) = process_data(test_size, starting_data, extra_data)

    tag = str(datetime.now())
    s3_dir = (
        "/projet-extraction-tableaux/data/page_selection_data/" + tag + "/"
    )

    if not bool(tokenize):
        # Save non-tokenized data on s3
        with tempfile.TemporaryDirectory() as tmpdirname:
            with open(tmpdirname + "/train_corpus.pkl", "wb") as f:
                pickle.dump(train_corpus, f)
            with open(tmpdirname + "/test_corpus.pkl", "wb") as f:
                pickle.dump(test_corpus, f)
            with open(tmpdirname + "/y_train.pkl", "wb") as f:
                pickle.dump(y_train, f)
            with open(tmpdirname + "/y_test.pkl", "wb") as f:
                pickle.dump(y_test, f)
            fs.put(tmpdirname, s3_dir, recursive=True)
    else:
        vectorizer, X_train, X_test = tokenize_processed_data(
            train_corpus, test_corpus, train_num_rates, test_num_rates
        )
        # Save vectorizer, X_train, y_train, X_test and y_test
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
    tokenize = sys.argv[4]

    main(test_size, int(starting_data), int(extra_data), int(tokenize))
