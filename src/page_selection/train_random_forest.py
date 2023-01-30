"""
Training a random forest model.
"""
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
    clean_page_content,
    extract_document_content,
    fit_transform_vectorizer,
    train_random_forest,
)
from .model_wrapper import RandomForestWrapper


def main(remote_server_uri: str, experiment_name: str, run_name: str):
    """
    Main method.

    Args:
        remote_server_uri (str): MLFlow server URI.
        experiment_name (str): MLFlow experiment name.
        run_name (str): MLFlow run name.
    """
    # TODO: clean up
    with open("data/updated_labels_filtered.json", "r") as fp:
        labels = json.load(fp)

    labeled_file_names = []
    valid_labels = []

    i = 0
    for file_name, file_labels in labels.items():
        # Keep documents with at least 1 table
        table_count = sum(file_labels)
        if table_count > 0:
            i += 1
            labeled_file_names.append(file_name)
            for label in file_labels:
                valid_labels.append(label)
            if i > 2:
                break

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

    flat_corpus = [item for sublist in corpus for item in sublist]
    vectorizer, vectorized_corpus = fit_transform_vectorizer(flat_corpus)

    X_train, X_test, y_train, y_test = train_test_split(
        vectorized_corpus, valid_labels, test_size=0.2, random_state=42
    )

    # Training classifier
    params = {
        "n_estimators": 100,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    }

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        clf, clf_descr, train_time = train_random_forest(
            params, X_train, y_train
        )

        # Saving model and tokenizer as pickle files
        with open("pickled_model.pkl", "wb") as f:
            pickle.dump(clf, f)
        with open("tokenizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

        artifacts = {
            "state_dict": "pickled_model.pkl",
            "tokenizer": "tokenizer.pkl",
        }

        mlflow.pyfunc.log_model(
            artifact_path=run_name,
            code_path=["src/page_selection/"],
            python_model=RandomForestWrapper(),
            artifacts=artifacts,
            registered_model_name="page_selection"
        )

        os.remove("pickled_model.pkl")
        os.remove("tokenizer.pkl")

        # Test time
        t0 = time()
        pred = clf.predict(X_test)
        test_time = time() - t0
        # Score
        score = metrics.accuracy_score(y_test, pred)

        for param, value in params.items():
            mlflow.log_param(param, value)
        mlflow.log_param("model_type", clf_descr)

        mlflow.log_metric("score_test", score)
        mlflow.log_metric("train_time", train_time)
        mlflow.log_metric("test_time", test_time)


if __name__ == "__main__":
    # MLFlow params
    remote_server_uri = sys.argv[1]
    experiment_name = sys.argv[2]
    run_name = sys.argv[3]

    main(remote_server_uri, experiment_name, run_name)
