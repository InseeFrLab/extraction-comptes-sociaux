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
    load_labeled_data,
    load_extra_labeled_data,
    get_numeric_char_rate,
)
from .model_wrapper import RandomForestWrapper
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy import sparse


def main(remote_server_uri: str, experiment_name: str, run_name: str):
    """
    Main method.

    Args:
        remote_server_uri (str): MLFlow server URI.
        experiment_name (str): MLFlow experiment name.
        run_name (str): MLFlow run name.
    """
    flat_corpus, valid_labels = load_labeled_data()
    flat_corpus_extra, valid_labels_extra = load_extra_labeled_data()
    flat_corpus += flat_corpus_extra
    valid_labels += valid_labels_extra

    # Add new feature : rate of numeric characters
    num_rates = [get_numeric_char_rate(content) for content in flat_corpus]

    # Split
    random_state = 42
    test_size = 0.2
    train_num_rates, test_num_rates = train_test_split(
        num_rates, test_size=test_size, random_state=random_state
    )
    train_corpus, test_corpus, y_train, y_test = train_test_split(
        flat_corpus,
        valid_labels,
        test_size=test_size,
        random_state=random_state,
    )

    vectorizer, X_train = fit_transform_vectorizer(train_corpus)
    X_train = sparse.hstack((X_train, np.array(train_num_rates)[:, None]))

    vectorizer, X_train = fit_transform_vectorizer(train_corpus)
    X_train = sparse.hstack((X_train, np.array(train_num_rates)[:, None]))

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
            registered_model_name="page_selection",
        )

        os.remove("pickled_model.pkl")
        os.remove("tokenizer.pkl")

        # Test time
        X_test = vectorizer.transform(test_corpus)
        X_test = sparse.hstack((X_test, np.array(test_num_rates)[:, None]))

        t0 = time()
        pred = clf.predict(X_test)
        test_time = time() - t0
        # Performance metrics
        accuracy = metrics.accuracy_score(y_test, pred)
        f1 = metrics.f1_score(y_test, pred)
        precision = metrics.precision_score(y_test, pred)
        recall = metrics.recall_score(y_test, pred)
        cm = metrics.confusion_matrix(y_test, pred)

        for param, value in params.items():
            mlflow.log_param(param, value)
        mlflow.log_param("model_type", clf_descr)

        mlflow.log_metric("acc_test", accuracy)
        mlflow.log_metric("f1_test", f1)
        mlflow.log_metric("precision_test", precision)
        mlflow.log_metric("recall_test", recall)

        # Log confusion matrix
        ax = plt.subplot()
        plot = sns.heatmap(cm, annot=True, fmt="g", ax=ax)

        # labels, title and ticks
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix")
        mlflow.log_figure(plot.get_figure(), "cm.png")

        # Log train and test times
        mlflow.log_metric("train_time", train_time)
        mlflow.log_metric("test_time", test_time)


if __name__ == "__main__":
    # MLFlow params
    remote_server_uri = sys.argv[1]
    experiment_name = sys.argv[2]
    run_name = sys.argv[3]

    main(remote_server_uri, experiment_name, run_name)
