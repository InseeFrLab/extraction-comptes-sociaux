"""
Training a random forest model.
"""
import pickle
import mlflow
import os
import sys
from time import time
import tempfile
import scipy
from sklearn import metrics
from .utils import (
    train_random_forest,
    fs,
)
from .model_wrapper import RandomForestWrapper
from matplotlib import pyplot as plt
import seaborn as sns


def main(
    remote_server_uri: str, experiment_name: str, run_name: str, tag: str
):
    """
    Main method.

    Args:
        remote_server_uri (str): MLFlow server URI.
        experiment_name (str): MLFlow experiment name.
        run_name (str): MLFlow run name.
    """
    # Load data
    s3_dir = "projet-extraction-tableaux/data/page_selection_data/" + tag + "/"
    with tempfile.TemporaryDirectory() as tmpdirname:
        fs.get(rpath=s3_dir, lpath=tmpdirname + "/", recursive=True)

        X_train = scipy.sparse.load_npz(tmpdirname + "/X_train.npz")
        X_test = scipy.sparse.load_npz(tmpdirname + "/X_test.npz")

        with open(tmpdirname + "/y_train.pkl", "rb") as f:
            y_train = pickle.load(f)
        with open(tmpdirname + "/y_test.pkl", "rb") as f:
            y_test = pickle.load(f)
        with open(tmpdirname + "/tokenizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

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
            code_path=["ca_extract/page_selection/"],
            python_model=RandomForestWrapper(),
            artifacts=artifacts,
            registered_model_name="page_selection",
        )

        os.remove("pickled_model.pkl")
        os.remove("tokenizer.pkl")

        # Test time
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
        mlflow.log_param("data_tag", tag)
        mlflow.log_param("model_type", clf_descr)

        mlflow.log_metric("acc_test", accuracy)
        mlflow.log_metric("f1_test", f1)
        mlflow.log_metric("precision_test", precision)
        mlflow.log_metric("recall_test", recall)

        # Log confusion matrix
        ax = plt.subplot()
        plot = sns.heatmap(cm, annot=True, fmt="g", ax=ax)
        print(cm)

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
    tag = sys.argv[4]

    main(remote_server_uri, experiment_name, run_name, tag)
