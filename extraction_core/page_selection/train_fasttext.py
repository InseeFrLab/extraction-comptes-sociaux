"""
Training a FastText model.
"""
import pickle
import mlflow
import os
import sys
from time import time
import tempfile
from sklearn import metrics
import fasttext
from .utils import fs
from .fasttext_wrapper import FastTextWrapper
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
        with open(tmpdirname + "/train_corpus.pkl", "rb") as f:
            train_corpus = pickle.load(f)
        with open(tmpdirname + "/test_corpus.pkl", "rb") as f:
            test_corpus = pickle.load(f)
        with open(tmpdirname + "/y_train.pkl", "rb") as f:
            y_train = pickle.load(f)
        with open(tmpdirname + "/y_test.pkl", "rb") as f:
            y_test = pickle.load(f)

    params = {
        "dim": 150,
        "lr": 0.3,
        "epoch": 60,
        "wordNgrams": 3,
        "minn": 3,
        "maxn": 4,
        "minCount": 3,
        "bucket": 2000000,
        "thread": 50,
        "loss": "ova",
        "label_prefix": "__label__",
    }

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        # Write training data in file and train model
        with open("train_text.txt", "w", encoding="utf-8") as file:
            for lib, label in zip(train_corpus, y_train):
                formatted_item = f"__label__{label} {lib}"
                file.write(f"{formatted_item}\n")

        t0 = time()
        model = fasttext.train_supervised(
            "train_text.txt", **params, verbose=2
        )
        train_time = time() - t0

        fasttext_model_path = run_name + ".bin"
        model.save_model(fasttext_model_path)

        artifacts = {
            "fasttext_model_path": fasttext_model_path,
            "train_data": "train_text.txt",
        }

        mlflow.pyfunc.log_model(
            artifact_path=run_name,
            code_path=["extraction_core/page_selection/"],
            python_model=FastTextWrapper(),
            artifacts=artifacts,
        )

        os.remove(fasttext_model_path)
        os.remove("train_text.txt")

        # Test time
        t0 = time()
        res = model.predict(test_corpus, k=1)
        test_time = time() - t0

        pred = [
            int(x[0].replace("__label__", "")) for x, y in zip(res[0], res[1])
        ]
        # Performance metrics
        accuracy = metrics.accuracy_score(y_test, pred)
        f1 = metrics.f1_score(y_test, pred)
        precision = metrics.precision_score(y_test, pred)
        recall = metrics.recall_score(y_test, pred)
        cm = metrics.confusion_matrix(y_test, pred)

        for param, value in params.items():
            mlflow.log_param(param, value)
        mlflow.log_param("data_tag", tag)
        mlflow.log_param("model_type", "fasttext")

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
