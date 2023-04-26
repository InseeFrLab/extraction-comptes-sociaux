"""
FastText wrapper for MLflow.
"""
import fasttext
import mlflow
import pandas as pd
import yaml


class FastTextWrapper(mlflow.pyfunc.PythonModel):
    """
    Class to wrap and use FastText Models.
    """

    def __init__(self):
        """
        Constructor.
        """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        To implement.
        """
        # pylint: disable=attribute-defined-outside-init
        self.model = fasttext.load_model(
            context.artifacts["fasttext_model_path"]
        )

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input: dict
    ) -> tuple:
        """
        To implement.
        """
        raise NotImplementedError()
