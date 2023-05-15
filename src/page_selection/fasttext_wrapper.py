"""
FastText wrapper for MLflow.
"""
import fasttext
import mlflow
import pandas as pd


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
        self._model = fasttext.load_model(
            context.artifacts["fasttext_model_path"]
        )

    def predict(
        self, context, input_model
    ) -> tuple:
        """
        Predict.

        Args:
            context: MLFlow context.
            input_model: Input for the model.
        """
        model_output = self._model.predict(input_model)

        predictions = [
            single_predictions[0].replace("__label__", "") for single_predictions in model_output[0]
        ]
        probas = [
            float(single_probas[0]) for single_probas in model_output[1]
        ]

        output = pd.DataFrame(
            {
                "predictions": predictions,
                "probas": probas,
            }
        )

        return output
