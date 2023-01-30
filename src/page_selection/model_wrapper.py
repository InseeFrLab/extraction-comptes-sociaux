"""
MLFlow wrapper class for a RandomForest model.
"""
import mlflow
import pickle
import pandas as pd


class RandomForestWrapper(mlflow.pyfunc.PythonModel):
    """
    RF wrapper.
    """

    def load_context(self, context):
        """
        Load context.

        Args:
            context: MLFlow context.
        """
        # Load in and deserialize the model tokenizer
        with open(context.artifacts["tokenizer"], "rb") as handle:
            self._model_tokenizer = pickle.load(handle)

        with open(context.artifacts["state_dict"], "rb") as handle:
            self._model = pickle.load(handle)

    def predict(self, context, input_model):
        """
        Predict.

        Args:
            context: MLFlow context.
            input_model: Input for the model.
        """
        vectorized_input = self._model_tokenizer.transform(input_model)
        predictions = self._model.predict(vectorized_input)
        probas = self._model.predict_proba(vectorized_input)

        output = pd.DataFrame(
            {
                "predictions": predictions,
                "probas": [page_probas[1] for page_probas in probas],
            }
        )

        return output
