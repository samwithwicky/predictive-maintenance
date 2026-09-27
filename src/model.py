from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "rul_random_forest_capped.pkl"
)

FEATURE_PATH = (
    PROJECT_ROOT
    / "models"
    / "rul_features.pkl"
)


# ============================================================
# MODEL INTERFACE
# ============================================================

class RULModel:
    """
    Production interface for the trained RUL model.

    Handles:
        - Model loading
        - Feature list loading
        - Artifact validation
        - RUL prediction
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        feature_path: Path = FEATURE_PATH,
    ):

        self.model_path = Path(model_path)
        self.feature_path = Path(feature_path)

        self.model = None
        self.features = None

        self._load()

    # --------------------------------------------------------
    # Load artifacts
    # --------------------------------------------------------

    def _load(self):

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Model file not found: "
                f"{self.model_path}"
            )

        if not self.feature_path.exists():

            raise FileNotFoundError(
                f"Feature list not found: "
                f"{self.feature_path}"
            )

        self.model = joblib.load(
            self.model_path
        )

        self.features = joblib.load(
            self.feature_path
        )

        self._validate()

    # --------------------------------------------------------
    # Validate artifacts
    # --------------------------------------------------------

    def _validate(self):

        if not self.features:

            raise ValueError(
                "Feature list is empty."
            )

        if len(set(self.features)) != len(
            self.features
        ):

            raise ValueError(
                "Feature list contains "
                "duplicate features."
            )

        expected_features = getattr(
            self.model,
            "n_features_in_",
            None,
        )

        if expected_features is not None:

            if expected_features != len(
                self.features
            ):

                raise ValueError(
                    "Model and feature list "
                    "do not match: "
                    f"model expects "
                    f"{expected_features}, "
                    f"feature list contains "
                    f"{len(self.features)}."
                )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    def predict(
        self,
        data: pd.DataFrame,
    ) -> pd.Series:
        """
        Predict RUL for multiple observations.
        """

        missing_features = [
            feature
            for feature in self.features
            if feature not in data.columns
        ]

        if missing_features:

            raise ValueError(
                "Missing required features: "
                + ", ".join(missing_features)
            )

        X = data[self.features]

        predictions = self.model.predict(X)

        # RUL cannot be negative.
        predictions = predictions.clip(
            min=0
        )

        return pd.Series(
            predictions,
            index=data.index,
            name="predicted_rul",
        )

    # --------------------------------------------------------
    # Single prediction
    # --------------------------------------------------------

    def predict_one(
        self,
        data: pd.DataFrame,
    ) -> float:
        """
        Predict RUL for exactly one observation.
        """

        predictions = self.predict(data)

        if len(predictions) != 1:

            raise ValueError(
                "predict_one() expects exactly "
                "one observation."
            )

        return float(
            predictions.iloc[0]
        )