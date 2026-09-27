import pandas as pd
import shap

from .model import RULModel


class RULExplainer:
    """
    SHAP-based explainability interface for the RUL model.

    This module explains individual predictions without
    changing the underlying model or prediction process.
    """

    def __init__(
        self,
        model: RULModel | None = None,
    ):
        self.model = (
            model
            if model is not None
            else RULModel()
        )

        self.explainer = shap.TreeExplainer(
            self.model.model
        )

    # --------------------------------------------------------
    # Generate SHAP explanation
    # --------------------------------------------------------

    def explain(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate SHAP values for one or more observations.

        Parameters
        ----------
        data : pandas.DataFrame
            Feature-engineered data.

        Returns
        -------
        pandas.DataFrame
            SHAP values for the model features.
        """

        missing_features = [
            feature
            for feature in self.model.features
            if feature not in data.columns
        ]

        if missing_features:
            raise ValueError(
                "Missing required features: "
                + ", ".join(missing_features)
            )

        X = data[self.model.features]

        shap_values = self.explainer.shap_values(X)

        shap_values = self._normalize_shap_values(
            shap_values
        )

        return pd.DataFrame(
            shap_values,
            columns=self.model.features,
            index=data.index,
        )

    # --------------------------------------------------------
    # Explain one observation
    # --------------------------------------------------------

    def explain_one(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate a detailed local explanation
        for exactly one observation.
        """

        if len(data) != 1:
            raise ValueError(
                "explain_one() expects exactly "
                "one observation."
            )

        X = data[self.model.features]

        shap_values = self.explainer.shap_values(X)

        shap_values = self._normalize_shap_values(
            shap_values
        )

        explanation = pd.DataFrame(
            {
                "feature": self.model.features,
                "feature_value": X.iloc[0].values,
                "shap_value": shap_values[0],
            }
        )

        # Magnitude is used only for sorting.
        # The original SHAP sign is preserved.
        explanation["abs_shap"] = (
            explanation["shap_value"].abs()
        )

        explanation["direction"] = (
            explanation["shap_value"]
            .apply(
                lambda value:
                "INCREASES RUL"
                if value > 0
                else "DECREASES RUL"
            )
        )

        explanation = (
            explanation
            .sort_values(
                "abs_shap",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        return explanation

    # --------------------------------------------------------
    # Top contributors
    # --------------------------------------------------------

    def top_contributors(
        self,
        data: pd.DataFrame,
        n: int = 10,
    ) -> pd.DataFrame:
        """
        Return the strongest local SHAP contributors.
        """

        explanation = self.explain_one(data)

        return explanation.head(n)

    # --------------------------------------------------------
    # SHAP output normalization
    # --------------------------------------------------------

    @staticmethod
    def _normalize_shap_values(
        shap_values,
    ):
        """
        Normalize SHAP output into a 2D array.

        Handles the standard TreeExplainer output
        used by the regression model.
        """

        values = shap_values

        if isinstance(values, list):
            values = values[0]

        values = pd.DataFrame(values).to_numpy()

        return values