from pathlib import Path

import pandas as pd

from .features import create_features
from .model import RULModel


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "test_FD001.txt"
)


# ============================================================
# DATASET COLUMNS
# ============================================================

COLUMNS = [
    "unit",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "T2",
    "T24",
    "T30",
    "T50",
    "P2",
    "P15",
    "P30",
    "Nf",
    "Nc",
    "epr",
    "Ps30",
    "phi",
    "NRf",
    "NRc",
    "BPR",
    "farB",
    "htBleed",
    "Nf_dmd",
    "PCNfR_dmd",
    "W31",
    "W32",
]


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(
    rul: float,
) -> str:
    """
    Classify an engine based on predicted RUL.
    """

    if rul <= 20:

        return "Maintenance Attention"

    if rul <= 50:

        return "Monitor"

    return "Normal"


# ============================================================
# INFERENCE ENGINE
# ============================================================

class InferenceEngine:
    """
    High-level interface for engine RUL inference.

    Pipeline:

        Raw sensor data
                ↓
        Feature engineering
                ↓
        Latest engine observation
                ↓
        RUL prediction
                ↓
        Risk classification
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

    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    def load_data(
        self,
        path: Path = TEST_DATA_PATH,
    ) -> pd.DataFrame:
        """
        Load raw engine sensor data.
        """

        path = Path(path)

        if not path.exists():

            raise FileNotFoundError(
                f"Engine data not found: "
                f"{path}"
            )

        data = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
        )

        data.columns = COLUMNS

        return data

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    def prepare(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply the project's feature-engineering
        pipeline.
        """

        return create_features(data)

    # --------------------------------------------------------
    # Available engines
    # --------------------------------------------------------

    @staticmethod
    def get_engine_ids(
        data: pd.DataFrame,
    ) -> list[int]:
        """
        Return all available engine IDs.
        """

        return sorted(
            data["unit"]
            .astype(int)
            .unique()
            .tolist()
        )

    # --------------------------------------------------------
    # Latest observation
    # --------------------------------------------------------

    @staticmethod
    def get_latest_observation(
        data: pd.DataFrame,
        engine_id: int,
    ) -> pd.DataFrame:
        """
        Return the latest available observation
        for a specific engine.
        """

        engine_data = (
            data[
                data["unit"] == engine_id
            ]
            .sort_values("cycle")
        )

        if engine_data.empty:

            raise ValueError(
                f"Engine {engine_id} "
                f"not found."
            )

        return engine_data.tail(1)

    # --------------------------------------------------------
    # Engine prediction
    # --------------------------------------------------------

    def predict_engine(
        self,
        data: pd.DataFrame,
        engine_id: int,
    ) -> dict:
        """
        Generate a complete prediction
        for one engine.
        """

        latest = (
            self.get_latest_observation(
                data,
                engine_id,
            )
        )

        predicted_rul = (
            self.model.predict_one(
                latest
            )
        )

        latest_cycle = int(
            latest["cycle"].iloc[0]
        )

        risk = classify_risk(
            predicted_rul
        )

        return {
            "engine_id": int(engine_id),
            "latest_cycle": latest_cycle,
            "predicted_rul": predicted_rul,
            "risk": risk,
        }