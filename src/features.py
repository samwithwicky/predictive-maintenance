import numpy as np
import pandas as pd


# ============================================================
# COLUMN DEFINITIONS
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


SENSOR_COLS = [
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
# FEATURE ENGINEERING
# ============================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the engineered features used by the trained RUL model.

    Includes:
        - Lag features
        - Rolling means
        - Rolling standard deviations
        - Rolling slopes

    Parameters
    ----------
    df : pandas.DataFrame
        Raw engine sensor data.

    Returns
    -------
    pandas.DataFrame
        Feature-engineered dataset.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    lag_features = {}

    for lag in [1, 2, 3]:

        for col in SENSOR_COLS:

            name = f"{col}_lag{lag}"

            lag_features[name] = (
                df.groupby("unit")[col]
                .shift(lag)
            )

    lag_df = pd.DataFrame(
        lag_features,
        index=df.index,
    )

    df = pd.concat(
        [df, lag_df],
        axis=1,
    )

    # --------------------------------------------------------
    # Rolling features
    # --------------------------------------------------------

    rolling_features = {}

    for window in [5, 10]:

        for col in SENSOR_COLS:

            grouped = df.groupby("unit")[col]

            mean_name = f"{col}_mean_{window}"
            std_name = f"{col}_std_{window}"

            rolling_features[mean_name] = (
                grouped.transform(
                    lambda x:
                    x.rolling(window).mean()
                )
            )

            rolling_features[std_name] = (
                grouped.transform(
                    lambda x:
                    x.rolling(window).std()
                )
            )

    rolling_df = pd.DataFrame(
        rolling_features,
        index=df.index,
    )

    df = pd.concat(
        [df, rolling_df],
        axis=1,
    )

    # --------------------------------------------------------
    # Rolling slopes
    # --------------------------------------------------------

    slope_features = {}

    def calculate_slope(series):

        return series.rolling(
            window
        ).apply(
            lambda values:
            np.polyfit(
                np.arange(len(values)),
                values,
                1,
            )[0],
            raw=True,
        )

    for window in [10, 20]:

        for col in SENSOR_COLS:

            name = f"{col}_slope_{window}"

            slope_features[name] = (
                df.groupby("unit")[col]
                .transform(calculate_slope)
            )

    slope_df = pd.DataFrame(
        slope_features,
        index=df.index,
    )

    df = pd.concat(
        [df, slope_df],
        axis=1,
    )

    # --------------------------------------------------------
    # Remove rows without sufficient history
    # --------------------------------------------------------

    df = df.dropna().copy()

    return df