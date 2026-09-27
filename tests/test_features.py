import pandas as pd

from src.features import create_features


COLUMNS = [
    "unit", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "T2", "T24", "T30", "T50",
    "P2", "P15", "P30",
    "Nf", "Nc", "epr", "Ps30",
    "phi", "NRf", "NRc", "BPR",
    "farB", "htBleed", "Nf_dmd",
    "PCNfR_dmd", "W31", "W32"
]


def load_test_data():
    return pd.read_csv(
        "data/raw/test_FD001.txt",
        sep=r"\s+",
        header=None,
        names=COLUMNS
    )


def test_feature_engineering_runs():
    df = load_test_data()

    result = create_features(df)

    assert not result.empty


def test_feature_engineering_removes_nan_rows():
    df = load_test_data()

    result = create_features(df)

    assert not result.isna().any().any()


def test_feature_engineering_preserves_engine_count():
    df = load_test_data()

    result = create_features(df)

    assert result["unit"].nunique() == 100


def test_feature_engineering_contains_expected_columns():
    df = load_test_data()

    result = create_features(df)

    expected_columns = {
        "unit",
        "cycle",
        "T50_lag1",
        "T50_lag2",
        "T50_lag3",
        "T50_mean_5",
        "T50_mean_10",
        "T50_std_5",
        "T50_std_10",
        "T50_slope_10",
        "T50_slope_20",
    }

    assert expected_columns.issubset(
        set(result.columns)
    )