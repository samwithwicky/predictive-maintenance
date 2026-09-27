import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. Load trained model
# =========================================================

model = joblib.load(
    "models/rul_random_forest_capped.pkl"
)

top_features = joblib.load(
    "models/rul_features.pkl"
)

print("Model loaded.")


# =========================================================
# 2. Load test sensor data
# =========================================================

test_df = pd.read_csv(
    "data/raw/test_FD001.txt",
    sep=r"\s+",
    header=None
)

test_df.columns = [
    "unit", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "T2", "T24", "T30", "T50",
    "P2", "P15", "P30",
    "Nf", "Nc", "epr", "Ps30",
    "phi", "NRf", "NRc", "BPR",
    "farB", "htBleed", "Nf_dmd",
    "PCNfR_dmd", "W31", "W32"
]

print("Test data loaded.")
print("Engines:", test_df["unit"].nunique())
print("Rows:", len(test_df))


# =========================================================
# 3. Load true test RUL values
# =========================================================

true_rul = pd.read_csv(
    "data/raw/RUL_FD001.txt",
    header=None
)

true_rul = true_rul[0].values


# =========================================================
# 4. Sensor columns
# =========================================================

sensor_cols = [
    "T2", "T24", "T30", "T50",
    "P2", "P15", "P30",
    "Nf", "Nc", "epr", "Ps30",
    "phi", "NRf", "NRc", "BPR",
    "farB", "htBleed", "Nf_dmd",
    "PCNfR_dmd", "W31", "W32"
]


# =========================================================
# 5. Feature engineering
# =========================================================

# Lag features

for lag in [1, 2, 3]:

    lag_features = {}

    for col in sensor_cols:

        name = f"{col}_lag{lag}"

        lag_features[name] = (
            test_df.groupby("unit")[col]
            .shift(lag)
        )

    test_df = pd.concat(
        [test_df, pd.DataFrame(lag_features)],
        axis=1
    )


# Rolling statistics

rolling_features = {}

for window in [5, 10]:

    for col in sensor_cols:

        rolling_group = (
            test_df.groupby("unit")[col]
        )

        rolling_features[
            f"{col}_mean_{window}"
        ] = (
            rolling_group
            .transform(
                lambda x:
                x.rolling(window).mean()
            )
        )

        rolling_features[
            f"{col}_std_{window}"
        ] = (
            rolling_group
            .transform(
                lambda x:
                x.rolling(window).std()
            )
        )


test_df = pd.concat(
    [test_df, pd.DataFrame(rolling_features)],
    axis=1
)


# Rolling slopes

slope_features = {}


def calculate_slope(x):

    return x.rolling(10).apply(
        lambda y:
        np.polyfit(
            np.arange(len(y)),
            y,
            1
        )[0],
        raw=True
    )


def calculate_slope_20(x):

    return x.rolling(20).apply(
        lambda y:
        np.polyfit(
            np.arange(len(y)),
            y,
            1
        )[0],
        raw=True
    )


for col in sensor_cols:

    slope_features[
        f"{col}_slope_10"
    ] = (
        test_df.groupby("unit")[col]
        .transform(calculate_slope)
    )

    slope_features[
        f"{col}_slope_20"
    ] = (
        test_df.groupby("unit")[col]
        .transform(calculate_slope_20)
    )


test_df = pd.concat(
    [test_df, pd.DataFrame(slope_features)],
    axis=1
)


print("Feature engineering complete.")


# =========================================================
# 6. Construct actual RUL for every test observation
# =========================================================

# Maximum observed cycle for each engine

max_cycles = (
    test_df
    .groupby("unit")["cycle"]
    .max()
)


# Map supplied final RUL to each engine

test_df["final_RUL"] = (
    test_df["unit"]
    .map(
        dict(
            zip(
                test_df["unit"].unique(),
                true_rul
            )
        )
    )
)


# Actual RUL at each cycle

test_df["actual_RUL"] = (
    test_df["final_RUL"]
    +
    (
        test_df["unit"]
        .map(max_cycles)
        -
        test_df["cycle"]
    )
)


# Apply same cap used during training

test_df["actual_RUL"] = (
    test_df["actual_RUL"]
    .clip(upper=125)
)


# =========================================================
# 7. Remove rows without enough history
# =========================================================

test_df = test_df.dropna(
    subset=top_features
).copy()


print(
    "Rows available after feature engineering:",
    len(test_df)
)


# =========================================================
# 8. Select exact training features
# =========================================================

X_test = test_df[top_features]

y_test = test_df["actual_RUL"]


print(
    "Test feature matrix:",
    X_test.shape
)


# =========================================================
# 9. Generate predictions
# =========================================================

print("\nGenerating predictions...")

predictions = model.predict(X_test)

predictions = np.clip(
    predictions,
    0,
    125
)

print("Predictions generated.")


# =========================================================
# 10. Calculate evaluation metrics
# =========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


# =========================================================
# 11. NASA-style RUL score
# =========================================================

errors = predictions - y_test

score = 0.0

for error in errors:

    if error < 0:

        score += (
            np.exp(-error / 13) - 1
        )

    else:

        score += (
            np.exp(error / 10) - 1
        )


# =========================================================
# 12. Results
# =========================================================

print("\n")
print("=" * 60)
print("HELD-OUT TEST SET EVALUATION")
print("=" * 60)

print(
    f"Test engines: "
    f"{test_df['unit'].nunique()}"
)

print(
    f"Test observations: "
    f"{len(test_df)}"
)

print()

print(
    f"MAE: "
    f"{mae:.3f} cycles"
)

print(
    f"RMSE: "
    f"{rmse:.3f} cycles"
)

print(
    f"R²: "
    f"{r2:.4f}"
)

print(
    f"NASA RUL Score: "
    f"{score:.3f}"
)

print("=" * 60)