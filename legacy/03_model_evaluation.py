import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------
# Load model and selected features
# ---------------------------------------------------------

model = joblib.load("models/rul_random_forest.pkl")
top_features = joblib.load("models/rul_features.pkl")


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(
    "data/raw/train_FD001.txt",
    sep=r"\s+",
    header=None
)

df.columns = [
    "unit", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "T2", "T24", "T30", "T50",
    "P2", "P15", "P30",
    "Nf", "Nc", "epr", "Ps30",
    "phi", "NRf", "NRc", "BPR",
    "farB", "htBleed", "Nf_dmd",
    "PCNfR_dmd", "W31", "W32"
]


# ---------------------------------------------------------
# Calculate RUL
# ---------------------------------------------------------

df["RUL"] = (
    df.groupby("unit")["cycle"].transform("max")
    - df["cycle"]
)


# ---------------------------------------------------------
# Sensor columns
# ---------------------------------------------------------

sensor_cols = [
    "T2", "T24", "T30", "T50",
    "P2", "P15", "P30",
    "Nf", "Nc", "epr", "Ps30",
    "phi", "NRf", "NRc", "BPR",
    "farB", "htBleed", "Nf_dmd",
    "PCNfR_dmd", "W31", "W32"
]


# ---------------------------------------------------------
# Same engine-level validation split
# ---------------------------------------------------------

units = df["unit"].unique()

train_units, val_units = train_test_split(
    units,
    test_size=0.2,
    random_state=42
)

val_df = df[df["unit"].isin(val_units)].copy()


# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------

lag_features = []

for lag in [1, 2, 3]:

    for col in sensor_cols:

        name = f"{col}_lag{lag}"

        val_df[name] = (
            val_df.groupby("unit")[col].shift(lag)
        )

        lag_features.append(name)


rolling_features = []
degradation_features = []


for window in [5, 10]:

    for col in sensor_cols:

        mean_name = f"{col}_mean_{window}"
        std_name = f"{col}_std_{window}"

        val_df[mean_name] = (
            val_df.groupby("unit")[col]
            .transform(
                lambda x: x.rolling(window).mean()
            )
        )

        val_df[std_name] = (
            val_df.groupby("unit")[col]
            .transform(
                lambda x: x.rolling(window).std()
            )
        )

        rolling_features.append(mean_name)
        degradation_features.append(std_name)


for window in [10, 20]:

    for col in sensor_cols:

        name = f"{col}_slope_{window}"

        def slope(x):
            return x.rolling(window).apply(
                lambda y: np.polyfit(
                    np.arange(len(y)),
                    y,
                    1
                )[0],
                raw=True
            )

        val_df[name] = (
            val_df.groupby("unit")[col]
            .transform(slope)
        )

        degradation_features.append(name)


# ---------------------------------------------------------
# Remove incomplete windows
# ---------------------------------------------------------

val_df = val_df.dropna().copy()


# ---------------------------------------------------------
# Generate predictions
# ---------------------------------------------------------

predictions = model.predict(
    val_df[top_features]
)


results = pd.DataFrame({
    "unit": val_df["unit"].values,
    "cycle": val_df["cycle"].values,
    "actual": val_df["RUL"].values,
    "predicted": predictions
})


# ---------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------

mae = mean_absolute_error(
    results["actual"],
    results["predicted"]
)

rmse = np.sqrt(
    mean_squared_error(
        results["actual"],
        results["predicted"]
    )
)

print("\nModel Evaluation")
print("----------------")
print("Features:", len(top_features))
print("MAE:", mae)
print("RMSE:", rmse)


# ---------------------------------------------------------
# Error calculations
# ---------------------------------------------------------

results["error"] = (
    results["actual"]
    - results["predicted"]
)

results["abs_error"] = (
    results["error"].abs()
)


# ---------------------------------------------------------
# Error by RUL region
# ---------------------------------------------------------

results["RUL_region"] = pd.cut(
    results["actual"],
    bins=[-1, 20, 50, 100, np.inf],
    labels=[
        "0-20",
        "21-50",
        "51-100",
        "100+"
    ]
)

print("\nError by RUL region:")

print(
    results.groupby(
        "RUL_region",
        observed=False
    )["abs_error"]
    .agg(["mean", "count"])
)


# ---------------------------------------------------------
# Prediction bias
# ---------------------------------------------------------

print("\nPrediction bias by RUL region:")

print(
    results.groupby(
        "RUL_region",
        observed=False
    )["error"]
    .agg(["mean", "min", "max", "count"])
)


# ---------------------------------------------------------
# Worst predictions
# ---------------------------------------------------------

print("\nWorst 20 predictions:")

print(
    results.sort_values(
        "abs_error",
        ascending=False
    ).head(20)
)


# ---------------------------------------------------------
# Actual vs Predicted
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    results["actual"],
    results["predicted"],
    alpha=0.4
)

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Actual vs Predicted RUL")

plt.plot(
    [0, results["actual"].max()],
    [0, results["actual"].max()],
    linestyle="--"
)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Error distribution
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    results["error"],
    bins=50
)

plt.xlabel("Prediction Error")
plt.ylabel("Count")
plt.title("RUL Prediction Error Distribution")

plt.tight_layout()
plt.show()