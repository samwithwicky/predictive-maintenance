import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error

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

rul_df = pd.read_csv(
    "data/raw/RUL_FD001.txt",
    sep=r"\s+",
    header=None
)

rul_df.columns = ["RUL"]


max_cycles = (
    test_df.groupby("unit")["cycle"]
    .max()
)

rul_df.index = range(1, len(rul_df) + 1)

test_df["RUL"] = (
    test_df["unit"].map(max_cycles)
    - test_df["cycle"]
    + test_df["unit"].map(rul_df["RUL"])
)

test_df["RUL"] = test_df["RUL"].clip(upper=125)


# print("Test samples:", len(test_df))
# print("Test engines:", test_df["unit"].nunique())
# print(test_df.head())


sensor_cols = [
    "T2", "T24", "T30", "T50",
    "P2", "P15", "P30",
    "Nf", "Nc", "epr", "Ps30",
    "phi", "NRf", "NRc", "BPR",
    "farB", "htBleed", "Nf_dmd",
    "PCNfR_dmd", "W31", "W32"
]

for lag in [1, 2, 3]:

    for col in sensor_cols:

        name = f"{col}_lag{lag}"

        test_df[name] = (
            test_df.groupby("unit")[col].shift(lag)
        )


for window in [5, 10]:

    for col in sensor_cols:

        mean_name = f"{col}_mean_{window}"
        std_name = f"{col}_std_{window}"

        test_df[mean_name] = (
            test_df.groupby("unit")[col]
            .transform(
                lambda x: x.rolling(window).mean()
            )
        )

        test_df[std_name] = (
            test_df.groupby("unit")[col]
            .transform(
                lambda x: x.rolling(window).std()
            )
        )


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

        test_df[name] = (
            test_df.groupby("unit")[col]
            .transform(slope)
        )


# Remove rows where the required history is unavailable
test_df = test_df.dropna().copy()


model = joblib.load(
    "models/rul_random_forest_capped.pkl"
)

top_features = joblib.load(
    "models/rul_features.pkl"
)

X_test = test_df[top_features]
y_test = test_df["RUL"]


predictions = model.predict(X_test)


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

print("\nFinal Test Evaluation")
print("---------------------")
print("MAE:", mae)
print("RMSE:", rmse)


# ---------------------------------------------------------
# Error analysis by RUL region
# ---------------------------------------------------------

results = pd.DataFrame({
    "unit": test_df["unit"].values,
    "cycle": test_df["cycle"].values,
    "actual": y_test.values,
    "predicted": predictions
})

results["error"] = (
    results["actual"] - results["predicted"]
)

results["abs_error"] = (
    results["error"].abs()
)

results["RUL_region"] = pd.cut(
    results["actual"],
    bins=[-1, 20, 50, 100, np.inf],
    labels=["0-20", "21-50", "51-100", "100+"]
)

print("\nError by RUL region:")
print(
    results.groupby(
        "RUL_region",
        observed=False
    )["abs_error"].agg(["mean", "count"])
)

# ---------------------------------------------------------
# Prediction bias by RUL region
# ---------------------------------------------------------

print("\nPrediction bias by RUL region:")

print(
    results.groupby(
        "RUL_region",
        observed=False
    )["error"].agg(
        ["mean", "min", "max", "count"]
    )
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
# Inspect one problematic engine
# ---------------------------------------------------------

engine = results[results["unit"] == 93]

print("\nEngine 93:")
print(
    engine[
        ["cycle", "actual", "predicted", "error"]
    ].head(20)
)

import matplotlib.pyplot as plt

plt.figure()

plt.plot(
    engine["cycle"],
    engine["actual"],
    label="Actual RUL"
)

plt.plot(
    engine["cycle"],
    engine["predicted"],
    label="Predicted RUL"
)

plt.xlabel("Cycle")
plt.ylabel("RUL")
plt.title("Engine 93 - Actual vs Predicted RUL")
plt.legend()
plt.show()