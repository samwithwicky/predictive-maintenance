import pandas as pd
import numpy as np
import joblib


# Load trained model
model = joblib.load(
    "models/rul_random_forest_capped.pkl"
)

# Load the feature list used during training
top_features = joblib.load(
    "models/rul_features.pkl"
)


# Load engine sensor data
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


# Sensor columns used for feature engineering
sensor_cols = [
    "T2", "T24", "T30", "T50",
    "P2", "P15", "P30",
    "Nf", "Nc", "epr", "Ps30",
    "phi", "NRf", "NRc", "BPR",
    "farB", "htBleed", "Nf_dmd",
    "PCNfR_dmd", "W31", "W32"
]

# Lag features
for lag in [1, 2, 3]:

    for col in sensor_cols:

        name = f"{col}_lag{lag}"

        test_df[name] = (
            test_df.groupby("unit")[col]
            .shift(lag)
        )


# Rolling mean and standard deviation
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


# Rolling slope
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


# Remove rows without enough historical data
test_df = test_df.dropna().copy()

# print("Feature engineering complete.")
# print("Rows after feature engineering:", len(test_df))

# Select the exact 60 features used during training
X_inference = test_df[top_features]

# print("Inference data prepared.")
# print("Samples:", len(X_inference))
# print("Features:", X_inference.shape[1])

# Generate RUL predictions
predictions = model.predict(X_inference)

# print("Predictions generated.")
# print("Number of predictions:", len(predictions))

# Attach predictions to the corresponding rows
test_df["predicted_RUL"] = predictions

# Keep only the latest cycle for each engine
latest_predictions = (
    test_df
    .sort_values(["unit", "cycle"])
    .groupby("unit")
    .tail(1)
)

RUL_NORMAL = 50
RUL_MONITOR = 20

def classify_risk(rul):

    if rul <= 0:
        return "Critical"

    elif rul <= RUL_MONITOR:
        return "Maintenance Attention"

    elif rul <= RUL_NORMAL:
        return "Monitor"

    else:
        return "Normal"


latest_predictions["risk_level"] = (
    latest_predictions["predicted_RUL"]
    .apply(classify_risk)
)

print("\nFleet Maintenance Status")
print("-------------------------")

print(
    latest_predictions[
        ["unit", "cycle", "predicted_RUL", "risk_level"]
    ].sort_values("predicted_RUL")
    .to_string(index=False)
)

print("\nRisk Distribution")
print("-----------------")

print(
    latest_predictions["risk_level"]
    .value_counts()
)