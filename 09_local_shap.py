import pandas as pd
import numpy as np
import joblib
import shap

model = joblib.load(
    "models/rul_random_forest_capped.pkl"
)

print("Model loaded.")

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

X_inference = test_df[top_features]


# ---------------------------------------------------------
# Select the latest observation for Engine 34
# ---------------------------------------------------------

engine_id = int(input("Enter engine number: "))

engine_data = (
    test_df[test_df["unit"] == engine_id]
    .sort_values("cycle")
)

if engine_data.empty:
    print(f"Engine {engine_id} not found.")
    exit()

latest_row = engine_data.tail(1)
X_engine = latest_row[top_features]




# ---------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_engine)

shap_values = np.asarray(shap_values).flatten()


# ---------------------------------------------------------
# Create explanation table
# ---------------------------------------------------------

explanation = pd.DataFrame({
    "feature": top_features,
    "feature_value": X_engine.iloc[0].values,
    "shap_value": shap_values
})

explanation["abs_shap"] = (
    explanation["shap_value"].abs()
)

explanation = explanation.sort_values(
    "abs_shap",
    ascending=False
)

# Features pushing RUL downward
negative_factors = (
    explanation[explanation["shap_value"] < 0]
    .sort_values("shap_value")
)

# Features pushing RUL upward
positive_factors = (
    explanation[explanation["shap_value"] > 0]
    .sort_values("shap_value", ascending=False)
)

print("\nFactors pushing RUL DOWN")
print("------------------------")
print(
    negative_factors[
        ["feature", "feature_value", "shap_value"]
    ].head(10).to_string(index=False)
)

print("\nFactors pushing RUL UP")
print("----------------------")
print(
    positive_factors[
        ["feature", "feature_value", "shap_value"]
    ].head(10).to_string(index=False)
)