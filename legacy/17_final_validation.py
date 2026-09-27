import os
import joblib
import pandas as pd
import numpy as np
import shap


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "models/rul_random_forest_capped.pkl"
FEATURE_PATH = "models/rul_features.pkl"
DATA_PATH = "data/raw/test_FD001.txt"


# =========================================================
# VALIDATION HEADER
# =========================================================

print("\n")
print("=" * 60)
print("FINAL MODEL VALIDATION")
print("=" * 60)


# =========================================================
# 1. CHECK REQUIRED FILES
# =========================================================

print("\n[1] Checking required files...")

required_files = [
    MODEL_PATH,
    FEATURE_PATH,
    DATA_PATH
]

for path in required_files:

    if os.path.exists(path):

        print(f"PASS: {path}")

    else:

        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


# =========================================================
# 2. LOAD MODEL AND FEATURES
# =========================================================

print("\n[2] Loading model artifacts...")

model = joblib.load(MODEL_PATH)

top_features = joblib.load(FEATURE_PATH)

print("PASS: Model loaded.")
print("PASS: Feature list loaded.")


# =========================================================
# 3. CHECK FEATURE LIST
# =========================================================

print("\n[3] Validating feature list...")

if not isinstance(top_features, list):

    top_features = list(top_features)


print(
    f"Number of saved features: "
    f"{len(top_features)}"
)


if len(top_features) != 60:

    raise ValueError(
        f"Expected 60 features, "
        f"found {len(top_features)}"
    )


if len(set(top_features)) != len(top_features):

    raise ValueError(
        "Duplicate features found."
    )


print("PASS: Exactly 60 unique features.")


# =========================================================
# 4. CHECK MODEL FEATURE COUNT
# =========================================================

print("\n[4] Checking model input size...")

model_feature_count = (
    model.n_features_in_
)


print(
    f"Model expects: "
    f"{model_feature_count} features"
)


if model_feature_count != len(top_features):

    raise ValueError(
        "Model feature count does not "
        "match saved feature list."
    )


print("PASS: Model and feature list match.")


# =========================================================
# 5. LOAD TEST DATA
# =========================================================

print("\n[5] Loading test data...")

columns = [
    "unit", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "T2", "T24", "T30", "T50",
    "P2", "P15", "P30",
    "Nf", "Nc", "epr", "Ps30",
    "phi", "NRf", "NRc", "BPR",
    "farB", "htBleed", "Nf_dmd",
    "PCNfR_dmd", "W31", "W32"
]


sensor_cols = [
    "T2", "T24", "T30", "T50",
    "P2", "P15", "P30",
    "Nf", "Nc", "epr", "Ps30",
    "phi", "NRf", "NRc", "BPR",
    "farB", "htBleed", "Nf_dmd",
    "PCNfR_dmd", "W31", "W32"
]


df = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    header=None
)

df.columns = columns

print(
    f"Loaded {len(df)} rows."
)


# =========================================================
# 6. RECREATE FEATURES
# =========================================================

print("\n[6] Running feature engineering...")


def create_features(df):

    df = df.copy()

    # -----------------------------------------------------
    # Lag features
    # -----------------------------------------------------

    lag_features = {}

    for lag in [1, 2, 3]:

        for col in sensor_cols:

            name = f"{col}_lag{lag}"

            lag_features[name] = (
                df.groupby("unit")[col]
                .shift(lag)
            )

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                lag_features,
                index=df.index
            )
        ],
        axis=1
    )


    # -----------------------------------------------------
    # Rolling statistics
    # -----------------------------------------------------

    rolling_features = {}

    for window in [5, 10]:

        for col in sensor_cols:

            mean_name = f"{col}_mean_{window}"
            std_name = f"{col}_std_{window}"

            grouped = df.groupby("unit")[col]

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

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                rolling_features,
                index=df.index
            )
        ],
        axis=1
    )


    # -----------------------------------------------------
    # Rolling slopes
    # -----------------------------------------------------

    slope_features = {}

    def slope(x):

        return x.rolling(window).apply(
            lambda y:
            np.polyfit(
                np.arange(len(y)),
                y,
                1
            )[0],
            raw=True
        )


    for window in [10, 20]:

        for col in sensor_cols:

            name = f"{col}_slope_{window}"

            slope_features[name] = (
                df.groupby("unit")[col]
                .transform(slope)
            )

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                slope_features,
                index=df.index
            )
        ],
        axis=1
    )


    return df.dropna().copy()


engine_data = create_features(df)

print(
    f"Rows after feature engineering: "
    f"{len(engine_data)}"
)


# =========================================================
# 7. CREATE MODEL INPUT
# =========================================================

print("\n[7] Validating model input...")

X = engine_data[top_features]

print(
    f"Input shape: {X.shape}"
)


if X.shape[1] != 60:

    raise ValueError(
        "Inference input does not contain "
        "exactly 60 features."
    )


# Check missing values

if X.isna().any().any():

    raise ValueError(
        "NaN values detected in model input."
    )


print("PASS: No NaN values.")


# =========================================================
# 8. RUN MODEL PREDICTIONS
# =========================================================

print("\n[8] Running model prediction...")

predictions = model.predict(X)

print(
    f"Generated predictions: "
    f"{len(predictions)}"
)


if len(predictions) != len(X):

    raise ValueError(
        "Prediction count does not "
        "match input row count."
    )


if not np.isfinite(predictions).all():

    raise ValueError(
        "Invalid prediction values detected."
    )


print("PASS: Predictions are valid.")


# =========================================================
# 9. CHECK RUL RANGE
# =========================================================

print("\n[9] Checking RUL output...")

negative_predictions = (
    predictions < 0
).sum()


print(
    f"Negative predictions: "
    f"{negative_predictions}"
)


if negative_predictions > 0:

    print(
        "WARNING: Model produced negative RUL values."
    )

else:

    print(
        "PASS: No negative RUL predictions."
    )


print(
    f"Minimum RUL: "
    f"{predictions.min():.2f}"
)

print(
    f"Maximum RUL: "
    f"{predictions.max():.2f}"
)


# =========================================================
# 10. TEST SINGLE-ENGINE INFERENCE
# =========================================================

print("\n[10] Testing single-engine inference...")

engine_id = 2

engine_history = engine_data[
    engine_data["unit"] == engine_id
].sort_values("cycle")


latest_row = engine_history.tail(1)

X_engine = latest_row[top_features]

single_prediction = float(
    model.predict(X_engine)[0]
)


print(
    f"Engine: {engine_id}"
)

print(
    f"Latest cycle: "
    f"{latest_row['cycle'].iloc[0]}"
)

print(
    f"Predicted RUL: "
    f"{single_prediction:.2f}"
)


if not np.isfinite(single_prediction):

    raise ValueError(
        "Single-engine prediction is invalid."
    )


print("PASS: Single-engine inference works.")


# =========================================================
# 11. TEST RISK CLASSIFICATION
# =========================================================

print("\n[11] Testing risk classification...")


def assign_risk(rul):

    if rul <= 20:
        return "Maintenance Attention"

    elif rul <= 50:
        return "Monitor"

    else:
        return "Normal"


risk = assign_risk(
    single_prediction
)


print(
    f"Risk level: {risk}"
)


valid_risks = [
    "Maintenance Attention",
    "Monitor",
    "Normal"
]


if risk not in valid_risks:

    raise ValueError(
        "Invalid risk classification."
    )


print("PASS: Risk classification works.")


# =========================================================
# 12. TEST SHAP
# =========================================================

print("\n[12] Testing SHAP explainability...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    X_engine
)

shap_values = np.asarray(
    shap_values
).flatten()


if len(shap_values) != len(top_features):

    raise ValueError(
        "SHAP output does not match "
        "feature count."
    )


if not np.isfinite(shap_values).all():

    raise ValueError(
        "Invalid SHAP values detected."
    )


print(
    f"SHAP values generated: "
    f"{len(shap_values)}"
)

print(
    "PASS: SHAP explanation works."
)


# =========================================================
# 13. FINAL RESULT
# =========================================================

print("\n")
print("=" * 60)
print("FINAL VALIDATION RESULT")
print("=" * 60)

print()
print("ALL CORE VALIDATION CHECKS PASSED.")
print()
print("Model artifact:       VALID")
print("Feature list:         VALID")
print("Feature count:        60")
print("Inference input:      VALID")
print("Predictions:          VALID")
print("Risk classification:  VALID")
print("SHAP explainability:  VALID")
print()
print("Predictive maintenance")
print("pipeline is ready for")
print("deployment / application integration.")
print()
print("=" * 60)