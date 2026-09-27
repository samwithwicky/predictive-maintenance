import pandas as pd
import numpy as np
import joblib
import shap


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "models/rul_random_forest_capped.pkl"
FEATURE_PATH = "models/rul_features.pkl"

DATA_PATH = "data/raw/test_FD001.txt"


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(MODEL_PATH)
top_features = joblib.load(FEATURE_PATH)

print("Model loaded.")


# =========================================================
# COLUMN DEFINITIONS
# =========================================================

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


# =========================================================
# FEATURE ENGINEERING
# =========================================================

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

    lag_df = pd.DataFrame(
        lag_features,
        index=df.index
    )

    df = pd.concat(
        [df, lag_df],
        axis=1
    )


    # -----------------------------------------------------
    # Rolling mean and standard deviation
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

    rolling_df = pd.DataFrame(
        rolling_features,
        index=df.index
    )

    df = pd.concat(
        [df, rolling_df],
        axis=1
    )


    # -----------------------------------------------------
    # Rolling slope
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

    slope_df = pd.DataFrame(
        slope_features,
        index=df.index
    )

    df = pd.concat(
        [df, slope_df],
        axis=1
    )


    # -----------------------------------------------------
    # Remove rows without enough history
    # -----------------------------------------------------

    df = df.dropna().copy()

    return df


# =========================================================
# LOAD ENGINE DATA
# =========================================================

df = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    header=None
)

df.columns = columns

print("Engine data loaded.")


# =========================================================
# FEATURE ENGINEERING
# =========================================================

engine_data = create_features(df)

print(
    "Feature engineering complete."
)

print(
    "Rows available:",
    len(engine_data)
)


# =========================================================
# SELECT ENGINE
# =========================================================

available_engines = sorted(
    engine_data["unit"].astype(int).unique().tolist()
)

print("\nAvailable engines:")
print(available_engines)

engine_id = int(
    input("\nEnter engine number: ")
)


if engine_id not in available_engines:

    raise ValueError(
        f"Engine {engine_id} not found."
    )


# =========================================================
# GET LATEST ENGINE OBSERVATION
# =========================================================

engine_history = engine_data[
    engine_data["unit"] == engine_id
].sort_values("cycle")


latest_row = engine_history.tail(1)

latest_cycle = int(
    latest_row["cycle"].iloc[0]
)


X_engine = latest_row[top_features]


# =========================================================
# PREDICT RUL
# =========================================================

predicted_rul = float(
    model.predict(X_engine)[0]
)

predicted_rul = max(
    0,
    predicted_rul
)


# =========================================================
# ASSIGN RISK
# =========================================================

def assign_risk(rul):

    if rul <= 20:
        return "Maintenance Attention"

    elif rul <= 50:
        return "Monitor"

    else:
        return "Normal"


risk_level = assign_risk(
    predicted_rul
)


# =========================================================
# SHAP EXPLANATION
# =========================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    X_engine
)

shap_values = np.asarray(
    shap_values
).flatten()


explanation = pd.DataFrame({

    "feature":
        top_features,

    "feature_value":
        X_engine.iloc[0].values,

    "shap_value":
        shap_values

})


explanation["abs_shap"] = (
    explanation["shap_value"].abs()
)


explanation = explanation.sort_values(
    "abs_shap",
    ascending=False
)


# =========================================================
# FINAL REPORT
# =========================================================

print("\n")
print("=" * 60)
print("PREDICTIVE MAINTENANCE REPORT")
print("=" * 60)

print(
    f"Engine: {engine_id}"
)

print(
    f"Latest cycle: {latest_cycle}"
)

print(
    f"Predicted RUL: "
    f"{predicted_rul:.1f} cycles"
)

print(
    f"Risk level: "
    f"{risk_level}"
)


print("\n")
print("-" * 60)
print("TOP CONTRIBUTING FACTORS")
print("-" * 60)


top_explanations = explanation.head(10)


for _, row in top_explanations.iterrows():

    direction = (
        "DECREASES RUL"
        if row["shap_value"] < 0
        else "INCREASES RUL"
    )

    print(
        f"{row['feature']:<20} "
        f"{direction:<5} "
        f"{abs(row['shap_value']):>8.3f}"
    )


print("\n")
print("-" * 60)
print("INTERPRETATION")
print("-" * 60)


if risk_level == "Maintenance Attention":

    print(
        "Maintenance attention is recommended."
    )

elif risk_level == "Monitor":

    print(
        "Engine should be monitored closely."
    )

else:

    print(
        "Engine is currently within the normal range."
    )


print("\n")
print("=" * 60)