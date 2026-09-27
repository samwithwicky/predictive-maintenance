import pandas as pd
import numpy as np
import joblib
import shap
import streamlit as st
import matplotlib.pyplot as plt


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "models/rul_random_forest_capped.pkl"
FEATURE_PATH = "models/rul_features.pkl"
DATA_PATH = "data/raw/test_FD001.txt"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)

    return model, features


model, top_features = load_model()


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

            grouped = df.groupby("unit")[col]

            rolling_features[
                f"{col}_mean_{window}"
            ] = grouped.transform(
                lambda x:
                x.rolling(window).mean()
            )

            rolling_features[
                f"{col}_std_{window}"
            ] = grouped.transform(
                lambda x:
                x.rolling(window).std()
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

    def calculate_slope(x, window):

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
                .transform(
                    lambda x:
                    calculate_slope(
                        x,
                        window
                    )
                )
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


    # -----------------------------------------------------
    # Remove insufficient-history rows
    # -----------------------------------------------------

    df = df.dropna().copy()

    return df


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None
    )

    df.columns = columns

    return df


raw_df = load_data()


# =========================================================
# FEATURE ENGINEERING
# =========================================================

@st.cache_data
def prepare_data(df):

    return create_features(df)


engine_data = prepare_data(raw_df)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Predictive Maintenance")

available_engines = sorted(
    engine_data["unit"]
    .astype(int)
    .unique()
    .tolist()
)

engine_id = st.sidebar.selectbox(
    "Select Engine",
    available_engines
)


# =========================================================
# SELECT ENGINE
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
# RUL PREDICTION
# =========================================================

predicted_rul = float(
    model.predict(X_engine)[0]
)

predicted_rul = max(
    0,
    predicted_rul
)


# =========================================================
# RISK CLASSIFICATION
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
# PAGE HEADER
# =========================================================

st.title(
    "Predictive Maintenance Dashboard"
)

st.caption(
    "Remaining Useful Life Prediction and "
    "Explainable Maintenance Monitoring"
)


# =========================================================
# KPI CARDS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Engine",
        engine_id
    )

with col2:

    st.metric(
        "Latest Cycle",
        latest_cycle
    )

with col3:

    st.metric(
        "Predicted RUL",
        f"{predicted_rul:.1f} cycles"
    )


# =========================================================
# RISK STATUS
# =========================================================

st.subheader("Current Maintenance Status")

if risk_level == "Maintenance Attention":

    st.error(
        "🔴 MAINTENANCE ATTENTION"
    )

    st.write(
        "The predicted remaining useful life "
        "is below the maintenance threshold."
    )

elif risk_level == "Monitor":

    st.warning(
        "🟡 MONITOR"
    )

    st.write(
        "The engine should be monitored closely."
    )

else:

    st.success(
        "🟢 NORMAL"
    )

    st.write(
        "The engine is currently within "
        "the normal predicted RUL range."
    )


# =========================================================
# RUL TRAJECTORY
# =========================================================

st.subheader("RUL Prediction Trajectory")


# Generate predictions for every usable cycle
trajectory_predictions = model.predict(
    engine_history[top_features]
)


trajectory = engine_history[
    ["cycle"]
].copy()

trajectory["predicted_RUL"] = (
    trajectory_predictions
)


fig, ax = plt.subplots()

ax.plot(
    trajectory["cycle"],
    trajectory["predicted_RUL"],
    label="Predicted RUL"
)

ax.axhline(
    50,
    linestyle="--",
    label="Monitor threshold"
)

ax.axhline(
    20,
    linestyle="--",
    label="Maintenance threshold"
)

ax.set_xlabel("Cycle")

ax.set_ylabel(
    "Predicted RUL"
)

ax.set_title(
    f"Engine {engine_id} RUL Trajectory"
)

ax.legend()

st.pyplot(fig)


# =========================================================
# SHAP EXPLANATION
# =========================================================

st.subheader(
    "Factors Affecting Current RUL Prediction"
)


explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    X_engine
)

shap_values = np.asarray(
    shap_values
).flatten()


explanation = pd.DataFrame({

    "Feature":
        top_features,

    "Value":
        X_engine.iloc[0].values,

    "SHAP":
        shap_values

})


explanation["Magnitude"] = (
    explanation["SHAP"].abs()
)


explanation = explanation.sort_values(
    "Magnitude",
    ascending=False
)


top_explanations = (
    explanation.head(10).copy()
)


top_explanations["Effect"] = (
    top_explanations["SHAP"]
    .apply(
        lambda x:
        "Increases RUL"
        if x > 0
        else "Decreases RUL"
    )
)


st.dataframe(
    top_explanations[
        [
            "Feature",
            "Value",
            "Effect",
            "SHAP"
        ]
    ],
    use_container_width=True,
    hide_index=True
)


# =========================================================
# SUMMARY
# =========================================================

st.subheader("System Summary")

st.write(
    f"Engine **{engine_id}** is currently at "
    f"cycle **{latest_cycle}** with an estimated "
    f"remaining useful life of approximately "
    f"**{predicted_rul:.1f} cycles**."
)