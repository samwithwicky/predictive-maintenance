import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# =========================================================
# Load data
# =========================================================

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


# =========================================================
# Calculate RUL
# =========================================================

df["RUL"] = (
    df.groupby("unit")["cycle"].transform("max")
    - df["cycle"]
)


# =========================================================
# Sensor columns
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
# Same train/validation split as before
# =========================================================

units = df["unit"].unique()

train_units, val_units = train_test_split(
    units,
    test_size=0.2,
    random_state=42
)

val_df = df[
    df["unit"].isin(val_units)
].copy()


# =========================================================
# Feature engineering
# =========================================================

# ---------------------------------------------------------
# Lag features
# ---------------------------------------------------------

for lag in [1, 2, 3]:

    for col in sensor_cols:

        name = f"{col}_lag{lag}"

        val_df[name] = (
            val_df.groupby("unit")[col]
            .shift(lag)
        )


# ---------------------------------------------------------
# Rolling mean and standard deviation
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Degradation slopes
# ---------------------------------------------------------

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


# Remove incomplete windows
val_df = val_df.dropna().copy()


# =========================================================
# Load final model and feature list
# =========================================================

model = joblib.load(
    "models/rul_random_forest_capped.pkl"
)

top_features = joblib.load(
    "models/rul_features.pkl"
)


# =========================================================
# Prepare validation features
# =========================================================

X_val = val_df[top_features]


print("SHAP Analysis")
print("------------")
print("Validation samples:", len(X_val))
print("Features:", len(top_features))


# =========================================================
# SHAP Tree Explainer
# =========================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_val)


print("SHAP values calculated.")


# =========================================================
# Global SHAP summary
# =========================================================

shap.summary_plot(
    shap_values,
    X_val,
    show=False
)

plt.title("SHAP Feature Importance")

plt.tight_layout()

plt.savefig(
    "observations/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# SHAP dependence plot for the most important feature
shap.dependence_plot(
    "T50_mean_10",
    shap_values,
    X_val
)