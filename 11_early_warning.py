import pandas as pd
import numpy as np
import joblib


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
# 2. Load complete run-to-failure training data
# =========================================================

train_df = pd.read_csv(
    "data/raw/train_FD001.txt",
    sep=r"\s+",
    header=None
)

train_df.columns = [
    "unit", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "T2", "T24", "T30", "T50",
    "P2", "P15", "P30",
    "Nf", "Nc", "epr", "Ps30",
    "phi", "NRf", "NRc", "BPR",
    "farB", "htBleed", "Nf_dmd",
    "PCNfR_dmd", "W31", "W32"
]

print("Training data loaded.")
print("Engines:", train_df["unit"].nunique())
print("Rows:", len(train_df))


# =========================================================
# 3. Calculate actual RUL
# =========================================================

max_cycles = (
    train_df
    .groupby("unit")["cycle"]
    .max()
)

train_df["max_cycle"] = (
    train_df["unit"].map(max_cycles)
)

train_df["actual_RUL"] = (
    train_df["max_cycle"] -
    train_df["cycle"]
)


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
# 5. Lag features
# =========================================================

for lag in [1, 2, 3]:

    for col in sensor_cols:

        name = f"{col}_lag{lag}"

        train_df[name] = (
            train_df.groupby("unit")[col]
            .shift(lag)
        )


# =========================================================
# 6. Rolling mean and standard deviation
# =========================================================

for window in [5, 10]:

    for col in sensor_cols:

        mean_name = f"{col}_mean_{window}"
        std_name = f"{col}_std_{window}"

        train_df[mean_name] = (
            train_df.groupby("unit")[col]
            .transform(
                lambda x:
                x.rolling(window).mean()
            )
        )

        train_df[std_name] = (
            train_df.groupby("unit")[col]
            .transform(
                lambda x:
                x.rolling(window).std()
            )
        )


# =========================================================
# 7. Rolling slope
# =========================================================

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

        train_df[name] = (
            train_df.groupby("unit")[col]
            .transform(slope)
        )


# =========================================================
# 8. Remove incomplete historical rows
# =========================================================

train_df = train_df.dropna().copy()

print(
    "Feature engineering complete."
)

print(
    "Rows available:",
    len(train_df)
)


# =========================================================
# 9. Prepare model input
# =========================================================

X = train_df[top_features]


# =========================================================
# 10. Generate predictions
# =========================================================

print("Generating predictions...")

train_df["predicted_RUL"] = (
    model.predict(X)
)

print("Predictions generated.")


# =========================================================
# 11. Assign risk levels
# =========================================================

def assign_risk(rul):

    if rul <= 20:
        return "Maintenance Attention"

    elif rul <= 50:
        return "Monitor"

    else:
        return "Normal"


train_df["risk_level"] = (
    train_df["predicted_RUL"]
    .apply(assign_risk)
)


# =========================================================
# 12. Find first Maintenance Attention warning
# =========================================================

warning_results = []

for unit, group in train_df.groupby("unit"):

    group = group.sort_values("cycle")

    warning_rows = group[
        group["risk_level"]
        == "Maintenance Attention"
    ]

    if warning_rows.empty:

        warning_results.append({
            "unit": unit,
            "failure_cycle":
                group["cycle"].max(),
            "first_warning_cycle": np.nan,
            "warning_lead_time": np.nan,
            "warning_detected": False
        })

    else:

        first_warning = (
            warning_rows.iloc[0]
        )

        failure_cycle = (
            group["cycle"].max()
        )

        warning_cycle = (
            first_warning["cycle"]
        )

        warning_results.append({
            "unit": unit,
            "failure_cycle":
                failure_cycle,
            "first_warning_cycle":
                warning_cycle,
            "warning_lead_time":
                failure_cycle -
                warning_cycle,
            "warning_detected": True
        })


warning_df = pd.DataFrame(
    warning_results
)


# =========================================================
# 13. Print warning results
# =========================================================

print("\n")
print("=" * 60)
print("EARLY WARNING EVALUATION")
print("=" * 60)

print(
    warning_df.to_string(
        index=False
    )
)


# =========================================================
# 14. Summary statistics
# =========================================================

detected = warning_df[
    warning_df["warning_detected"]
]

detection_rate = (
    len(detected) /
    len(warning_df)
) * 100


print("\n")
print("Warning Summary")
print("----------------")

print(
    f"Total engines: "
    f"{len(warning_df)}"
)

print(
    f"Engines with warning: "
    f"{len(detected)}"
)

print(
    f"Detection rate: "
    f"{detection_rate:.2f}%"
)


if not detected.empty:

    print(
        f"Mean warning lead time: "
        f"{detected['warning_lead_time'].mean():.2f} cycles"
    )

    print(
        f"Median warning lead time: "
        f"{detected['warning_lead_time'].median():.2f} cycles"
    )

    print(
        f"Minimum warning lead time: "
        f"{detected['warning_lead_time'].min():.0f} cycles"
    )

    print(
        f"Maximum warning lead time: "
        f"{detected['warning_lead_time'].max():.0f} cycles"
    )


print("\n")
print("=" * 60)