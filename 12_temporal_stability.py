import pandas as pd
import numpy as np
import joblib


# =========================================================
# 1. Load trained model and feature list
# =========================================================

model = joblib.load(
    "models/rul_random_forest_capped.pkl"
)

top_features = joblib.load(
    "models/rul_features.pkl"
)

print("Model loaded.")


# =========================================================
# 2. Load training data
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


# =========================================================
# 3. Sensor columns
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
# 4. Lag features
# =========================================================

for lag in [1, 2, 3]:

    for col in sensor_cols:

        name = f"{col}_lag{lag}"

        train_df[name] = (
            train_df.groupby("unit")[col]
            .shift(lag)
        )


# =========================================================
# 5. Rolling mean and standard deviation
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
# 6. Rolling slope
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
# 7. Remove incomplete rows
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
# 8. Generate predictions
# =========================================================

X = train_df[top_features]

train_df["predicted_RUL"] = (
    model.predict(X)
)

print("Predictions generated.")


# =========================================================
# 9. Sort chronologically
# =========================================================

train_df = train_df.sort_values(
    ["unit", "cycle"]
).copy()


# =========================================================
# 10. Calculate prediction change
# =========================================================

train_df["rul_change"] = (
    train_df
    .groupby("unit")["predicted_RUL"]
    .diff()
)


# =========================================================
# 11. Identify upward prediction movements
# =========================================================

train_df["prediction_increased"] = (
    train_df["rul_change"] > 0
)


# =========================================================
# 12. Absolute prediction movement
# =========================================================

train_df["absolute_change"] = (
    train_df["rul_change"].abs()
)


# =========================================================
# 13. Calculate engine-level stability statistics
# =========================================================

stability = (
    train_df
    .groupby("unit")
    .agg(
        mean_absolute_change=(
            "absolute_change",
            "mean"
        ),

        max_upward_jump=(
            "rul_change",
            "max"
        ),

        max_downward_jump=(
            "rul_change",
            "min"
        ),

        percentage_increases=(
            "prediction_increased",
            "mean"
        )
    )
    .reset_index()
)


stability["percentage_increases"] *= 100


# =========================================================
# 14. Print engine-level results
# =========================================================

print("\n")
print("=" * 65)
print("TEMPORAL PREDICTION STABILITY")
print("=" * 65)

print(
    stability.to_string(
        index=False
    )
)


# =========================================================
# 15. Overall statistics
# =========================================================

print("\n")
print("Overall Stability Summary")
print("-------------------------")

print(
    f"Mean absolute RUL change: "
    f"{train_df['absolute_change'].mean():.3f}"
)

print(
    f"Largest upward RUL jump: "
    f"{train_df['rul_change'].max():.3f}"
)

print(
    f"Largest downward RUL jump: "
    f"{train_df['rul_change'].min():.3f}"
)

print(
    f"Percentage of predictions that increased: "
    f"{train_df['prediction_increased'].mean() * 100:.2f}%"
)


# =========================================================
# 16. Identify engines with largest instability
# =========================================================

print("\n")
print("Most Unstable Engines")
print("---------------------")

print(
    stability
    .sort_values(
        "mean_absolute_change",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


print("\n")
print("=" * 65)