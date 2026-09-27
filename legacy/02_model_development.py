import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Load data
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


# RUL
df["RUL"] = (
    df.groupby("unit")["cycle"].transform("max")
    - df["cycle"]
)


# Sensors
sensor_cols = [
    "T2", "T24", "T30", "T50",
    "P2", "P15", "P30",
    "Nf", "Nc", "epr", "Ps30",
    "phi", "NRf", "NRc", "BPR",
    "farB", "htBleed", "Nf_dmd",
    "PCNfR_dmd", "W31", "W32"
]


# Split by engine
units = df["unit"].unique()

train_units, val_units = train_test_split(
    units,
    test_size=0.2,
    random_state=42
)

train_df = df[df["unit"].isin(train_units)].copy()
val_df = df[df["unit"].isin(val_units)].copy()


# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------

lag_features = []

for lag in [1, 2, 3]:
    for col in sensor_cols:

        name = f"{col}_lag{lag}"

        train_df[name] = train_df.groupby("unit")[col].shift(lag)
        val_df[name] = val_df.groupby("unit")[col].shift(lag)

        lag_features.append(name)


rolling_features = []
degradation_features = []

for window in [5, 10]:
    for col in sensor_cols:

        mean_name = f"{col}_mean_{window}"
        std_name = f"{col}_std_{window}"

        train_df[mean_name] = train_df.groupby("unit")[col].transform(
            lambda x: x.rolling(window).mean()
        )

        val_df[mean_name] = val_df.groupby("unit")[col].transform(
            lambda x: x.rolling(window).mean()
        )

        train_df[std_name] = train_df.groupby("unit")[col].transform(
            lambda x: x.rolling(window).std()
        )

        val_df[std_name] = val_df.groupby("unit")[col].transform(
            lambda x: x.rolling(window).std()
        )

        rolling_features.append(mean_name)
        degradation_features.append(std_name)


for window in [10, 20]:
    for col in sensor_cols:

        name = f"{col}_slope_{window}"

        def slope(x):
            return x.rolling(window).apply(
                lambda y: np.polyfit(
                    np.arange(len(y)), y, 1
                )[0],
                raw=True
            )

        train_df[name] = train_df.groupby("unit")[col].transform(slope)
        val_df[name] = val_df.groupby("unit")[col].transform(slope)

        degradation_features.append(name)


# Remove incomplete windows
train_df = train_df.dropna()
val_df = val_df.dropna()


# All available features
features = (
    sensor_cols
    + lag_features
    + rolling_features
    + degradation_features
)


# ---------------------------------------------------------
# Train full model
# ---------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(
    train_df[features],
    train_df["RUL"]
)

predictions = model.predict(
    val_df[features]
)


# ---------------------------------------------------------
# Feature selection
# ---------------------------------------------------------

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

top_features = (
    importance
    .sort_values("importance", ascending=False)
    .head(60)["feature"]
    .tolist()
)


# ---------------------------------------------------------
# Train final Top-60 model
# ---------------------------------------------------------

final_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

final_model.fit(
    train_df[top_features],
    train_df["RUL"]
)

predictions = final_model.predict(
    val_df[top_features]
)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

mae = mean_absolute_error(
    val_df["RUL"],
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        val_df["RUL"],
        predictions
    )
)

print("Top-60 Random Forest")
print("Features:", len(top_features))
print("MAE:", mae)
print("RMSE:", rmse)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

joblib.dump(
    final_model,
    "models/rul_random_forest.pkl"
)

joblib.dump(
    top_features,
    "models/rul_features.pkl"
)

print("Model saved.")