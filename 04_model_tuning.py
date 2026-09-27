import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV


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


# Feature engineering

lag_features = []

for lag in [1, 2, 3]:

    for col in sensor_cols:

        name = f"{col}_lag{lag}"

        train_df[name] = (
            train_df.groupby("unit")[col].shift(lag)
        )

        val_df[name] = (
            val_df.groupby("unit")[col].shift(lag)
        )

        lag_features.append(name)


rolling_features = []
degradation_features = []

for window in [5, 10]:

    for col in sensor_cols:

        mean_name = f"{col}_mean_{window}"
        std_name = f"{col}_std_{window}"

        train_df[mean_name] = (
            train_df.groupby("unit")[col]
            .transform(
                lambda x: x.rolling(window).mean()
            )
        )

        val_df[mean_name] = (
            val_df.groupby("unit")[col]
            .transform(
                lambda x: x.rolling(window).mean()
            )
        )

        train_df[std_name] = (
            train_df.groupby("unit")[col]
            .transform(
                lambda x: x.rolling(window).std()
            )
        )

        val_df[std_name] = (
            val_df.groupby("unit")[col]
            .transform(
                lambda x: x.rolling(window).std()
            )
        )

        rolling_features.append(mean_name)
        degradation_features.append(std_name)


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

        val_df[name] = (
            val_df.groupby("unit")[col]
            .transform(slope)
        )

        degradation_features.append(name)


# Remove incomplete windows
train_df = train_df.dropna().copy()
val_df = val_df.dropna().copy()


# Load Top-60 feature list
top_features = joblib.load(
    "models/rul_features.pkl"
)


# Prepare training and validation data
X_train = train_df[top_features]
y_train = train_df["RUL"].clip(upper=125)

X_val = val_df[top_features]
y_val = val_df["RUL"].clip(upper=125)

# Hyperparameter search

param_grid = {
    "n_estimators": [100,200,300],
    "max_depth": [None, 10,20,30],
    "min_samples_split": [2,5,10],
    "min_samples_leaf": [1,2,4],
    "max_features": ["sqrt",0.5,1.0]
}

rf = RandomForestRegressor(random_state = 42,n_jobs = -1)

search = RandomizedSearchCV(
    estimator = rf,
    param_distributions = param_grid,
    n_iter = 15,
    scoring = "neg_mean_absolute_error",
    cv = 3,
    random_state = 42,
    n_jobs = -1,
    verbose = 1
)

search.fit( X_train,y_train)

# print("\nBest parameters:")
# print(search.best_params_)

# print("\nBest CV MAE:")
# print(search.best_score_)



# ---------------------------------------------------------
# Train tuned model
# ---------------------------------------------------------

tuned_model = RandomForestRegressor(
    **search.best_params_,
    random_state=42,
    n_jobs=-1
)

tuned_model.fit(X_train, y_train)


# Predict on existing validation set
tuned_predictions = tuned_model.predict(X_val)


# Evaluate
tuned_mae = mean_absolute_error(
    y_val,
    tuned_predictions
)

tuned_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        tuned_predictions
    )
)

print("\nTuned Model")
print("-----------")
print("MAE:", tuned_mae)
print("RMSE:", tuned_rmse)

joblib.dump(
    search.best_estimator_,
    "models/rul_random_forest_capped.pkl"
)

print("Tuned model saved.")