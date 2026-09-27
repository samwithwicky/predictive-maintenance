import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("data/raw/train_FD001.txt" , sep=r"\s+" , header=None)
"""
print(df.shape)
print(df.head(5))
print(df.tail(5))

for col in df.columns:
    if df[col].nunique() == 1:
        print(f"Column {col}: {df[col].unique()}")
"""

columns = [
    "unit",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "T2",
    "T24",
    "T30",
    "T50",
    "P2",
    "P15",
    "P30",
    "Nf",
    "Nc",
    "epr",
    "Ps30",
    "phi",
    "NRf",
    "NRc",
    "BPR",
    "farB",
    "htBleed",
    "Nf_dmd",
    "PCNfR_dmd",
    "W31",
    "W32"
]

df.columns = columns
"""
print(df.head())
print(df.shape)
print(df.isna().sum().sum())
"""

max_cycle =df.groupby("unit")["cycle"].transform("max")

df["RUL"] = max_cycle - df["cycle"]

"""
print(df.head())
print(df["RUL"].describe())
print(
    df.groupby("unit")["RUL"]
      .agg(["min", "max"])
      .head(10)
)
"""

"""
engine =df[df["unit"] == 1]

plt.plot(engine["cycle"],engine["T30"])

plt.xlabel("Cycle")
plt.ylabel("T30")
plt.title("Engine 1 - T30 Over time")
plt.show()
"""

sensor_cols = [
    "T2", "T24", "T30", "T50",
    "P2", "P15", "P30",
    "Nf", "Nc", "epr", "Ps30",
    "phi", "NRf", "NRc", "BPR",
    "farB", "htBleed", "Nf_dmd",
    "PCNfR_dmd", "W31", "W32"
]
"""
for col in sensor_cols:
    print(f"{col:10} unique values: {df[col].nunique()}")

"""
"""
correlation = df[ sensor_cols + ["RUL"]].corr()["RUL"].drop("RUL")
print(correlation.sort_values())

correlation.sort_values().plot(kind = "barh")
plt.xlabel("Correlation with rul")
plt.title("Sensor correlation with RUL")
plt.tight_layout()
plt.show()
"""

units = df["unit"].unique()

train_units, val_units = train_test_split(units, test_size = 0.2, random_state = 42)
train_df = df[df["unit"].isin(train_units)].copy()
val_df = df[df["unit"].isin(val_units)].copy()
"""
print("Training engines:",len(train_units))
print("Valdation engines:", len(val_units))

print("Training rows", len(train_df))
print("Valdation rows", len(val_df))
"""
"""

x_train = train_df[sensor_cols]
y_train = train_df["RUL"]

x_val = val_df[sensor_cols]
y_val = val_df["RUL"]

model = RandomForestRegressor(n_estimators = 100, random_state = 42, n_jobs= -1)
model.fit(x_train,y_train)

y_pred = model.predict(x_val)

print("Actual: ",y_val.iloc[:10].values)
print("Predicted:", y_pred[:10])

mae = mean_absolute_error(y_val, y_pred)

rmse = np.sqrt(mean_squared_error(y_val,y_pred))

print("MAE:",mae)
print("RMSE:",rmse)


features_with_op = sensor_cols + [
    "op_setting_1",
    "op_setting_2",
    "op_setting_3"
]



x_train_op = train_df[features_with_op]
x_val_op = val_df[features_with_op]

model_op = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model_op.fit(x_train_op, y_train)

y_pred_op = model_op.predict(x_val_op)

mae_op = mean_absolute_error(y_val, y_pred_op)
rmse_op = np.sqrt(mean_squared_error(y_val, y_pred_op))

print("With operational settings")
print("MAE:", mae_op)
print("RMSE:", rmse_op)

"""
lag_features = []

for lag in[1,2,3]:
    for col in sensor_cols:

        lag_cols = f"{col}_lag{lag}"

        train_df[f"{col}_lag{lag}"] = (train_df.groupby("unit")[col].shift(lag))
        val_df[f"{col}_lag{lag}"] = (val_df.groupby("unit")[col].shift(lag))

        lag_features.append(lag_cols)


rolling_features = []
degradation_features = []

for window in [5,10]:
    for col in sensor_cols:

        mean_name = f"{col}_mean_{window}"

        train_df[mean_name] = train_df.groupby("unit")[col].transform(lambda x: x.rolling(window).mean())
        val_df[mean_name] = val_df.groupby("unit")[col].transform(lambda x: x.rolling(window).mean())

        rolling_features.append(mean_name)

        std_name = f"{col}_std_{window}"
        train_df[std_name] = train_df.groupby("unit")[col].transform(lambda x:x.rolling(window).std())
        val_df[std_name] = val_df.groupby("unit")[col].transform(lambda x: x.rolling(window).std())

        degradation_features.append(std_name)

for window in [10,20]:
    for col in sensor_cols:

        slope_name = f"{col}_slope_{window}"
        def calculate_slope(x):
            return x.rolling(window).apply(lambda y: np.polyfit(np.arange(len(y)),y,1)[0],raw = True)
        train_df[slope_name] = train_df.groupby("unit")[col].transform(calculate_slope)
        val_df[slope_name] = val_df.groupby("unit")[col].transform(calculate_slope)

        degradation_features.append(slope_name)


print(train_df.shape)
print(val_df.shape)

train_df = train_df.dropna().copy()
val_df = val_df.dropna().copy()

# feature_sets = {
#     "Raw sensors": sensor_cols,

#     "Raw + Lag": (
#         sensor_cols
#         + lag_features
#     ),

#     "Raw + Lag + Rolling Mean": (
#         sensor_cols
#         + lag_features
#         + rolling_features
#     ),

#     "Raw + Lag + Rolling Mean + Degradation": (
#         sensor_cols
#         + lag_features
#         + rolling_features
#         + degradation_features
#     )
# }

# ablation_results = []

# for name, features in feature_sets.items():

#     print(f"\nRunning: {name}")
#     print(f"Number of features: {len(features)}")

#     x_train = train_df[features]
#     y_train = train_df["RUL"]

#     x_val = val_df[features]
#     y_val = val_df["RUL"]

#     model = RandomForestRegressor(
#         n_estimators=100,
#         random_state=42,
#         n_jobs=-1
#     )

#     model.fit(x_train, y_train)

#     predictions = model.predict(x_val)

#     mae = mean_absolute_error(
#         y_val,
#         predictions
#     )

#     rmse = np.sqrt(
#         mean_squared_error(
#             y_val,
#             predictions
#         )
#     )

#     ablation_results.append({
#         "Feature set": name,
#         "Features": len(features),
#         "MAE": mae,
#         "RMSE": rmse
#     })


# ablation_results = pd.DataFrame(ablation_results)

# print("\nFeature Ablation Results:")
# print(ablation_results)

# print(train_df.shape)
# print(val_df.shape)



full_features = (
    sensor_cols
    + lag_features
    + rolling_features
    + degradation_features
)

x_train_full = train_df[full_features]
y_train_full = train_df["RUL"]

x_val_full = val_df[full_features]
y_val_full = val_df["RUL"]

model_full = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model_full.fit(x_train_full, y_train_full)

predictions_full = model_full.predict(x_val_full)

# mae_full = mean_absolute_error(
#     y_val_full,
#     predictions_full
# )

# rmse_full = np.sqrt(
#     mean_squared_error(
#         y_val_full,
#         predictions_full
#     )
# )

# print("\nFull Model")
# print("MAE:", mae_full)
# print("RMSE:", rmse_full)


# Feature importance
feature_importance = pd.DataFrame({
    "feature": full_features,
    "importance": model_full.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

# ---------------------------------------------------------
# Top feature model
# ---------------------------------------------------------

top_n = 60

top_features = feature_importance.head(top_n)["feature"].tolist()

print(f"\nTraining model with top {top_n} features")
print("Features:", top_features)

x_train_top = train_df[top_features]
y_train_top = train_df["RUL"]

x_val_top = val_df[top_features]
y_val_top = val_df["RUL"]

model_top = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model_top.fit(x_train_top, y_train_top)

y_pred_top = model_top.predict(x_val_top)

mae_top = mean_absolute_error(
    y_val_top,
    y_pred_top
)

rmse_top = np.sqrt(
    mean_squared_error(
        y_val_top,
        y_pred_top
    )
)

print("\nTop 60 Feature Model")
print("MAE:", mae_top)
print("RMSE:", rmse_top)


# ---------------------------------------------------------
# Error analysis - Top 60 model
# ---------------------------------------------------------

results_top = pd.DataFrame({
    "unit": val_df["unit"].values,
    "cycle": val_df["cycle"].values,
    "actual": y_val_top.values,
    "predicted": y_pred_top
})

results_top["error"] = (
    results_top["actual"] - results_top["predicted"]
)

results_top["abs_error"] = (
    results_top["error"].abs()
)

results_top["RUL_region"] = pd.cut(
    results_top["actual"],
    bins=[-1, 20, 50, 100, np.inf],
    labels=["0-20", "21-50", "51-100", "100+"]
)

print("\nError by RUL region - Top 60:")
print(
    results_top.groupby(
        "RUL_region",
        observed=False
    )["abs_error"].agg(["mean", "count"])
)

print("\nPrediction bias by RUL region - Top 60:")
print(
    results_top.groupby(
        "RUL_region",
        observed=False
    )["error"].agg(["mean", "min", "max", "count"])
)

print("\nWorst 20 predictions - Top 60:")
print(
    results_top.sort_values(
        "abs_error",
        ascending=False
    ).head(20)
)

# print("\nTop 30 Features:")
# print(feature_importance.head(30))


# # ---------------------------------------------------------
# # Plot top 20 features
# # ---------------------------------------------------------

# top_features = feature_importance.head(20)

# plt.figure(figsize=(10, 8))

# plt.barh(
#     top_features["feature"][::-1],
#     top_features["importance"][::-1]
# )

# plt.xlabel("Importance")
# plt.ylabel("Feature")
# plt.title("Top 20 Random Forest Features")

# plt.tight_layout()
# plt.show()

# ---------------------------------------------------------
# Feature selection experiment
# ---------------------------------------------------------

# feature_counts = [20, 40, 60, 100, 210]

# selection_results = []

# for n_features in feature_counts:

#     selected_features = feature_importance.head(n_features)["feature"].tolist()

#     print(f"\nRunning Top {n_features} Features")

#     x_train_selected = train_df[selected_features]
#     x_val_selected = val_df[selected_features]

#     model_selected = RandomForestRegressor(
#         n_estimators=100,
#         random_state=42,
#         n_jobs=-1
#     )

#     model_selected.fit(
#         x_train_selected,
#         y_train_full
#     )

#     predictions_selected = model_selected.predict(
#         x_val_selected
#     )

#     mae = mean_absolute_error(
#         y_val_full,
#         predictions_selected
#     )

#     rmse = np.sqrt(
#         mean_squared_error(
#             y_val_full,
#             predictions_selected
#         )
#     )

#     selection_results.append({
#         "Features": n_features,
#         "MAE": mae,
#         "RMSE": rmse
#     })


# selection_results = pd.DataFrame(selection_results)

# print("\nFeature Selection Results:")
# print(selection_results)


# temporal_features = sensor_cols + lag_features + rolling_features + degradation_features
# #print("Number of temporal features:", len(temporal_features))

# x_train_t = train_df[temporal_features]
# y_train_t = train_df["RUL"]

# x_val_t = val_df[temporal_features]
# y_val_t = val_df["RUL"]

# #print(x_train_t.shape)
# #print(x_val_t.shape)

# model_temporal = RandomForestRegressor(n_estimators= 100,random_state = 42, n_jobs = -1)
# model_temporal.fit(x_train_t,y_train_t)

# y_pred_t = model_temporal.predict(x_val_t)

# mae_temporal = mean_absolute_error(
#     y_val_t,
#     y_pred_t
# )

# rmse_temporal = np.sqrt(
#     mean_squared_error(
#         y_val_t,
#         y_pred_t
#     )
# )

# print("Temporal + Rolling Features")
# print("MAE:", mae_temporal)
# print("RMSE:", rmse_temporal)

# feature_importance = pd.DataFrame({
#     "feature":temporal_features,"importance":model_temporal.feature_importances_
# })

# feature_importance = feature_importance.sort_values("importance",ascending = False)

# top_features = feature_importance.head(20)
# plt.figure()
# plt.barh(top_features["feature"][::-1],top_features["importance"][::-1])

# plt.xlabel("feature")
# plt.ylabel("importance")
# plt.title("top 20 random forest features")
# plt.show()

# results = pd.DataFrame({
#     "unit": val_df["unit"].values,
#     "cycle": val_df["cycle"].values,
#     "actual": y_val_t.values,
#     "predicted": y_pred_t
# })

# results["error"] = results["actual"] - results["predicted"]
# results["abs_error"] = results["error"].abs()

# results["RUL_region"] = pd.cut(
#     results["actual"],
#     bins=[-1, 20, 50, 100, np.inf],
#     labels=["0-20", "21-50", "51-100", "100+"]
# )

# print("\nError by RUL region:")
# print(
#     results.groupby("RUL_region", observed=False)["abs_error"]
#     .agg(["mean", "count"])
# )

# print("\nPrediction bias by RUL region:")
# print(
#     results.groupby("RUL_region", observed=False)["error"]
#     .agg(["mean", "min", "max", "count"])
# )

# print("\nWorst 20 predictions:")
# print(
#     results.sort_values(
#         "abs_error",
#         ascending=False
#     ).head(20)
# )