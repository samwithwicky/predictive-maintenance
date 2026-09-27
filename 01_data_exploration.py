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


lag_features = sensor_cols

for lag in[1,2,3]:
    for col in lag_features:
        train_df[f"{col}_lag{lag}"] = (train_df.groupby("unit")[col].shift(lag))
        val_df[f"{col}_lag{lag}"] = (val_df.groupby("unit")[col].shift(lag))

rolling_features = []

for window in [5,10]:
    for col in sensor_cols:

        feature_name = f"{col}_mean_{window}"

        train_df[feature_name] = train_df.groupby("unit")[col].transform(lambda x: x.rolling(window).mean())
        val_df[feature_name] = val_df.groupby("unit")[col].transform(lambda x: x.rolling(window).mean())

        rolling_features.append(feature_name)



print(train_df.shape)
print(val_df.shape)

train_df = train_df.dropna().copy()
val_df = val_df.dropna().copy()

print(train_df.shape)
print(val_df.shape)

lag_cols = []

for lag in [1,2,3]:
    for col in sensor_cols:
        lag_cols.append(f"{col}_lag{lag}")


temporal_features = sensor_cols + lag_cols + rolling_features
#print("Number of temporal features:", len(temporal_features))

x_train_t = train_df[temporal_features]
y_train_t = train_df["RUL"]

x_val_t = val_df[temporal_features]
y_val_t = val_df["RUL"]

#print(x_train_t.shape)
#print(x_val_t.shape)

model_temporal = RandomForestRegressor(n_estimators= 100,random_state = 42, n_jobs = -1)
model_temporal.fit(x_train_t,y_train_t)

y_pred_t = model_temporal.predict(x_val_t)

mae_temporal = mean_absolute_error(
    y_val_t,
    y_pred_t
)

rmse_temporal = np.sqrt(
    mean_squared_error(
        y_val_t,
        y_pred_t
    )
)

print("Temporal + Rolling Features")
print("MAE:", mae_temporal)
print("RMSE:", rmse_temporal)

# E"""rror by RUL region
"""
results = pd.DataFrame({
    "unit": val_df["unit"].values,
    "cycle": val_df["cycle"].values,
    "actual": y_val_t.values,
    "predicted": y_pred_t
})

results["error"] = results["actual"] - results["predicted"]
results["abs_error"] = results["error"].abs()

worst_predictions = results.sort_values(
    "abs_error",
    ascending=False
).head(20)

print("\nWorst 20 predictions:")
print(worst_predictions)

worst_units = [1,31,84]

for unit in worst_units:
    engine = df[df["unit"] == unit]

    plt.figure()
    plt.plot(
        engine["cycle"],engine["T30"]
    )
    plt.xlabel("cycle")
    plt.ylabel("T30")
    plt.title(f"engine{unit} cycle v t30")
    plt.show()
"""

feature_importance = pd.DataFrame({
    "feature":temporal_features,"importance":model_temporal.feature_importances_
})

feature_importance = feature_importance.sort_values("importance",ascending = False)

print("Top 20 features")
print(df.head(20))

top_features = feature_importance.head(20)
plt.figure()

plt.barh(top_features["feature"][::-1],top_features["importance"][::-1])

plt.xlabel("feature")
plt.ylabel("importance")
plt.title("top 20 random forest features")
plt.show()

