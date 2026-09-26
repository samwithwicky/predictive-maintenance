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

x_train = train_df[sensor_cols]
y_train = train_df["RUL"]

x_val = val_df[sensor_cols]
y_val = val_df["RUL"]

model = RandomForestRegressor(n_estimators = 100, random_state = 42, n_jobs= -1)
model.fit(x_train,y_train)

y_pred = model.predict(x_val)
"""
print("Actual: ",y_val.iloc[:10].values)
print("Predicted:", y_pred[:10])
"""
mae = mean_absolute_error(y_val, y_pred)

rmse = np.sqrt(mean_squared_error(y_val,y_pred))

print("MAE:",mae)
print("RMSE:",rmse)

"""
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