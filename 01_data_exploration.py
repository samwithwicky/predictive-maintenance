import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
engine =df[df["unit"] == 1]

plt.plot(engine["cycle"],engine["T30"])

plt.xlabel("Cycle")
plt.ylabel("T30")
plt.title("Engine 1 - T30 Over time")
plt.show()