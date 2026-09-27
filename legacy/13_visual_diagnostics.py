import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os


# =========================================================
# 1. Load model
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
# 3. Calculate actual RUL
# =========================================================

max_cycle = (
    train_df
    .groupby("unit")["cycle"]
    .transform("max")
)

train_df["actual_RUL"] = (
    max_cycle - train_df["cycle"]
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
# 6. Rolling statistics
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
# 7. Rolling slopes
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
# 8. Remove incomplete rows
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
# 9. Generate predictions
# =========================================================

X = train_df[top_features]

train_df["predicted_RUL"] = (
    model.predict(X)
)

print("Predictions generated.")


# =========================================================
# 10. Create output directory
# =========================================================

output_dir = "outputs/diagnostics"

os.makedirs(
    output_dir,
    exist_ok=True
)


# =========================================================
# 11. Select engines to visualize
# =========================================================

engines = [
    1,
    23,
    34,
    46,
    77
]


# =========================================================
# 12. Plot each engine
# =========================================================

for engine_id in engines:

    engine = train_df[
        train_df["unit"] == engine_id
    ].copy()

    if engine.empty:
        print(
            f"Engine {engine_id} not found."
        )
        continue

    engine = engine.sort_values(
        "cycle"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        engine["cycle"],
        engine["actual_RUL"],
        label="Actual RUL"
    )

    plt.plot(
        engine["cycle"],
        engine["predicted_RUL"],
        label="Predicted RUL"
    )

    plt.xlabel("Cycle")

    plt.ylabel("RUL")

    plt.title(
        f"Engine {engine_id} — Actual vs Predicted RUL"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    filename = (
        f"engine_{engine_id}_rul.png"
    )

    filepath = os.path.join(
        output_dir,
        filename
    )

    plt.savefig(
        filepath,
        dpi=150
    )

    plt.close()

    print(
        f"Saved: {filepath}"
    )


# =========================================================
# 13. Create combined plot
# =========================================================

plt.figure(
    figsize=(12, 7)
)

for engine_id in engines:

    engine = train_df[
        train_df["unit"] == engine_id
    ].copy()

    if engine.empty:
        continue

    engine = engine.sort_values(
        "cycle"
    )

    plt.plot(
        engine["cycle"],
        engine["predicted_RUL"],
        label=f"Engine {engine_id}"
    )


plt.xlabel("Cycle")

plt.ylabel("Predicted RUL")

plt.title(
    "Predicted RUL Trajectories"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

combined_path = os.path.join(
    output_dir,
    "combined_rul_trajectories.png"
)

plt.savefig(
    combined_path,
    dpi=150
)

plt.close()

print(
    f"Saved: {combined_path}"
)


# =========================================================
# 14. Final message
# =========================================================

print("\n")
print("=" * 60)
print("VISUAL DIAGNOSTICS COMPLETE")
print("=" * 60)

print(
    f"Plots saved to: {output_dir}"
)