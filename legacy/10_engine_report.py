import pandas as pd
import numpy as np
import joblib
import shap


# =========================================================
# 1. Load model and features
# =========================================================

model = joblib.load(
    "models/rul_random_forest_capped.pkl"
)

top_features = joblib.load(
    "models/rul_features.pkl"
)

print("Model loaded.")


# =========================================================
# 2. Ask for engine
# =========================================================

engine_id = int(
    input("Enter engine number: ")
)


# =========================================================
# 3. Load test data
# =========================================================

test_df = pd.read_csv(
    "data/raw/test_FD001.txt",
    sep=r"\s+",
    header=None
)

test_df.columns = [
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

        test_df[name] = (
            test_df.groupby("unit")[col]
            .shift(lag)
        )


# =========================================================
# 6. Rolling mean and standard deviation
# =========================================================

for window in [5, 10]:

    for col in sensor_cols:

        mean_name = f"{col}_mean_{window}"
        std_name = f"{col}_std_{window}"

        test_df[mean_name] = (
            test_df.groupby("unit")[col]
            .transform(
                lambda x:
                x.rolling(window).mean()
            )
        )

        test_df[std_name] = (
            test_df.groupby("unit")[col]
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

        test_df[name] = (
            test_df.groupby("unit")[col]
            .transform(slope)
        )


# =========================================================
# 8. Remove incomplete historical rows
# =========================================================

test_df = test_df.dropna().copy()


# =========================================================
# 9. Select requested engine
# =========================================================

engine_data = test_df[
    test_df["unit"] == engine_id
].copy()

if engine_data.empty:

    print(
        f"\nEngine {engine_id} not found "
        "or does not have enough history."
    )

    exit()


# =========================================================
# 10. Get latest observation
# =========================================================

latest_row = engine_data.tail(1)

X_engine = latest_row[top_features]

latest_cycle = latest_row[
    "cycle"
].iloc[0]


# =========================================================
# 11. Predict RUL
# =========================================================

predicted_rul = model.predict(
    X_engine
)[0]


# =========================================================
# 12. Determine risk level
# =========================================================

if predicted_rul <= 20:

    risk_level = "Maintenance Attention"

elif predicted_rul <= 50:

    risk_level = "Monitor"

else:

    risk_level = "Normal"


# =========================================================
# 13. SHAP explanation
# =========================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    X_engine
)

shap_values = np.asarray(
    shap_values
).flatten()


# =========================================================
# 14. Create explanation table
# =========================================================

explanation = pd.DataFrame({

    "feature": top_features,

    "feature_value":
        X_engine.iloc[0].values,

    "shap_value":
        shap_values

})

explanation["abs_shap"] = (
    explanation["shap_value"].abs()
)


# =========================================================
# 15. Separate factors
# =========================================================

factors_down = (
    explanation[
        explanation["shap_value"] < 0
    ]
    .sort_values(
        "shap_value"
    )
)

factors_up = (
    explanation[
        explanation["shap_value"] > 0
    ]
    .sort_values(
        "shap_value",
        ascending=False
    )
)


# =========================================================
# 16. Print report
# =========================================================

print("\n")
print("=" * 55)
print("ENGINE HEALTH REPORT")
print("=" * 55)

print(f"Engine: {engine_id}")
print(f"Latest cycle: {latest_cycle}")
print(
    f"Predicted RUL: "
    f"{predicted_rul:.2f} cycles"
)
print(f"Risk level: {risk_level}")


# =========================================================
# 17. Factors pushing RUL down
# =========================================================

print("\nFactors pushing RUL DOWN")
print("-" * 30)

if factors_down.empty:

    print("None")

else:

    print(
        factors_down[
            [
                "feature",
                "feature_value",
                "shap_value"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


# =========================================================
# 18. Factors pushing RUL up
# =========================================================

print("\nFactors pushing RUL UP")
print("-" * 30)

if factors_up.empty:

    print("None")

else:

    print(
        factors_up[
            [
                "feature",
                "feature_value",
                "shap_value"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


print("\n" + "=" * 55)