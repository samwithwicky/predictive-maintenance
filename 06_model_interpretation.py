import pandas as pd
import joblib


# Load final model
model = joblib.load(
    "models/rul_random_forest_capped.pkl"
)

# Load feature names
features = joblib.load(
    "models/rul_features.pkl"
)


# Feature importance
importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})


# Sort
importance = importance.sort_values(
    "importance",
    ascending=False
)


print("Top 20 Features")
print("----------------")

print(
    importance.head(20).to_string(index=False)
)

