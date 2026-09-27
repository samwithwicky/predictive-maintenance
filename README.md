````markdown
# Predictive Maintenance – Remaining Useful Life Prediction

## 1. Project Overview

This project develops a machine learning system for **predicting the Remaining Useful Life (RUL)** of aircraft engine components using sensor data.

The project uses the **NASA C-MAPSS FD001 dataset** and a **Random Forest Regressor** to estimate how many operating cycles remain before an engine reaches the end of its useful life.

The project also includes feature engineering, model evaluation, feature importance analysis, and SHAP-based explainability.

---

# 2. Objectives

The main objectives of the project are:

- Predict the Remaining Useful Life (RUL) of an engine.
- Identify sensor measurements that are useful for RUL prediction.
- Capture degradation trends using time-based features.
- Evaluate model performance using MAE and RMSE.
- Analyze model predictions and errors.
- Provide explanations for model predictions using SHAP.

---

# 3. Dataset

### Dataset

**NASA C-MAPSS FD001**

The dataset contains simulated aircraft engine degradation data collected over multiple operating cycles.

Each observation contains:

- Engine/unit ID
- Operating cycle
- 3 operating settings
- 21 sensor measurements

The target variable is calculated as:

```text
RUL = Maximum cycle of engine − Current cycle
```
````

The project currently uses:

```text
data/raw/train_FD001.txt
```

---

# 4. System Requirements

### Software

- Python 3.x
- Git
- Virtual environment (`.venv`)

### Main Python Libraries

- pandas
- NumPy
- scikit-learn
- matplotlib
- joblib
- SHAP
- pytest

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

# 5. Project Structure

```text
predictive-maintenance/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── train_FD001.txt
│
├── src/
│   ├── __init__.py
│   ├── features.py
│   ├── model.py
│   ├── inference.py
│   └── explainability.py
│
├── tests/
│   ├── test_features.py
│   ├── test_model.py
│   ├── test_inference.py
│   └── test_explainability.py
│
├── reports/
│   └── figures/
│
├── legacy/
│   ├── models/
│   └── previous project scripts
│
├── requirements.txt
├── pytest.ini
└── README.md
```

---

# 6. Feature Engineering

The raw sensor measurements are transformed into additional temporal features.

### Lag Features

Previous sensor values are included using:

```text
lag 1
lag 2
lag 3
```

### Rolling Features

Rolling statistics are calculated using:

```text
5-cycle window
10-cycle window
```

The main rolling features are:

- Rolling mean
- Rolling standard deviation

### Degradation Features

Sensor trends are captured using rolling slopes over:

```text
10-cycle window
20-cycle window
```

These features allow the model to use not only the current sensor value but also its recent behaviour.

---

# 7. Model

The primary machine learning model is:

**Random Forest Regressor**

Random Forest was selected because it can model nonlinear relationships between sensor behaviour and RUL without requiring the relationships to be explicitly defined.

The model uses the engineered feature set selected during the feature-selection stage.

---

# 8. Data Splitting

The dataset is split at the **engine level** rather than randomly splitting individual observations.

This prevents observations from the same engine appearing in both training and validation data.

The current validation split uses:

```text
Validation size: 20%
Random state: 42
```

---

# 9. Model Evaluation

The model is evaluated using:

### Mean Absolute Error (MAE)

Measures the average absolute difference between predicted and actual RUL.

```text
MAE = mean(|Actual RUL − Predicted RUL|)
```

### Root Mean Squared Error (RMSE)

Measures prediction error while giving greater weight to larger errors.

```text
RMSE = sqrt(mean((Actual RUL − Predicted RUL)²))
```

Additional analysis includes:

- Actual vs predicted RUL
- Prediction error distribution
- Error by RUL region
- Prediction bias
- Worst predictions
- Temporal prediction behaviour

---

# 10. Feature Analysis

Feature analysis was performed using:

- Sensor-to-RUL correlation
- Random Forest feature importance
- SHAP feature importance

Important engineered features include several rolling and degradation-based measurements, particularly features derived from:

- T50
- T24
- Nc
- Ps30
- BPR
- W31
- W32

The analysis shows that temporal features can provide useful information about engine degradation beyond individual sensor readings.

---

# 11. Explainability

SHAP is used to understand how individual features influence model predictions.

The SHAP analysis helps answer:

- Which features influence predictions most?
- Does a high or low feature value increase predicted RUL?
- Which features have the largest impact on individual predictions?

This makes the model easier to analyse from an engineering perspective.

---

# 12. Testing

The project includes automated tests for the main components:

```text
tests/
├── test_features.py
├── test_model.py
├── test_inference.py
└── test_explainability.py
```

Run the tests using:

```bash
pytest
```

---

# 13. Application

The project contains an application layer:

```text
app/app.py
```

The application is intended to provide a practical interface for using the trained RUL prediction system.

---

# 14. Legacy Code

Earlier experimentation and development scripts are stored under:

```text
legacy/
```

These files are retained for reference and reproducibility but are not part of the main project pipeline.

---

# 15. Current Status

The project currently includes:

- NASA C-MAPSS FD001 data processing
- RUL calculation
- Temporal feature engineering
- Random Forest RUL prediction
- Feature selection
- Model evaluation
- Error analysis
- Random Forest feature importance
- SHAP explainability
- Automated tests
- Application layer
- Organized project structure

---

# 16. Future Work

Possible future improvements include:

- Testing additional machine learning models.
- Improving temporal degradation modelling.
- Evaluating performance on additional C-MAPSS datasets.
- Improving uncertainty estimation.
- Developing a production-ready inference pipeline.
- Integrating real industrial equipment data.
- Extending the system toward an industrial predictive-maintenance application.

---

## Project Goal

The long-term goal is to develop a reliable and explainable **predictive maintenance system** that can estimate equipment health and Remaining Useful Life from operational sensor data.

```

```
