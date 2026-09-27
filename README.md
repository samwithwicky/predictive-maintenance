````markdown
# Predictive Maintenance & Remaining Useful Life Prediction

A machine-learning based predictive maintenance system for estimating the **Remaining Useful Life (RUL)** of industrial equipment from multivariate time-series sensor data.

The project focuses on learning degradation patterns from engine sensor measurements and transforming noisy cycle-by-cycle observations into temporal features that can be used to estimate how many operating cycles remain before failure.

The current implementation uses a **Random Forest Regressor** together with extensive temporal feature engineering, feature selection, hyperparameter tuning, and SHAP-based model interpretation.

---

## Overview

Predictive maintenance aims to move maintenance decisions from a fixed schedule or reactive failure response toward a **condition-based approach**.

Instead of asking:

> "Has the equipment failed?"

the objective is to estimate:

> "How much useful operating life does the equipment have remaining?"

This project approaches that problem as a supervised regression task.

For every engine operating cycle, the model receives sensor-derived features and predicts:

**Remaining Useful Life (RUL) = Number of operating cycles remaining until failure**

The project uses historical run-to-failure engine data to learn relationships between sensor behaviour and degradation.

---

## Project Objectives

The project was developed around several objectives:

- Explore multivariate industrial sensor data.
- Understand how individual sensors behave throughout equipment degradation.
- Calculate RUL from run-to-failure trajectories.
- Identify sensors that contain useful degradation information.
- Engineer temporal features from noisy sensor measurements.
- Compare different feature-set sizes.
- Train a Random Forest regression model for RUL prediction.
- Tune the model using cross-validation.
- Evaluate prediction error across different RUL ranges.
- Investigate temporal stability of predictions.
- Explain model behaviour using SHAP.
- Identify the sensor-derived features that contribute most strongly to RUL predictions.
- Build a cleaner and more maintainable project structure around the final ML pipeline.

---

# Dataset

The project uses the **NASA C-MAPSS FD001** run-to-failure turbofan engine dataset.

The raw training file contains multiple engines, with each engine represented by a sequence of operating cycles.

Each observation contains:

- Engine/unit identifier
- Operating cycle
- Operating settings
- Multiple sensor measurements

The dataset is treated as a time-series problem because the same engine appears repeatedly across consecutive operating cycles.

### RUL calculation

For each engine:

```text
RUL = Maximum cycle of that engine - Current cycle
````

For example, if an engine fails at cycle 200:

| Cycle | RUL |
| ----: | --: |
|     1 | 199 |
|    50 | 150 |
|   100 | 100 |
|   150 |  50 |
|   199 |   1 |
|   200 |   0 |

This converts the run-to-failure trajectories into a supervised regression target.

---

# Why Temporal Features Matter

A major observation during exploratory analysis was that individual sensor measurements can be highly noisy.

For example, raw T30 measurements from an engine fluctuate considerably from cycle to cycle.

However, the underlying behaviour becomes much clearer after applying temporal transformations such as:

* Rolling means
* Rolling standard deviations
* Lag features
* Rolling slopes

This is important because equipment degradation is not necessarily represented by a single sensor value.

The **trajectory of a sensor over time** can contain considerably more information.

The project therefore focuses heavily on temporal feature engineering.

---

# Feature Engineering

The original sensor measurements are transformed into several classes of temporal features.

## 1. Lag Features

Previous sensor observations are incorporated using lags of:

```text
1 cycle
2 cycles
3 cycles
```

For example:

```text
T50_lag1
T50_lag2
T50_lag3
```

These allow the model to capture short-term temporal behaviour.

---

## 2. Rolling Mean Features

Rolling averages are calculated using:

```text
5-cycle window
10-cycle window
```

Examples:

```text
T50_mean_5
T50_mean_10
T30_mean_10
Nc_mean_10
Ps30_mean_5
```

Rolling means reduce high-frequency sensor noise while retaining the underlying operating/degradation trend.

---

## 3. Rolling Standard Deviation

Rolling standard deviation is calculated using the same temporal windows.

Examples:

```text
T50_std_5
T50_std_10
Nc_std_5
Nc_std_10
```

These features capture changes in sensor variability.

Increasing variability can represent changing operating behaviour even when the average sensor value has not changed substantially.

---

## 4. Degradation Slope Features

Longer temporal windows are used to estimate the rate of change of sensors.

The project uses:

```text
10-cycle slope
20-cycle slope
```

Examples:

```text
T50_slope_20
Ps30_slope_20
phi_slope_20
Nf_slope_20
NRf_slope_20
```

The slope is calculated using a linear fit over the rolling window.

Conceptually:

```text
sensor value
     |
     |        /
     |      /
     |    /
     |  /
     | /
     +---------------- cycle
```

This allows the model to distinguish between:

```text
high sensor value
```

and

```text
sensor value that is rapidly increasing
```

which are not necessarily equivalent from a degradation perspective.

---

# Feature Selection

Different feature-set sizes were evaluated to determine whether increasing the number of features substantially improved model performance.

The recorded results were:

| Number of Features |       MAE |      RMSE |
| -----------------: | --------: | --------: |
|                 20 | 20.425520 | 29.930155 |
|                 40 | 20.261098 | 29.578469 |
|                 60 | 20.220038 | 29.838807 |
|                100 | 20.246667 | 29.507941 |
|                210 | 20.220657 | 29.429673 |

### Observation

Increasing the feature count produces relatively small changes in MAE.

The results suggest that a relatively compact subset of engineered features contains most of the useful predictive information, while larger feature sets primarily affect the behaviour of larger errors as reflected in RMSE.

The project therefore uses feature importance and model interpretation alongside raw feature-count comparisons rather than assuming that more features automatically produce a better model.

---

# Model

The primary predictive model is a:

## Random Forest Regressor

Random Forest was selected because it can model:

* Non-linear relationships
* Feature interactions
* Different sensor scales
* Complex relationships between temporal features and RUL

It also provides feature importance information that can be complemented by SHAP for more detailed interpretation.

The model predicts:

```text
Input:
Temporal sensor features

        ↓

Random Forest Regressor

        ↓

Predicted RUL
```

---

# Validation Strategy

The dataset is split at the **engine/unit level** rather than randomly splitting individual observations.

This is important for a run-to-failure time-series problem.

The validation procedure uses:

```text
80% of engines → training
20% of engines → validation
```

with:

```text
random_state = 42
```

The purpose is to ensure that validation engines are distinct from training engines.

This is preferable to randomly distributing individual cycles between training and validation because observations from the same engine are temporally related.

---

# Model Tuning

The Random Forest hyperparameters are tuned using `RandomizedSearchCV`.

The search includes:

```text
n_estimators:
100, 200, 300

max_depth:
None, 10, 20, 30

min_samples_split:
2, 5, 10

min_samples_leaf:
1, 2, 4

max_features:
sqrt, 0.5, 1.0
```

The optimization objective is:

```text
Negative Mean Absolute Error
```

Three-fold cross-validation is used during the randomized search.

The final tuned Random Forest is then trained using the selected parameters.

---

# Evaluation Metrics

Two primary regression metrics are used.

## Mean Absolute Error — MAE

```text
MAE = mean(|Actual RUL - Predicted RUL|)
```

MAE represents the average absolute prediction error in operating cycles.

A lower value means predictions are, on average, closer to the actual RUL.

---

## Root Mean Squared Error — RMSE

```text
RMSE = sqrt(mean((Actual RUL - Predicted RUL)^2))
```

RMSE gives greater weight to large prediction errors.

This makes it particularly useful for identifying whether a model occasionally produces large RUL estimation failures.

---

# Error Analysis

The prediction error is defined as:

```text
Error = Actual RUL - Predicted RUL
```

Therefore:

```text
Positive error → Model underpredicted RUL
Negative error → Model overpredicted RUL
```

Errors are analysed across different RUL regions:

```text
0–20 cycles
21–50 cycles
51–100 cycles
100+ cycles
```

This is important because an identical numerical error does not necessarily have the same practical significance at every point in an equipment lifecycle.

---

# Observations From Model Behaviour

## 1. Stronger behaviour at lower RUL

The Actual vs Predicted RUL analysis shows that predictions follow the general relationship between actual and predicted RUL.

The model is particularly concentrated around the expected relationship at lower RUL values.

---

## 2. Compression at high RUL

At higher actual RUL values, predictions become increasingly compressed into a narrower range.

This produces a visible tendency toward underprediction of high-RUL observations.

This behaviour is important because it means the model should not be interpreted as equally accurate across the entire lifecycle.

---

## 3. Prediction error is concentrated around zero

The temporal prediction-error distribution has its largest concentration around zero.

This indicates that a substantial portion of predictions have relatively small errors.

However, the distribution also contains longer tails, demonstrating that individual large prediction errors remain present.

---

# Sensor Analysis

Correlation analysis was used to investigate relationships between individual sensors and RUL.

Several sensors showed substantial correlation with RUL, including:

### Negative correlation

* `Ps30`
* `T50`
* `BPR`
* `T24`
* `htBleed`
* `T30`
* `Nf`
* `NRf`
* `Nc`
* `NRc`

### Positive correlation

* `phi`
* `P30`
* `W32`
* `W31`

Correlation is used here as an exploratory analysis rather than as a direct measure of model importance.

A sensor can have relatively weak linear correlation with RUL while still contributing strongly through non-linear or temporal relationships.

---

# Random Forest Feature Importance

The Random Forest feature-importance analysis identified several temporal features as particularly important.

The strongest features included:

```text
T50_mean_10
T50_mean_5
htBleed_mean_10
W32_mean_10
Nc_mean_10
Ps30_mean_5
Nc_mean_5
Ps30_slope_20
NRc_mean_10
T30_mean_10
BPR_mean_10
T24_mean_10
phi_slope_20
NRf_mean_10
Nf_slope_20
```

A notable pattern is that many of the strongest features are **engineered temporal features rather than raw sensors**.

---

# SHAP Explainability

SHAP is used to investigate how individual features influence the Random Forest predictions.

The global SHAP analysis identified features including:

```text
T50_mean_10
T24_mean_10
T50_mean_5
Nc_mean_5
Ps30_slope_20
phi_slope_20
Nc_mean_10
Nf_slope_20
BPR_mean_10
Ps30_mean_5
W31_mean_5
W31_mean_10
W32_mean_10
T30_mean_10
NRf_slope_20
P30_slope_20
Ps30_mean_10
NRc_mean_10
BPR_slope_20
T50_slope_20
```

This provides a more detailed interpretation than standard Random Forest feature importance.

---

# T50 as a Major Degradation Signal

One of the strongest observations from the analysis is the importance of:

```text
T50_mean_10
```

T50 appears repeatedly among the most important features across different analyses.

The raw T50 signal is noisy, but its rolling mean reveals a clearer degradation trajectory.

The SHAP analysis also shows that the relationship between `T50_mean_10` and the model output is non-linear.

This supports the use of temporal transformations instead of relying solely on instantaneous sensor values.

---

# Model Interpretation: Important Distinction

Feature importance should not be interpreted as:

> "This sensor directly causes the equipment to fail."

The model is learning statistical relationships from the available historical data.

For example:

```text
T50_mean_10
```

being highly influential means that the feature contributes strongly to the model's predictions.

It does not by itself establish a physical causal relationship.

This distinction is particularly important when applying machine-learning models to industrial systems.

---

# Project Architecture

The repository has been reorganized to separate the current implementation from the earlier experimental scripts.

```text
predictive-maintenance/
│
├── app/
│   └── app.py
│
├── src/
│   ├── __init__.py
│   ├── explainability.py
│   ├── features.py
│   ├── inference.py
│   └── model.py
│
├── tests/
│   ├── test_explainability.py
│   ├── test_features.py
│   ├── test_inference.py
│   └── test_model.py
│
├── reports/
│   └── figures/
│       ├── Actual vs predicted rul temporal.png
│       ├── Engine 31-T50 degradation.png
│       ├── SHAP values.png
│       ├── Sensor vs Rul.png
│       ├── T50 shap analysis.png
│       ├── engine 1 -cycle v t30.png
│       ├── feature selection metrics.png
│       ├── shap_summary.png
│       ├── temporal prediction errors.png
│       └── top 20 features(after slope additon).png
│
├── legacy/
│   ├── models/
│   ├── 01_data_exploration.py
│   ├── 02_model_development.py
│   ├── 03_model_evaluation.py
│   ├── 04_model_tuning.py
│   ├── 05_final_test.py
│   ├── 06_model_interpretation.py
│   ├── 07_shap_analysis.py
│   ├── 08_inference.py
│   ├── 09_local_shap.py
│   ├── 10_engine_report.py
│   ├── 11_early_warning.py
│   ├── 12_temporal_stability.py
│   ├── 13_visual_diagnostics.py
│   ├── 14_test_eval.py
│   ├── 15_production_inference.py
│   ├── 16_dashboard.py
│   ├── 17_final_validation.py
│   └── Readme.md
│
├── data/
│   └── raw/
│       └── train_FD001.txt
│
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

---

# Source Code Organization

The current codebase separates the main responsibilities of the system.

### `src/features.py`

Responsible for feature engineering and transformation of sensor time-series data.

The feature pipeline includes temporal transformations such as:

* Lags
* Rolling statistics
* Degradation slopes

---

### `src/model.py`

Contains model-related functionality.

The Random Forest model is separated from the exploratory scripts so that model development can be reused by other parts of the system.

---

### `src/inference.py`

Responsible for generating predictions from trained model components.

This provides a separation between:

```text
training
```

and

```text
inference
```

which is important when moving toward deployment.

---

### `src/explainability.py`

Contains explainability functionality, including the project's SHAP-based analysis.

---

### `app/app.py`

Provides the application layer built on top of the ML components.

The application is intended to consume the model/inference functionality rather than containing the entire experimental ML workflow itself.

---

# Testing

The repository contains tests covering the primary ML components:

```text
tests/
├── test_features.py
├── test_model.py
├── test_inference.py
└── test_explainability.py
```

Tests are run using `pytest`.

From the project root:

```bash
pytest
```

The repository also contains:

```text
pytest.ini
```

for pytest configuration.

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd predictive-maintenance
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

The current project separates the reusable ML implementation from the historical experimentation scripts.

The archived scripts are located under:

```text
legacy/
```

These scripts document the development process and earlier experiments.

The current implementation should use the modules under:

```text
src/
```

and the application under:

```text
app/
```

---

# Reproducibility

The project uses fixed random seeds where applicable.

For example:

```python
random_state=42
```

is used for:

* Engine-level train/validation splitting
* Random Forest
* Randomized hyperparameter search

This helps make experiments reproducible.

However, exact reproduction can still depend on:

* Python version
* Package versions
* Scikit-learn version
* Dataset version
* Hardware
* Randomized search implementation

The dependency environment is therefore captured in:

```text
requirements.txt
```

---

# Current Limitations

This project should be considered an ML research/engineering prototype rather than a production predictive-maintenance system.

## 1. Dataset limitation

The current work is based on the FD001 subset of the C-MAPSS dataset.

Real industrial equipment can exhibit substantially different:

* Operating conditions
* Failure mechanisms
* Sensor characteristics
* Maintenance interventions
* Environmental conditions

Therefore, performance on C-MAPSS should not be interpreted as direct evidence of performance on a real industrial asset.

---

## 2. RUL is an estimated target

RUL is derived from run-to-failure trajectories.

In an actual industrial deployment, an asset may be:

* Maintained before failure
* Replaced
* Taken offline
* Operated under changing conditions

Therefore, real-world RUL estimation is more complicated than the controlled run-to-failure setting.

---

## 3. High-RUL prediction behaviour

The current model shows a visible compression of predictions at high actual RUL values.

This indicates that further work is required if accurate long-horizon RUL estimation is a primary objective.

---

## 4. Model choice

Random Forest provides a strong and interpretable baseline, but it is not the only possible approach.

The temporal nature of the problem makes other approaches worth investigating, including:

* Gradient boosting
* XGBoost
* LightGBM
* Temporal neural networks
* LSTM/GRU models
* Temporal convolutional models
* Transformer-based time-series models

These should be compared against the Random Forest baseline rather than assumed to be better.

---

## 5. Dataset shift

A model trained on one fleet, equipment type, or operating environment may not generalize directly to another.

A production system would require additional validation under realistic operating conditions.

---

# Future Development

Potential future development areas include:

### Model development

* Compare Random Forest with gradient boosting methods.
* Investigate dedicated time-series models.
* Evaluate different RUL target transformations.
* Investigate high-RUL prediction bias.
* Perform stronger cross-engine validation.
* Evaluate uncertainty in RUL predictions.

### Feature engineering

* Additional degradation indicators.
* Adaptive rolling windows.
* Operating-condition normalization.
* Sensor health indicators.
* Trend-change detection.
* Change-point detection.

### Explainability

* Local SHAP explanations for individual engines.
* Engine-level degradation reports.
* Feature interaction analysis.
* Explainable early-warning thresholds.

### Productionization

* Model versioning.
* Automated inference pipeline.
* API layer.
* Monitoring.
* Data validation.
* Prediction drift detection.
* Model performance monitoring.
* Retraining workflow.

---

# Engineering Philosophy

The project follows a progression from:

```text
Raw sensor data
       ↓
Exploratory analysis
       ↓
RUL construction
       ↓
Temporal feature engineering
       ↓
Feature selection
       ↓
Baseline model
       ↓
Hyperparameter tuning
       ↓
Error analysis
       ↓
Explainability
       ↓
Reusable ML modules
       ↓
Application
```

The goal is not simply to produce a single RUL prediction.

The broader objective is to understand **how degradation information emerges from industrial sensor data and how that information can be transformed into a maintainable machine-learning system.**

---

# Key Findings

The analysis performed so far leads to several important observations:

### 1. Temporal information is highly valuable

Rolling means and degradation slopes repeatedly appear among the most important model features.

### 2. T50 is particularly informative

T50-derived features, especially:

```text
T50_mean_10
T50_mean_5
T50_slope_20
```

appear prominently in model-importance and SHAP analyses.

### 3. Noise reduction helps expose degradation

Raw sensor measurements can be highly variable.

Rolling statistics provide a smoother representation of the underlying trajectory.

### 4. Feature quantity has diminishing returns

Increasing the number of features from 20 to 210 produces relatively modest changes in MAE.

This suggests that feature quality and representation are more important than simply generating more features.

### 5. Model errors are not uniform

The model performs differently across the RUL range.

In particular, high-RUL observations show a visible compression of predicted values.

### 6. Explainability reveals non-linear relationships

SHAP analysis demonstrates that important features do not necessarily have a simple linear relationship with predicted RUL.

---

# Project Status

The project has progressed from exploratory experimentation toward a more structured ML system.

### Completed

* [x] Dataset exploration
* [x] RUL calculation
* [x] Sensor analysis
* [x] Engine-level validation split
* [x] Lag feature engineering
* [x] Rolling statistical features
* [x] Degradation slope features
* [x] Feature selection experiments
* [x] Random Forest baseline
* [x] Hyperparameter tuning
* [x] Error analysis
* [x] Temporal prediction analysis
* [x] Random Forest feature importance
* [x] SHAP analysis
* [x] Feature engineering module
* [x] Model module
* [x] Inference module
* [x] Explainability module
* [x] Automated tests
* [x] Application layer
* [x] Legacy experimentation archive
* [x] Repository restructuring

### In progress / future

* [ ] Stronger model comparison
* [ ] Improved high-RUL performance
* [ ] Prediction uncertainty
* [ ] More robust temporal validation
* [ ] Production-oriented monitoring
* [ ] Real-world industrial validation

---

# Repository History

The `legacy/` directory contains the earlier development pipeline.

These scripts are intentionally preserved because they document the progression of the project, including:

* Exploratory analysis
* Initial model development
* Evaluation
* Hyperparameter tuning
* SHAP analysis
* Inference experiments
* Early-warning experiments
* Temporal stability analysis
* Visualization
* Final validation experiments

The current `src/` implementation separates reusable functionality from those exploratory scripts.

This makes the repository easier to maintain without discarding the experimental history.

---

# Technologies

The project is primarily implemented in Python.

Core technologies include:

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Matplotlib
* SHAP
* Pytest

---

# Disclaimer

This project is an experimental predictive-maintenance system developed using a public run-to-failure dataset.

Predictions produced by the model should not be treated as direct maintenance instructions or as a substitute for qualified engineering inspection.

A production predictive-maintenance system would require validation against real equipment, domain-specific failure mechanisms, sensor reliability, operational constraints, uncertainty estimation, and appropriate engineering and safety procedures.

---

That will make the README much more defensible when someone reviewing your GitHub asks, *"Okay, but what does the final model actually achieve?"*
```
