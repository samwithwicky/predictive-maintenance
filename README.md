# Predictive Maintenance — Remaining Useful Life Prediction

An end-to-end, production-oriented machine learning pipeline for **Predictive Maintenance and Remaining Useful Life (RUL) estimation** of aircraft engines using the **NASA C-MAPSS FD001 dataset**.

The project covers the complete workflow from **raw sensor data and feature engineering to model training, evaluation, explainability, testing, saved models, and inference**.

---

## 1. Project Overview

Predictive maintenance aims to estimate the remaining operating life of equipment so that maintenance can be planned before failure occurs.

This project predicts the **Remaining Useful Life (RUL)** of aircraft engines from their operating conditions and sensor measurements.

### Objectives

* Predict engine Remaining Useful Life (RUL)
* Extract useful information from sensor data
* Capture degradation using time-based features
* Train and evaluate a regression model
* Analyze prediction errors and model behaviour
* Explain predictions using SHAP
* Build a reusable and testable ML pipeline

---

## 2. Dataset

### NASA C-MAPSS FD001

The project uses the **NASA C-MAPSS FD001 dataset**, containing simulated aircraft engine degradation data recorded over multiple operating cycles.

Each observation contains:

* Engine / unit ID
* Operating cycle
* 3 operating settings
* 21 sensor measurements

The RUL target is derived from the engine's operating cycle and its final recorded cycle.

---

## 3. Pipeline

```text
Raw Sensor Data
      ↓
Data Preparation
      ↓
RUL Target Construction
      ↓
Feature Engineering
      ↓
Time-based Features
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Interpretation
      ↓
Validation & Testing
      ↓
Saved Model
      ↓
Inference
```

The pipeline separates **data processing, feature engineering, model logic, inference, and explainability** into reusable components.

---

## 4. Feature Engineering

The raw sensor measurements are transformed into features that capture both the current engine condition and its degradation over time.

Features include:

* Rolling means
* Time-window statistics
* Sensor trends / slopes
* Multiple temporal windows

This allows the model to learn degradation patterns rather than relying only on individual sensor readings.

---

## 5. Model

### Random Forest Regressor

The primary model is a **Random Forest Regressor**.

Model performance is evaluated using:

* **MAE** — Mean Absolute Error
* **RMSE** — Root Mean Squared Error
* Actual vs. predicted RUL
* Prediction error distribution
* Temporal prediction behaviour

FINAL Model Results:

| Metric |            Result             |
| ------ | ----------------------------- |
| MAE    | 11.45212339356427 RUL cycles  |
| RMSE   | 15.319480429904301 RUL cycles |



---

## 6. Explainability

The pipeline includes model interpretation using **feature importance** and **SHAP**.

### Feature Importance

Identifies which engineered features contribute most to the Random Forest model.

### SHAP

SHAP is used to investigate:

* Which features influence predictions
* The direction of their influence
* How feature values affect model output
* Individual prediction behaviour

The analysis identified strong contributions from features derived from sensors such as **T50, T24, Nc, Ps30, W31, and W32**.

---

## 7. Project Structure

```text
predictive-maintenance/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       ├── RUL_FD001.txt
│       ├── test_FD001.txt
│       └── train_FD001.txt
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
│   └── 17_final_validation.py
│
├── models/
│   ├── rul_features.pkl
│   └── rul_random_forest_capped.pkl
│
├── outputs/
│   └── diagnostics/
│
├── reports/
│   └── figures/
│
├── src/
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
├── .gitignore
├── pytest.ini
├── README.md
└── requirements.txt
```

---

## 8. Installation

Clone the repository:

```bash
git clone https://github.com/samwithwicky/predictive-maintenance.git
cd predictive-maintenance
```

Create a virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 9. Running

The reusable implementation is located in:

```text
src/
```

Trained model artifacts and feature configuration are stored in:

```text
models/
```

Run the test suite:

```bash
pytest
```

Generated diagnostics and analysis figures are available in:

```text
outputs/diagnostics/
reports/figures/
```

The `legacy/` directory contains the earlier experimental and development scripts used while building and validating the current pipeline.

---

## 10. Engineering & Production Workflow

The project was developed with an **end-to-end production mindset**, rather than as a standalone model-training notebook.

It includes:

* Modular source code
* Reusable feature engineering
* Saved model artifacts
* Dedicated inference code
* Model explainability
* Automated tests
* Validation scripts
* Diagnostic outputs
* Structured project organization
* Separation of experimental and reusable code

The structure is intended to make the system easier to **test, reproduce, extend, and eventually deploy**.

---

## 11. Learning Through the Project

This project was used as a **hands-on learning vehicle for building production-oriented ML systems**.

Instead of learning machine learning concepts separately and applying them later, the concepts were learned while building the complete system.

The project provided practical experience with:

* Working with time-series sensor data
* Designing ML features
* Building reusable ML components
* Training and evaluating models
* Understanding model errors
* Model explainability with SHAP
* Writing automated tests
* Model serialization and inference
* Organizing an ML codebase
* Separating experimentation from reusable code
* Validation and reliability
* Thinking about deployment

The main outcome was not only a trained model, but an understanding of how an ML model fits into a **complete engineering pipeline**.

---

## 12. Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* SHAP
* Matplotlib
* Pytest
* Joblib

---

## 13. Status

**Status: Functional ML prototype / production-oriented pipeline**

The current implementation focuses on RUL prediction using the NASA C-MAPSS FD001 dataset.

The architecture provides a foundation for future extensions such as **real-time sensor ingestion, early-warning systems, monitoring, and deployment in industrial predictive maintenance environments**.
