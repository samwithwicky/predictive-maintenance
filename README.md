# Predictive Maintenance — Remaining Useful Life Prediction

An end-to-end, production-oriented machine learning pipeline for **Predictive Maintenance and Remaining Useful Life (RUL) estimation** of aircraft engines using the **NASA C-MAPSS FD001 dataset**.

The project covers the complete ML lifecycle from **raw sensor data and feature engineering to model training, evaluation, explainability, testing, inference, application development, and containerized deployment**.

---

## 1. Project Overview

Predictive maintenance aims to estimate the remaining operating life of equipment so that maintenance can be planned before failure occurs.

This project predicts the **Remaining Useful Life (RUL)** of aircraft engines from their operating conditions and sensor measurements.

### Objectives

- Predict engine Remaining Useful Life (RUL)
- Extract useful information from sensor data
- Capture degradation using time-based features
- Train and evaluate a regression model
- Analyze prediction errors and model behaviour
- Explain predictions using SHAP
- Build a reusable and testable ML pipeline
- Serve the trained model through an application
- Containerize and deploy the application using Docker

---

## 2. Dataset

### NASA C-MAPSS FD001

The project uses the **NASA C-MAPSS FD001 dataset**, containing simulated aircraft engine degradation data recorded over multiple operating cycles.

Each observation contains:

- Engine / unit ID
- Operating cycle
- 3 operating settings
- 21 sensor measurements

The RUL target is derived from the engine's operating cycle and its final recorded cycle.

---

## 3. End-to-End Pipeline

The project was developed as a complete ML engineering workflow rather than only a model-training experiment.

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
Model Serialization
      ↓
Inference Pipeline
      ↓
Streamlit Application
      ↓
Docker Containerization
      ↓
Deployed Application
```

The pipeline separates **data processing, feature engineering, model logic, inference, explainability, application logic, and deployment** into reusable components.

---

## 4. Feature Engineering

The raw sensor measurements are transformed into features that capture both the current engine condition and its degradation over time.

Features include:

- Rolling means
- Time-window statistics
- Sensor trends / slopes
- Multiple temporal windows

This allows the model to learn degradation patterns rather than relying only on individual sensor readings.

---

## 5. Model

### Random Forest Regressor

The primary model is a **Random Forest Regressor**.

Model performance is evaluated using:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- Actual vs. predicted RUL
- Prediction error distribution
- Temporal prediction behaviour

### Final Model Results

| Metric |             Result |
| ------ | -----------------: |
| MAE    | 11.4521 RUL cycles |
| RMSE   | 15.3195 RUL cycles |

---

## 6. Explainability

The pipeline includes model interpretation using **feature importance** and **SHAP**.

### Feature Importance

Identifies which engineered features contribute most to the Random Forest model.

### SHAP

SHAP is used to investigate:

- Which features influence predictions
- The direction of their influence
- How feature values affect model output
- Individual prediction behaviour

The analysis identified strong contributions from features derived from sensors such as **T50, T24, Nc, Ps30, W31, and W32**.

---

## 7. Application & Deployment

The trained model is exposed through a **Streamlit application** that provides a user-facing interface for RUL prediction and model explanations.

The application uses the reusable components in `src/` rather than directly depending on the experimental scripts in `legacy/`.

### Run locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/app.py
```

The application will be available at:

```text
http://localhost:8501
```

### Docker Deployment

The application is also containerized using **Docker**, allowing it to run independently of the local Python environment.

Build the Docker image:

```bash
docker build -t predictive-maintenance .
```

Run the container:

```bash
docker run -p 8501:8501 predictive-maintenance
```

Then open:

```text
http://localhost:8501
```

The Docker image contains the application, reusable ML source code, trained model artifacts, required data, and Python dependencies.

Development-only files and historical scripts are excluded from the deployment image using `.dockerignore`.

---

## 8. Project Structure

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
├── .dockerignore
├── .gitignore
├── Dockerfile
├── pytest.ini
├── README.md
└── requirements.txt
```

---

## 9. Installation

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

## 10. Testing

The project includes automated tests for the reusable ML components.

Run:

```bash
pytest
```

Tests cover:

- Feature engineering
- Model behaviour
- Inference
- Explainability

---

## 11. Engineering & Production Workflow

The project was developed with an **end-to-end production mindset**, rather than as a standalone model-training notebook.

It includes:

- Modular source code
- Reusable feature engineering
- Saved model artifacts
- Dedicated inference code
- Model explainability
- Automated tests
- Validation scripts
- Diagnostic outputs
- Structured project organization
- Separation of experimental and reusable code
- Streamlit application
- Docker containerization
- Reproducible application deployment

The result is a complete workflow from **data to a runnable, containerized ML application**.

---

## 12. Learning Through the Project

This project was used as a **hands-on learning vehicle for building end-to-end ML systems**.

Instead of learning machine learning concepts separately and applying them later, the concepts were learned while building the complete system.

The project provided practical experience with:

- Working with time-series sensor data
- Designing ML features
- Building reusable ML components
- Training and evaluating models
- Understanding model errors
- Model explainability with SHAP
- Writing automated tests
- Model serialization and inference
- Organizing an ML codebase
- Separating experimentation from reusable code
- Validation and reliability
- Building a user-facing ML application
- Containerizing an application with Docker
- Understanding the transition from an ML model to a deployable system

The main outcome was not only a trained model, but practical experience with how an ML system is **developed, tested, packaged, served, and deployed end to end**.

---

## 13. Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Matplotlib
- Pytest
- Joblib
- Streamlit
- Docker

---

## 14. Status

**Status: Complete end-to-end ML prototype with containerized deployment**

The current implementation focuses on RUL prediction using the NASA C-MAPSS FD001 dataset.

The completed pipeline covers:

```text
Data
 ↓
Feature Engineering
 ↓
Model Development
 ↓
Evaluation
 ↓
Explainability
 ↓
Testing
 ↓
Inference
 ↓
Application
 ↓
Docker
 ↓
Deployment
```

The architecture provides a foundation for future extensions such as **real-time sensor ingestion, early-warning systems, monitoring, and additional industrial predictive maintenance applications**.
