# Server Overload Explorer

A small FastAPI application for exploring a machine-learning workflow around server telemetry. It generates a synthetic server snapshot and returns an early-warning score from a saved XGBoost classifier.

> **Project status:** educational prototype. The data preparation notebook creates part of the dataset and target synthetically. The app is useful for demonstrating an end-to-end ML workflow, not for operating real infrastructure.

## What this project demonstrates

- feature engineering from server telemetry
- a classification model for an overload-warning signal
- a separate temperature-regression experiment
- serving a saved model through FastAPI
- a simple browser UI that calls the API

## Run locally

Requirements: Python 3.10+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000` for the project page, `http://127.0.0.1:8000/dashboard` for the demo, and `http://127.0.0.1:8000/docs` for the generated API documentation.

## Project layout

```text
main.py                    FastAPI application and inference endpoints
templates/                 Landing page and interactive dashboard
static/                    Dashboard styles and browser-side logic
data_cleaning.ipynb        Data preparation and feature engineering
model_train.ipynb          Classifier and regression experiments
model2_regression.ipynb    Follow-up regression analysis
*_xgb.pkl, feature_names.pkl  Saved experiment artifacts
```

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Confirms the service and saved feature list are available. |
| `POST /simulate` | Builds a synthetic telemetry row from a few dashboard inputs and scores it. |
| `POST /predict` | Scores a complete feature payload in the saved model's expected format. |

The dashboard uses `/simulate`; it is the recommended way to explore the application.

## Methodology and limitations

This repository previously reported near-perfect classification and regression metrics. Those numbers should **not** be interpreted as real-world performance.

The preparation notebook constructs `Overload` from a weighted stress formula that includes CPU utilisation, CPU temperature, power, cooling efficiency, and fan health. Several of those ingredients were also supplied as model features. This makes a strong score easy to obtain because the target is partly defined by the predictors. Temperature is also regenerated with a smooth autoregressive process, which makes the regression experiment easier than a noisy production setting.

For a credible next experiment, I would:

1. Keep a time-ordered holdout rather than relying only on randomly selected nodes.
2. Train a baseline that excludes the direct ingredients of the synthetic stress formula.
3. Compare against a simple rule-based baseline and report precision, recall, PR-AUC, and ROC-AUC.
4. Report the new results as synthetic-data experiments, including the overload prevalence and the chosen operating threshold.

The saved model remains in this repository so the application can be demonstrated, but its historical metrics are intentionally not presented as a headline result.

## Data and responsible use

The notebook references the public `kevinkonstas/server-load-dataset` dataset and then adds synthetic variables and transformations. Do not use the current model for real alerting, capacity planning, or safety decisions without a separate data-validation and evaluation process.

## Future work

- rebuild the training pipeline from a versioned dataset
- add a leakage-resistant, time-based evaluation notebook
- add automated API tests and deployment configuration
- replace locally committed model binaries with versioned release artifacts or an object store
