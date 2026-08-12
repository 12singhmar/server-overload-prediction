"""FastAPI service for exploring the server-overload demonstration model.

The model and data in this repository are educational and synthetic.  See the
README before using the output for any operational decision.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "server_overload_xgb.pkl"
FEATURES_PATH = BASE_DIR / "feature_names.pkl"

app = FastAPI(title="Server Overload Explorer", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

model = joblib.load(MODEL_PATH)
feature_names = list(joblib.load(FEATURES_PATH))


def score(row: dict[str, Any]) -> dict[str, Any]:
    """Return the demonstration model score after checking its feature set."""
    missing = [name for name in feature_names if name not in row]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"message": "Missing model features", "features": missing},
        )

    frame = pd.DataFrame([row])[feature_names]
    probability = float(model.predict_proba(frame)[0][1])
    return {
        "overload_probability": round(probability, 4),
        "early_warning": int(probability >= 0.3),
        "threshold": 0.3,
        "notice": "Educational output from a synthetic dataset; not for production monitoring.",
    }


@app.get("/")
def landing(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/dashboard")
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok", "model_features": len(feature_names)}


@app.post("/predict")
def predict(data: dict[str, Any]):
    return score(data)


@app.post("/simulate")
def simulate(data: dict[str, Any]):
    """Generate a plausible synthetic snapshot for the dashboard demo."""
    cpu = float(data.get("CPU_Utilization_%", 50))
    temp = float(data.get("CPU1_Temp_C", 45))
    power = float(data.get("System_Power_W", 250))
    fan = float(data.get("Avg_Fan_Speed_RPM", 3500))
    hour = int(data.get("hour", 12))

    row = {
        "CPU_Utilization_%": cpu,
        "Memory_Utilization_%": 30 + cpu * 0.35 + np.random.normal(0, 2),
        "CPU1_Temp_C": temp,
        "CPU2_Temp_C": temp + np.random.normal(0.5, 0.3),
        "Inlet_Temp_C": 18 + np.random.normal(0, 1),
        "Exhaust_Temp_C": temp + 10 + np.random.normal(0, 1),
        "System_Power_W": power,
        "Avg_Fan_Speed_RPM": fan,
        "Air_Flow_CFM": 15 + fan / 300 + np.random.normal(0, 1),
        "Voltage_V": 12 + np.random.normal(0, 0.05),
        "Clock_Speed_GHz": 2.5 + (cpu / 100) * 0.5 + np.random.normal(0, 0.05),
        "hour": hour,
        "day_of_week": 2,
    }

    for base, value in [
        ("CPU_Utilization_%", cpu),
        ("CPU1_Temp_C", temp),
        ("CPU2_Temp_C", row["CPU2_Temp_C"]),
        ("System_Power_W", power),
        ("Avg_Fan_Speed_RPM", fan),
    ]:
        noise = np.random.normal(0, 1, size=3)
        lags = [value * 0.95 + noise[0], value * 0.90 + noise[1], value * 0.85 + noise[2]]
        for index, lag in enumerate(lags, start=1):
            row[f"{base}_lag{index}"] = round(float(lag), 4)
        rolling_values = [value, lags[0], lags[1]]
        row[f"{base}_roll_mean_3"] = round(float(np.mean(rolling_values)), 4)
        row[f"{base}_roll_std_3"] = round(float(np.std(rolling_values, ddof=1)), 4)

    return {"features": row, **score(row)}
