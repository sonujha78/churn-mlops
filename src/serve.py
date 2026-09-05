import os
import json
import mlflow
import mlflow.pyfunc
import pandas as pd
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = os.getenv("MODEL_NAME", "churn-model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
LOG_FILE = "prediction_logs.jsonl"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = None
model_version_info = {}


def load_production_model():
    global model, model_version_info
    client = mlflow.tracking.MlflowClient()
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.pyfunc.load_model(model_uri)

    version_details = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model_version_info = {
        "name": MODEL_NAME,
        "version": version_details.version,
        "alias": MODEL_ALIAS,
        "run_id": version_details.run_id,
    }
    print(f"Loaded model: {model_version_info}")


@app.on_event("startup")
def startup_event():
    load_production_model()


class CustomerFeatures(BaseModel):
    tenure_months: int
    monthly_charges: float
    total_charges: float
    contract_type: int   # 0=month-to-month, 1=one-year, 2=two-year (label-encoded)
    internet_service: int
    tech_support: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return model_version_info


@app.post("/predict")
def predict(features: CustomerFeatures):
    input_df = pd.DataFrame([features.dict()])
    proba = model.predict(input_df)

    # pyfunc models may return array of probabilities or classes depending on flavor
    prob_value = float(proba[0]) if hasattr(proba, "__getitem__") else float(proba)
    prediction = "churn" if prob_value >= 0.5 else "no_churn"

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": features.dict(),
        "churn_probability": prob_value,
        "prediction": prediction,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {
        "churn_probability": round(prob_value, 4),
        "prediction": prediction,
    }
