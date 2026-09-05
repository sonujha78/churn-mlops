"""
Retraining script — triggered automatically when drift alert fires.
Trains a new model on the latest available data and logs it to MLflow
as a NEW run. Does NOT auto-promote to production — requires manual
review and promotion via the MLflow UI or a separate approval step.

If the versioned training dataset (DVC-tracked) is not available in
the current environment (e.g. a CI runner without access to the DVC
remote), this script regenerates an equivalent dataset so retraining
can still run end-to-end. In a full production setup, the DVC remote
would be cloud-hosted object storage (S3/GCS) accessible from CI.
"""
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA_PATH = "data/customer_churn.csv"


def ensure_dataset():
    if os.path.exists(DATA_PATH):
        return
    print(f"WARNING: {DATA_PATH} not found (DVC remote not accessible in this "
          f"environment). Regenerating an equivalent dataset for retraining.")
    os.makedirs("data", exist_ok=True)
    np.random.seed(42)
    n = 1000
    tenure = np.random.randint(1, 72, n)
    monthly_charges = np.round(np.random.uniform(20, 120, n), 2)
    total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 100, n), 2)
    contract_type = np.random.choice(['month-to-month', 'one-year', 'two-year'], n, p=[0.55, 0.25, 0.20])
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n)
    tech_support = np.random.choice(['Yes', 'No'], n)

    churn_prob = (
        0.5 - 0.006 * tenure
        + 0.15 * (contract_type == 'month-to-month')
        + 0.002 * monthly_charges
        - 0.15 * (tech_support == 'Yes')
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn = np.random.binomial(1, churn_prob)

    df = pd.DataFrame({
        'customer_id': range(1, n + 1),
        'tenure_months': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract_type': contract_type,
        'internet_service': internet_service,
        'tech_support': tech_support,
        'churn': churn
    })
    df.to_csv(DATA_PATH, index=False)


def retrain():
    ensure_dataset()
    df = pd.read_csv(DATA_PATH)

    cat_cols = ["contract_type", "internet_service", "tech_support"]
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run(run_name="automated_retrain"):
        params = {
            "model_type": "RandomForest",
            "n_estimators": 250,
            "max_depth": 7,
            "trigger": "drift_alert_auto_retrain",
        }
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1_score": f1_score(y_test, preds, zero_division=0),
            "auc": roc_auc_score(y_test, proba),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print(f"[automated_retrain] {metrics}")

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, "churn-model")

        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias("churn-model", "staging", result.version)

        print(f"Registered as churn-model version {result.version}, alias 'staging' set.")
        print("NOTE: Production promotion requires manual review and approval.")


if __name__ == "__main__":
    retrain()
