"""
Retraining script — triggered automatically when drift alert fires.
Trains a new model on the latest available data and logs it to MLflow
as a NEW run. Does NOT auto-promote to production — requires manual
review and promotion via the MLflow UI or a separate approval step.
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA_PATH = "data/customer_churn.csv"


def retrain():
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
