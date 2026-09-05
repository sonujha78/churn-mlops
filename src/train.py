import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ---- Load data ----
df = pd.read_csv("data/customer_churn.csv")

# ---- Preprocess ----
cat_cols = ["contract_type", "internet_service", "tech_support"]
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_experiment("churn-prediction")

def log_experiment(model_name, model, params, is_xgboost=False):
    with mlflow.start_run(run_name=model_name):
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

        if is_xgboost:
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        print(f"[{model_name}] {metrics}")
        return metrics


if __name__ == "__main__":
    # Experiment 1: Logistic Regression
    params1 = {"model_type": "LogisticRegression", "max_iter": 500, "C": 1.0}
    log_experiment(
        "logistic_regression",
        LogisticRegression(max_iter=params1["max_iter"], C=params1["C"]),
        params1,
    )

    # Experiment 2: Random Forest
    params2 = {"model_type": "RandomForest", "n_estimators": 200, "max_depth": 6}
    log_experiment(
        "random_forest",
        RandomForestClassifier(
            n_estimators=params2["n_estimators"],
            max_depth=params2["max_depth"],
            random_state=42,
        ),
        params2,
    )

    # Experiment 3: XGBoost
    params3 = {
        "model_type": "XGBoost",
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
    }
    log_experiment(
        "xgboost",
        XGBClassifier(
            n_estimators=params3["n_estimators"],
            max_depth=params3["max_depth"],
            learning_rate=params3["learning_rate"],
            eval_metric="logloss",
            random_state=42,
        ),
        params3,
        is_xgboost=True,
    )

    print("All experiments logged to MLflow.")
