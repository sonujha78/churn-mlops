import time
import json
import os
import pandas as pd
from prometheus_client import start_http_server, Gauge, Counter

from drift import calculate_feature_drift

# ---- Config ----
TRAINING_DATA_PATH = "data/customer_churn.csv"
LOG_FILE = "prediction_logs.jsonl"
WINDOW_SIZE = int(os.getenv("DRIFT_WINDOW_SIZE", "50"))   # recompute every N new requests
CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", "10"))
NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "total_charges"]

# ---- Prometheus metrics ----
DRIFT_SCORE = Gauge("model_drift_score", "Overall PSI drift score (avg across features)")
FEATURE_DRIFT = Gauge("model_feature_drift_score", "PSI drift score per feature", ["feature"])
PREDICTION_COUNT = Counter("model_prediction_total", "Total number of predictions served")
CHURN_PREDICTIONS = Counter("model_churn_prediction_total", "Total predictions classified as churn")
CHURN_RATE = Gauge("model_churn_rate", "Rolling churn prediction rate (last window)")

_last_seen_count = 0


def load_training_data():
    df = pd.read_csv(TRAINING_DATA_PATH)
    return df


def read_all_logs():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def run_monitor():
    global _last_seen_count

    training_df = load_training_data()
    print(f"Loaded training data: {training_df.shape}")

    start_http_server(9100)
    print("Prometheus metrics server started on :9100/metrics")

    while True:
        records = read_all_logs()
        total = len(records)

        # Update total prediction counter (only increment by new records)
        new_records = records[_last_seen_count:]
        for r in new_records:
            PREDICTION_COUNT.inc()
            if r.get("prediction") == "churn":
                CHURN_PREDICTIONS.inc()
        _last_seen_count = total

        if total >= 10:
            window = records[-WINDOW_SIZE:] if total > WINDOW_SIZE else records
            live_df = pd.DataFrame([r["input"] for r in window])

            feature_scores, overall_psi = calculate_feature_drift(
                training_df, live_df, NUMERIC_FEATURES
            )

            DRIFT_SCORE.set(overall_psi)
            for feat, score in feature_scores.items():
                FEATURE_DRIFT.labels(feature=feat).set(score)

            churn_count = sum(1 for r in window if r.get("prediction") == "churn")
            churn_rate = churn_count / len(window) if window else 0.0
            CHURN_RATE.set(churn_rate)

            print(f"[monitor] total_requests={total} window={len(window)} "
                  f"overall_psi={overall_psi:.4f} churn_rate={churn_rate:.2f}")
        else:
            print(f"[monitor] waiting for more data... ({total}/10 requests)")

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_monitor()
