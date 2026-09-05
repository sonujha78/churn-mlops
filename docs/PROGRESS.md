# MLOps Churn Pipeline — Progress Log
Project: Customer Churn Prediction with Drift Detection
Repo: https://github.com/sonujha78/churn-mlops
---
## Step 1: Environment & Project Setup
- Created project directory `churn-mlops`
- Initialized Git repo, connected to GitHub remote
- Created Python virtual environment (`venv`)
- Created folder structure: data/, models/, src/, notebooks/, .github/workflows/, monitoring/, docs/
- Added `.gitignore` for venv, data files, model files, mlruns
Commands used:
```bash
mkdir -p ~/churn-mlops && cd ~/churn-mlops
git init
git branch -M main
python3 -m venv venv
source venv/bin/activate
mkdir -p data models src notebooks .github/workflows monitoring docs
git remote add origin https://github.com/sonujha78/churn-mlops.git
git push -u origin main
```
Status: ✅ Done
---
## Step 2: DVC Setup & Dataset Versioning
- Installed DVC (`pip install dvc`)
- Initialized DVC in the project (`dvc init`)
- Created sample customer churn dataset (`data/customer_churn.csv`)
- Tracked dataset with DVC (`dvc add`)
- Configured local DVC remote storage (`~/dvc-storage`)
- Pushed data to DVC remote (`dvc push`)
Commands used:
```bash
pip install dvc
dvc init
git add .dvc .dvcignore
git commit -m "chore: initialize DVC"
dvc add data/customer_churn.csv
git add data/customer_churn.csv.dvc
git commit -m "feat: add initial dataset (v1) tracked with DVC"
dvc remote add -d myremote ~/dvc-storage
git add .dvc/config
git commit -m "chore: configure DVC local remote storage"
dvc push
git push
```
Status: ✅ Done
---
## Step 3: Training Pipeline & Experiment Tracking (MLflow)
- Installed MLflow (`pip install mlflow xgboost`)
- Created `src/train.py` — trains 3 models: Logistic Regression, Random Forest, XGBoost
- Logged params, metrics (accuracy, precision, recall, f1, auc), and model artifacts to MLflow
- Fixed dataset to have real signal (churn probability based on tenure, contract type, charges, tech support)
- Fixed XGBoost logging issue by using `mlflow.xgboost.log_model` instead of `mlflow.sklearn.log_model`
- Compared 3 runs in MLflow UI (Parallel Coordinates Plot)
- Registered best model (random_forest, f1_score=0.57) as "churn-model" version 1
- Promoted version 1 through aliases: `staging` -> `production`
Commands used:
```bash
pip install mlflow xgboost
python3 src/train.py
mlflow ui --host 0.0.0.0 --port 5000
# Register best model
python3 -c "
import mlflow
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name('churn-prediction')
runs = client.search_runs(experiment.experiment_id, order_by=['metrics.f1_score DESC'])
best_run = runs[0]
model_uri = f'runs:/{best_run.info.run_id}/model'
mlflow.register_model(model_uri, 'churn-model')
"
# Promote via aliases
python3 -c "
import mlflow
client = mlflow.tracking.MlflowClient()
client.set_registered_model_alias('churn-model', 'staging', 1)
client.set_registered_model_alias('churn-model', 'production', 1)
"
```
Status: ✅ Done
---
## Step 4: Model Serving (FastAPI + Docker)
- Installed FastAPI, uvicorn, pydantic
- Created `src/serve.py` — loads production model from MLflow registry using alias-based URI (`models:/churn-model@production`)
- Implemented `/health`, `/model-info`, and `/predict` endpoints
- Every prediction logged to `prediction_logs.jsonl` (timestamp, input features, probability, prediction) — this feeds drift detection
- Tested locally with uvicorn — all endpoints working
- Created `requirements-serve.txt` (minimal deps for serving, separate from full dev environment)
- Built Dockerfile (python:3.12-slim base), staged pip installs to handle slow network / timeouts
- Fixed artifact path mismatch issue by mounting mlflow.db and mlruns at the exact same absolute host path inside the container, and overriding MLFLOW_TRACKING_URI accordingly
- Also mounted prediction_logs.jsonl to host so live logs are visible outside the container
- Verified containerized API: /health, /model-info, /predict all working correctly
Commands used:
```bash
pip install fastapi uvicorn pydantic
docker build -t churn-serving-api:latest .
docker run -d \
  --name churn-api \
  -p 8000:8000 \
  -v ~/churn-mlops/mlflow.db:/home/sonu/churn-mlops/mlflow.db \
  -v ~/churn-mlops/mlruns:/home/sonu/churn-mlops/mlruns \
  -v ~/churn-mlops/prediction_logs.jsonl:/app/prediction_logs.jsonl \
  -e MLFLOW_TRACKING_URI="sqlite:////home/sonu/churn-mlops/mlflow.db" \
  churn-serving-api:latest
curl http://localhost:8000/health
curl http://localhost:8000/model-info
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{...}'
```
Status: ✅ Done
---
## Step 5: Drift Detection
- Created `src/drift.py` — computes Population Stability Index (PSI) per numeric feature (tenure_months, monthly_charges, total_charges) comparing training distribution vs live prediction traffic
- Made PSI calculation robust: clips out-of-range live values into training range before binning (avoids np.histogram silently dropping outliers and inflating PSI), and adapts bucket count to live sample size (avoids noisy PSI with small windows)
- Created `src/monitor.py` — background loop that reads `prediction_logs.jsonl`, computes drift on a rolling window (last 50 requests), and exposes metrics via `prometheus_client` on port 9100
- Exposed metrics: `model_drift_score` (overall PSI), `model_feature_drift_score` (per-feature PSI), `model_prediction_total`, `model_churn_prediction_total`, `model_churn_rate`
- Installed and configured Prometheus (v2.54.1) to scrape the monitor's `/metrics` endpoint every 5s
- Configured a Prometheus alert rule (`HighDataDrift`) that fires when `model_drift_score > 0.25` for 30s
- Installed Grafana (v11.2.0), connected Prometheus as a data source
- Built a Grafana dashboard ("Churn Model Monitoring") with 3 panels: Data Drift Score (PSI) with a 0.25 threshold line, Prediction Volume (per sec), Churn Prediction Rate
- Verified with normal (training-like) traffic that drift score stays low (~0.11-0.24), confirming baseline works correctly before simulating real drift
Commands used:
```bash
pip install prometheus-client
# Run monitor (exposes :9100/metrics)
python3 src/monitor.py
# Prometheus setup
mkdir -p monitoring/prometheus && cd monitoring/prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.54.1/prometheus-2.54.1.linux-amd64.tar.gz
tar xvfz prometheus-2.54.1.linux-amd64.tar.gz
# prometheus.yml scrapes localhost:9100, alert_rules.yml defines HighDataDrift (> 0.25)
./prometheus --config.file=prometheus.yml --web.listen-address=:9090
# Grafana setup
cd ~/churn-mlops/monitoring
wget https://dl.grafana.com/oss/release/grafana-11.2.0.linux-amd64.tar.gz
tar -zxvf grafana-11.2.0.linux-amd64.tar.gz
cd grafana-v11.2.0
./bin/grafana-server --homepath .
# Added Prometheus data source (http://localhost:9090) in Grafana UI
# Built dashboard with drift score / prediction volume / churn rate panels
```
Screenshots: `docs/screenshots/04-grafana-dashboard-normal-traffic.png`
Status: ✅ Done
---
## Step 6: Automated Retraining Trigger
- Created `src/retrain.py` — retrains a RandomForest model on the latest data and logs a NEW MLflow run (does NOT auto-promote to production)
- New model is registered and given the `staging` alias only — promotion to `production` requires manual review, per task requirement
- Made retraining script resilient: if the DVC-tracked dataset isn't available in the environment (e.g. a CI runner without access to the local DVC remote), it regenerates an equivalent dataset so the pipeline still runs end-to-end. In a full production setup, the DVC remote would be cloud-hosted object storage (S3/GCS) accessible from CI.
- Created `.github/workflows/retrain.yml` — GitHub Actions workflow triggered by a `repository_dispatch` event (`drift_alert`), also supports manual `workflow_dispatch` for testing
- Created `src/trigger_retrain.py` — simulates what a Prometheus Alertmanager webhook receiver would do when the `HighDataDrift` alert fires: calls the GitHub API to dispatch the `drift_alert` event
- Verified end-to-end: triggered the workflow manually, it ran on GitHub Actions, retrained the model, and registered a new version with the `staging` alias (confirmed via Actions run #2 — success)
Commands used:
```bash
pip install requests
# Trigger retraining workflow (simulates alert webhook firing)
export GITHUB_TOKEN="<personal_access_token>"
python3 src/trigger_retrain.py "drift_detected"
```
Status: ✅ Done
---
## Step 7: Simulating Drift
- Sent synthetic traffic with values far outside the training distribution (tenure_months ~150-170 vs training's 1-72, monthly_charges ~300-350 vs training's 20-120) to simulate a real-world customer base shift
- Confirmed `model_drift_score` rose from a normal baseline (~0.24) to 8.19 — a ~33x increase over the 0.25 alert threshold
- Confirmed the Grafana "Data Drift Score (PSI)" panel shows a sharp, clearly visible vertical spike at the moment shifted traffic was sent
- Confirmed the Prometheus `HighDataDrift` alert transitioned from Inactive to Firing (verified via both the Prometheus UI and the `/api/v1/alerts` API, which returned `"state": "firing"` with a description noting the threshold breach)
- This demonstrates the core scenario the task is built around: a model that keeps returning healthy HTTP 200 responses while silently degrading is detected without needing ground-truth labels
Commands used:
```bash
# Simulate drifted traffic (shifted well outside training ranges)
for i in {1..60}; do
  tenure=$((RANDOM % 20 + 150))
  monthly=$((RANDOM % 50 + 300))
  total=$((tenure * monthly))
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d "{\"tenure_months\": $tenure, \"monthly_charges\": $monthly, \"total_charges\": $total, \"contract_type\": $((RANDOM % 3)), \"internet_service\": $((RANDOM % 3)), \"tech_support\": $((RANDOM % 2))}" > /dev/null
  sleep 0.15
done

# Verify drift score and alert state
curl -s http://localhost:9100/metrics | grep model_drift_score
curl -s http://localhost:9090/api/v1/alerts | python3 -m json.tool
```
Screenshots:
- `docs/screenshots/05-grafana-drift-spike.png` — Grafana dashboard showing the drift score spike
- `docs/screenshots/06-prometheus-alert-firing.png` — Prometheus UI showing HighDataDrift alert firing
- `docs/screenshots/07-prometheus-alert-api-response.png` — Prometheus API confirming firing state with PSI value
Status: ✅ Done
---
## Step 8: Documentation & Screenshots
- MLflow experiment comparison screenshots captured (3 runs: logistic_regression, random_forest, xgboost)
- Model registry screenshots captured (version 1, staging -> production alias promotion)
- Grafana dashboard screenshots captured (normal traffic baseline + simulated drift spike)
- Prometheus alert screenshots captured (Inactive state and Firing state)
- All screenshots organized under `docs/screenshots/`
- This PROGRESS.md file itself serves as the running documentation log for every step, command, and fix applied throughout the project
Status: ✅ Done

---

## Project Complete

All 8 steps of the MLOps Customer Churn Prediction pipeline with drift detection are done:
data versioning (DVC) -> experiment tracking & model registry (MLflow) -> model serving (FastAPI + Docker)
-> drift detection (PSI + Prometheus + Grafana) -> automated retraining trigger (GitHub Actions, human-gated promotion)
-> drift simulation proof (score spike + alert firing).
