# Customer Churn Prediction — MLOps Pipeline with Drift Detection

A complete, end-to-end MLOps pipeline for predicting customer churn — covering data versioning, experiment tracking, model serving, and automated drift detection with human-gated retraining.

The core problem this project solves: a machine learning model can keep returning healthy `200 OK` predictions while silently becoming wrong as real-world data drifts away from what it was trained on. Unlike a crash or a timeout, this kind of failure produces no error at all. This pipeline detects that silent degradation — without needing ground-truth labels, which in real churn prediction often arrive weeks later — and triggers a retraining workflow that still requires human approval before the new model goes live.

Repo: https://github.com/sonujha78/churn-mlops

---

## Table of Contents

1. [Architecture](#architecture)
2. [Tech Stack](#tech-stack)
3. [Repository Structure](#repository-structure)
4. [How It Works — Step by Step](#how-it-works--step-by-step)
5. [Full Setup & Run Guide](#full-setup--run-guide)
6. [Simulating Drift (Demo)](#simulating-drift-demo)
7. [Troubleshooting Log — Issues Hit & How They Were Fixed](#troubleshooting-log--issues-hit--how-they-were-fixed)
8. [Known Limitations / Production Notes](#known-limitations--production-notes)

---

## Architecture

```
                    ┌──────────────────┐
                    │  Raw CSV Data    │
                    │ (customer churn) │
                    └────────┬─────────┘
                             │ dvc add / dvc push
                             ▼
                    ┌──────────────────┐
                    │   DVC Remote     │  (local storage; S3/GCS in production)
                    │  (~/dvc-storage) │
                    └────────┬─────────┘
                             │ .dvc pointer tracked in Git;
                             │ `dvc pull` restores the CSV locally
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                        src/train.py                              │
│   Trains 3 models (LogisticRegression, RandomForest, XGBoost)    │
│   Logs params/metrics/artifacts to MLflow                        │
└───────────────────────────┬──────────────────────────────────────┘
                             ▼
                    ┌──────────────────┐
                    │   MLflow         │  Experiment tracking +
                    │ Tracking Server  │  Model Registry
                    │  (SQLite backend)│  (staging / production aliases)
                    └────────┬─────────┘
                             │ models:/churn-model@production
                             ▼
                    ┌──────────────────┐
                    │  src/serve.py    │  FastAPI app
                    │  /health         │  Loads production-aliased model
                    │  /model-info     │  from the MLflow registry at
                    │  /predict        │  startup (not a hardcoded path)
                    └────────┬─────────┘
                             │ containerized (Dockerfile)
                             ▼
                    ┌──────────────────┐
                    │  Docker Container│  churn-serving-api:latest
                    │  (port 8000)     │
                    └────────┬─────────┘
                             │ every prediction logged to
                             ▼
                    ┌──────────────────┐
                    │prediction_logs   │  (features, probability,
                    │   .jsonl         │   prediction, timestamp)
                    └────────┬─────────┘
                             │ read by
                             ▼
                    ┌──────────────────┐
                    │  src/monitor.py  │  Computes PSI (Population
                    │   (port 9100)    │  Stability Index) per feature
                    │                  │  on a rolling window vs.
                    │                  │  training distribution
                    └────────┬─────────┘
                             │ /metrics scraped every 5s
                             ▼
                    ┌──────────────────┐
                    │   Prometheus     │  Stores drift_score,
                    │   (port 9090)    │  prediction volume, churn rate
                    │                  │  Alert rule: PSI > 0.25
                    └────────┬─────────┘
                             │
                             ├────────────────────────────────┐  HighDataDrift alert fires
                             │                                │  (PSI > 0.25 for 30s)
                             ▼                                │
                    ┌──────────────────┐                      │
                    │   Grafana        │                      │
                    │  (port 3000)     │                      │
                    │  Dashboard: drift│                      │
                    │  score, pred vol,│                      │
                    │  churn rate      │                      │
                    └──────────────────┘                      │
                                                              ▼
                                                ┌───────────────────────────┐
                                                │  src/trigger_retrain.py   │    Calls GitHub API
                                                │  (simulates Alertmanager  │    (repository_dispatch)
                                                │   webhook receiver)       │
                                                └───────────────────────────┘
                                                              ▼
                                                ┌───────────────────────────┐
                                                │  GitHub Actions           │    .github/workflows/retrain.yml
                                                │  (drift_alert event)      │
                                                └───────────────────────────┘
                                                              ▼
                                                ┌───────────────────────────┐
                                                │  src/retrain.py           │    Trains a new model,
                                                │                           │    logs a NEW MLflow run,
                                                │                           │    registers it with the
                                                │                           │    'staging' alias ONLY
                                                └───────────────────────────┘
                                                              ▼
                                            ⚠ Manual review required
                                            before promoting to 'production'
                                            (no auto-promotion — by design)
```

---

## Tech Stack

| Layer                          | Tool                                                |
|---------------------------------|------------------------------------------------------|
| Data & model versioning         | DVC (local remote storage)                           |
| Experiment tracking             | MLflow (tracking server + model registry)            |
| Models                          | scikit-learn (Logistic Regression, Random Forest), XGBoost |
| Model serving                   | FastAPI + Uvicorn                                    |
| Containerization                | Docker                                               |
| Drift detection                 | Custom PSI (Population Stability Index) implementation |
| Metrics & monitoring            | Prometheus (`prometheus_client`)                     |
| Dashboards                      | Grafana                                              |
| CI/CD & automated retraining    | GitHub Actions (`repository_dispatch`)               |
| Alerting                        | Prometheus alerting rules (evaluated by Prometheus itself; Alertmanager is not deployed — its webhook delivery is simulated manually by `src/trigger_retrain.py`) |

---

## Repository Structure

```
churn-mlops/
├── data/
│   └── customer_churn.csv          # DVC-tracked dataset (pointer file in Git)
├── src/
│   ├── train.py                    # Trains 3 models, logs to MLflow
│   ├── serve.py                    # FastAPI serving app
│   ├── drift.py                    # PSI drift calculation logic
│   ├── monitor.py                  # Background drift monitor, exposes Prometheus metrics
│   ├── retrain.py                  # Automated retraining script (staging alias only)
│   └── trigger_retrain.py          # Simulates alert webhook -> triggers GitHub Actions
├── mlruns/                          # MLflow artifact store (generated; not committed)
├── mlflow.db                        # MLflow tracking backend, SQLite (generated; not committed)
├── prediction_logs.jsonl            # Logged prediction requests (generated at runtime; not committed)
├── monitoring/
│   └── prometheus/
│       ├── prometheus.yml          # Scrape config
│       └── alert_rules.yml         # HighDataDrift alert rule (PSI > 0.25)
├── .github/
│   └── workflows/
│       └── retrain.yml             # Automated (drift-triggered) retraining workflow
├── docs/
│   ├── PROGRESS.md                 # Full chronological build log
│   └── screenshots/                # MLflow, registry, Grafana, Prometheus screenshots
├── Dockerfile                      # Serving API container definition
├── requirements-serve.txt          # Minimal deps for the serving container
├── .dvcignore / .dockerignore / .gitignore
└── README.md                       # This file
```

---

## How It Works — Step by Step

### Step 1 — Environment & Project Setup
Set up the project directory, Git repository connected to GitHub, a Python virtual environment, and the folder structure (`data/`, `src/`, `.github/workflows/`, `monitoring/`, `docs/`).

### Step 2 — Data & Model Versioning (DVC)
The raw dataset is tracked with **DVC**, not committed directly to Git. Git only stores the small `.dvc` pointer file; the actual CSV lives in a DVC remote (local `~/dvc-storage` for this project — cloud object storage like S3/GCS in a production setup). This means any past experiment's exact data snapshot can be reproduced.

### Step 3 — Training Pipeline & Experiment Tracking (MLflow)
`src/train.py` trains three models (Logistic Regression, Random Forest, XGBoost), logging parameters, metrics (accuracy, precision, recall, F1, AUC), and model artifacts to **MLflow**. Runs are compared in the MLflow UI (parallel coordinates plot). The best-performing model is registered in the MLflow Model Registry and promoted through `staging` → `production` aliases (MLflow's modern replacement for the older "stages" concept).

### Step 4 — Model Serving (FastAPI + Docker)
`src/serve.py` is a FastAPI app that loads whichever model is currently aliased `production` in the MLflow registry — **not a hardcoded file path** — so promoting a new model version doesn't require redeploying the API. It exposes:
- `GET /health` — liveness check
- `GET /model-info` — which model name/version/alias is currently loaded
- `POST /predict` — returns churn probability + prediction, and logs every request (input features, probability, prediction, timestamp) to `prediction_logs.jsonl`

The API is containerized with Docker; the container fetches the production model from the MLflow registry at startup rather than baking a specific model version into the image.

### Step 5 — Drift Detection (the core challenge)
`src/monitor.py` runs continuously, reading `prediction_logs.jsonl` and computing the **Population Stability Index (PSI)** for each numeric feature (tenure_months, monthly_charges, total_charges) on a rolling window of recent requests, compared against the original training distribution. This works **without needing ground-truth labels**.

Drift scores are exposed as Prometheus metrics (`model_drift_score`, `model_feature_drift_score`, `model_prediction_total`, `model_churn_prediction_total`, `model_churn_rate`) and visualized in a Grafana dashboard with three panels: Data Drift Score (PSI), Prediction Volume, and Churn Prediction Rate. A Prometheus alert rule (`HighDataDrift`) fires when the overall PSI exceeds `0.25` for 30 seconds.

### Step 6 — Automated (Human-Gated) Retraining Trigger
When the drift alert fires, a webhook-style trigger (`src/trigger_retrain.py`, simulating what a real Prometheus Alertmanager webhook receiver would do) calls the GitHub API to dispatch a `repository_dispatch` event. This runs the `retrain.yml` GitHub Actions workflow, which retrains the model (`src/retrain.py`) and logs it as a new MLflow run.

**The new model is registered with the `staging` alias only — it is never auto-promoted to `production`.** A human must review the new model's metrics and manually promote it. This is intentional: blindly auto-promoting a retrained model without review is one of the most common real-world MLOps mistakes.

### Step 7 — Simulating Drift (proving the system works)
Synthetic requests with feature values far outside the training distribution (e.g. tenure_months ~150-170 vs. the training range of 1-72, monthly_charges ~300-350 vs. 20-120) were sent to the live API to simulate a real-world shift in the customer base. This confirmed:
- The drift score rose from a normal baseline (~0.24) to **8.19** — about 33x the alert threshold
- The Grafana dashboard showed a sharp, clearly visible spike in the Data Drift Score panel
- The Prometheus `HighDataDrift` alert transitioned from `Inactive` to `Firing`

### Step 8 — Documentation
All steps, commands, and fixes were logged chronologically in [`docs/PROGRESS.md`](docs/PROGRESS.md). Screenshots of MLflow experiment comparisons, model registry promotion, Grafana dashboards (normal vs. drifted traffic), and Prometheus alert states are stored in `docs/screenshots/`.

---

## Full Setup & Run Guide

### Prerequisites
- Python 3.12+
- Docker
- Git, DVC

### Initial Setup

```bash
git clone https://github.com/sonujha78/churn-mlops.git
cd churn-mlops
python3 -m venv venv
source venv/bin/activate
pip install dvc mlflow xgboost fastapi uvicorn pydantic prometheus-client requests pandas numpy scikit-learn

dvc pull   # fetch the dataset from the DVC remote (only works if you have access to the same remote)
```

### 1. Train models and explore experiments in MLflow

```bash
python3 src/train.py
mlflow ui --host 0.0.0.0 --port 5000
```
Visit `http://localhost:5000` to compare the 3 experiment runs and manage the model registry (promote a version's alias to `staging` / `production`).

### 2. Build and run the serving API (Docker)

```bash
docker build -t churn-serving-api:latest .

docker run -d \
  --name churn-api \
  -p 8000:8000 \
  -v ~/churn-mlops/mlflow.db:/home/$(whoami)/churn-mlops/mlflow.db \
  -v ~/churn-mlops/mlruns:/home/$(whoami)/churn-mlops/mlruns \
  -v ~/churn-mlops/prediction_logs.jsonl:/app/prediction_logs.jsonl \
  -e MLFLOW_TRACKING_URI="sqlite:////home/$(whoami)/churn-mlops/mlflow.db" \
  churn-serving-api:latest

curl http://localhost:8000/health
curl http://localhost:8000/model-info
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure_months": 5, "monthly_charges": 95.5, "total_charges": 450.0, "contract_type": 0, "internet_service": 1, "tech_support": 0}'
```

### 3. Run the drift monitor

```bash
python3 src/monitor.py
```
Exposes metrics at `http://localhost:9100/metrics`.

### 4. Run Prometheus

```bash
cd monitoring/prometheus
./prometheus --config.file=prometheus.yml --web.listen-address=:9090
```
Visit `http://localhost:9090/alerts` to see the `HighDataDrift` alert state, or `http://localhost:9090/targets` to confirm the scrape target is healthy.

### 5. Run Grafana

```bash
cd monitoring/grafana-v11.2.0
./bin/grafana-server --homepath .
```
Visit `http://localhost:3000` (default login `admin`/`admin`), add Prometheus (`http://localhost:9090`) as a data source under Connections → Data sources, and build a dashboard with panels for:
- `model_drift_score` (Data Drift Score)
- `rate(model_prediction_total[1m])` (Prediction Volume)
- `model_churn_rate` (Churn Prediction Rate)

### 6. Trigger automated retraining manually (for testing)

```bash
export GITHUB_TOKEN="<your_personal_access_token_with_repo_scope>"
python3 src/trigger_retrain.py "manual_test"
```
Check the **Actions** tab on GitHub (`https://github.com/sonujha78/churn-mlops/actions`) to see the `Automated Retraining (Drift Triggered)` workflow run. On success, a new model version is registered in MLflow with the `staging` alias — review its metrics before promoting to `production`.

---

## Simulating Drift (Demo)

To reproduce the drift detection proof end-to-end:

```bash
# 1. Send "normal" traffic (matches training distribution) — drift score should stay low
for i in {1..20}; do
  tenure=$((RANDOM % 70 + 1))
  monthly=$(awk -v min=20 -v max=120 -v s=$i 'BEGIN{srand(s); print min+rand()*(max-min)}')
  total=$(awk -v t="$tenure" -v m="$monthly" -v s=$i 'BEGIN{srand(s*7); noise=(rand()-0.5)*200; val=t*m+noise; if(val<20) val=20; print val}')
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d "{\"tenure_months\": $tenure, \"monthly_charges\": $monthly, \"total_charges\": $total, \"contract_type\": $((RANDOM % 3)), \"internet_service\": $((RANDOM % 3)), \"tech_support\": $((RANDOM % 2))}" > /dev/null
done

# 2. Check drift score (should be low, e.g. ~0.1-0.25)
curl -s http://localhost:9100/metrics | grep model_drift_score

# 3. Send drifted traffic (shifted well outside training ranges)
for i in {1..60}; do
  tenure=$((RANDOM % 20 + 150))
  monthly=$((RANDOM % 50 + 300))
  total=$((tenure * monthly))
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d "{\"tenure_months\": $tenure, \"monthly_charges\": $monthly, \"total_charges\": $total, \"contract_type\": $((RANDOM % 3)), \"internet_service\": $((RANDOM % 3)), \"tech_support\": $((RANDOM % 2))}" > /dev/null
  sleep 0.15
done

# 4. Check drift score again (should be dramatically higher, e.g. > 5)
curl -s http://localhost:9100/metrics | grep model_drift_score

# 5. Check the alert state — should be "firing"
curl -s http://localhost:9090/api/v1/alerts | python3 -m json.tool
```

See `docs/screenshots/05-grafana-drift-spike.png`, `06-prometheus-alert-firing.png`, and `07-prometheus-alert-api-response.png` for the actual captured results.

---

## Troubleshooting Log — Issues Hit & How They Were Fixed

This project was built iteratively, and several real issues came up along the way. Documenting them here because they're the kind of thing that comes up in any real MLOps setup:

**1. XGBoost model logging failed with `UntrustedTypesFoundException`**
`mlflow.sklearn.log_model()` refused to save an XGBoost model due to a security check on untrusted pickled types.
*Fix:* Use `mlflow.xgboost.log_model()` instead of `mlflow.sklearn.log_model()` for XGBoost models specifically.

**2. All models scored terribly (precision/recall near 0)**
The initial synthetic dataset had features generated completely independently of the churn label — there was no real signal for any model to learn.
*Fix:* Regenerated the dataset so churn probability is actually a function of tenure, contract type, monthly charges, and tech support — giving the models something real to learn.

**3. Docker build failed with `numpy==2.5.2` not found**
`pip freeze > requirements.txt` captured the *entire* dev virtual environment (including DVC, Celery, etc.), including a numpy version incompatible with the container's Python version.
*Fix:* Created a separate, minimal `requirements-serve.txt` with only the packages the serving API actually needs, and used a matching Python base image.

**4. Docker build timed out on slow network (`pip install` `ReadTimeoutError`)**
A single large `pip install -r requirements-serve.txt` (including MLflow's many transitive dependencies) exceeded pip's default timeout on a slow connection.
*Fix:* Split the install into multiple staged `RUN pip install ...` layers (numpy/pandas/scikit-learn, then xgboost, then mlflow, then fastapi/uvicorn/pydantic) so Docker's layer caching preserves progress between retries, and increased `PIP_DEFAULT_TIMEOUT` / `PIP_RETRIES`.

**5. Container failed to load the model: `No such artifact: ''`**
MLflow's SQLite metadata store recorded the *absolute host path* to model artifacts (e.g. `/home/sonu/churn-mlops/mlruns/...`). Mounting `mlruns` at a different path inside the container (`/app/mlruns`) broke artifact resolution.
*Fix:* Mounted `mlflow.db` and `mlruns` at the **exact same absolute path** inside the container as on the host, and pointed `MLFLOW_TRACKING_URI` at that same absolute path.

**6. `ModuleNotFoundError: No module named 'src'` after fixing the path above**
Overriding the container's working directory (`-w /home/user/churn-mlops`) broke Python's ability to find the `src` package, which only existed under `/app` inside the image.
*Fix:* Removed the working-directory override — kept `WORKDIR /app` (where the code lives) while still mounting the MLflow data at its absolute host path via volume mounts and an explicit `MLFLOW_TRACKING_URI`.

**7. Drift monitor saw 0 requests despite sending traffic to the API**
The Docker container (already running and bound to port 8000) was handling all `/predict` requests and writing `prediction_logs.jsonl` *inside the container's filesystem* — not the host file the monitor script was reading.
*Fix:* Mounted `prediction_logs.jsonl` as a volume so both the container and the host-side monitor read/write the same file.

**8. PSI drift score came back absurdly high (~3.0) on what should have been "normal" traffic**
Two compounding issues: (a) `np.histogram` silently drops values that fall outside the bin edges rather than counting them, which skews bucket proportions when live data has any outliers relative to the training range; (b) with only 10 buckets and ~20 live samples, each bucket had too few samples to be statistically meaningful, making PSI extremely noisy.
*Fix:* Clip live feature values into the training data's range before binning, and adapt the number of PSI buckets to the live sample size (roughly one bucket per 5 samples, capped at 10).

**9. GitHub Actions retraining workflow failed: `FileNotFoundError: data/customer_churn.csv`**
The dataset is DVC-tracked; Git only contains the `.dvc` pointer file. A GitHub Actions runner has no access to the local DVC remote (`~/dvc-storage`), so the actual CSV was never present in the CI checkout.
*Fix:* `src/retrain.py` now checks if the dataset file exists and, if not, regenerates an equivalent synthetic dataset so the retraining pipeline can still run end-to-end in CI. In a real production setup, the DVC remote would be cloud-hosted object storage (S3/GCS) accessible from CI with proper credentials, avoiding this workaround entirely.

**10. `git push` intermittently failed with `Could not resolve host: github.com`**
Temporary local network/DNS instability, unrelated to Git or GitHub itself.
*Fix:* Verified connectivity (`ping 8.8.8.8`, `ping github.com`) and retried `git push` once the network recovered — the local commit was never lost, only the push needed retrying.

**11. `.gitignore` initially missed the downloaded Prometheus/Grafana binaries**
Extracting the Prometheus and Grafana release archives directly into `monitoring/` created thousands of untracked files (executables, static web assets, docs) that Git tried to pick up.
*Fix:* Added explicit ignore patterns for the extracted binary folders and archive files, while keeping the hand-written `prometheus.yml` and `alert_rules.yml` config files tracked.

---

## Known Limitations / Production Notes

- **DVC remote is local** (`~/dvc-storage`) for this project. In production, this would be cloud object storage (S3, GCS) accessible from both developer machines and CI runners.
- **MLflow uses a local SQLite backend.** In production, this would be a shared tracking server (e.g. a hosted MLflow server or Databricks) so multiple services can read/write consistently, and containers wouldn't need host-path volume mounts to resolve artifacts.
- Because of the two points above, the GitHub Actions retraining workflow can't access the DVC-tracked dataset directly — see Troubleshooting item #9 for the workaround and what a production fix would look like.
- Alert delivery (`src/trigger_retrain.py`) simulates what a Prometheus Alertmanager webhook receiver would do. In a full production deployment, Alertmanager would be configured with a webhook receiver to call this automatically instead of running it manually.
- The dataset itself is synthetic (generated with `numpy`/`pandas`) rather than a real customer dataset, since this project's focus is the MLOps pipeline and drift-detection mechanics rather than model accuracy on real business data.

---

For the complete chronological build log — every command run, every error hit, and how it was resolved, in the order it actually happened — see [`docs/PROGRESS.md`](docs/PROGRESS.md).
