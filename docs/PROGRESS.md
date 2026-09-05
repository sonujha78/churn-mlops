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
Status: 🔲 Not Started

## Step 5: Drift Detection
Status: 🔲 Not Started

## Step 6: Automated Retraining Trigger
Status: 🔲 Not Started

## Step 7: Simulating Drift
Status: 🔲 Not Started

## Step 8: Documentation & Screenshots
Status: 🔲 Not Started
