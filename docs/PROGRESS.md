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

Commands used:

```bash
pip install dvc
dvc init
git add .dvc .dvcignore
git commit -m "chore: initialize DVC"

dvc add data/customer_churn.csv
git add data/customer_churn.csv.dvc data/.gitignore
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
Status: 🔲 Not Started

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
