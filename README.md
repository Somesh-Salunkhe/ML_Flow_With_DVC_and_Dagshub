# ML Pipeline with DVC, MLflow & DagsHub

[![DVC](https://img.shields.io/badge/DVC-945DD6?style=flat&logo=dataversioncontrol&logoColor=white)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![DagsHub](https://img.shields.io/badge/DagsHub-FF4C00?style=flat)](https://dagshub.com/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An end-to-end machine learning pipeline demonstrating best practices for **data versioning**, **experiment tracking**, and **pipeline reproducibility** using DVC, MLflow, and DagsHub. The pipeline trains a Random Forest Classifier on the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Pipeline Stages](#pipeline-stages)
- [Getting Started](#getting-started)
- [Running the Pipeline](#running-the-pipeline)
- [Experiment Tracking](#experiment-tracking)
- [Configuration](#configuration)

---

## Overview

This project addresses a common pain point in ML workflows: **lack of reproducibility and traceability**. By combining DVC for pipeline and data versioning with MLflow for experiment tracking (hosted on DagsHub), every run is logged, every dataset version is tracked, and results can be reproduced deterministically across environments.

**Key capabilities:**
- Versioned datasets and models with DVC remote storage (DagsHub / S3)
- Automatic pipeline re-execution when data, code, or parameters change
- Hyperparameter and metric logging via MLflow
- Side-by-side experiment comparison through the MLflow UI

---

## Project Structure

```
ML_Flow_With_DVC_and_Dagshub/
│
├── data/
│   ├── raw/                  # Original dataset (tracked by DVC)
│   └── processed/            # Train / val / test splits (DVC outputs)
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── src/
│   ├── preprocess.py         # Data loading and preprocessing
│   ├── train.py              # Model training + MLflow logging
│   └── evaluate.py           # Model evaluation + MLflow logging
│
├── models/
│   └── model.pkl             # Trained Random Forest model (DVC output)
│
├── .dvc/                     # DVC configuration and cache metadata
├── dvc.yaml                  # Pipeline stage definitions
├── dvc.lock                  # Locked pipeline state (for reproducibility)
├── params.yaml               # Hyperparameters and file paths
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Technology Stack

| Tool | Role |
|------|------|
| **Python 3.8+** | Core programming language |
| **DVC** | Data & model versioning, pipeline orchestration |
| **MLflow** | Experiment tracking — metrics, parameters, artifacts |
| **DagsHub** | Remote DVC storage + hosted MLflow tracking server |
| **scikit-learn** | Random Forest Classifier |
| **pandas / numpy** | Data manipulation |
| **PyYAML** | Configuration parsing (`params.yaml`) |
| **python-dotenv** | Environment variable management |

---

## Pipeline Stages

The pipeline is defined in `dvc.yaml` and consists of three sequential stages:

### 1. Preprocess
- **Script:** `src/preprocess.py`
- **Input:** `data/raw/data.csv`
- **Output:** `data/processed/train.csv`, `val.csv`, `test.csv`
- Renames columns, performs basic cleaning, and splits data into train/val/test sets.

### 2. Train
- **Script:** `src/train.py`
- **Input:** Processed CSVs from the previous stage
- **Output:** `models/model.pkl`
- Trains a Random Forest Classifier using hyperparameters from `params.yaml` and logs them to MLflow.

### 3. Evaluate
- **Script:** `src/evaluate.py`
- **Input:** `models/model.pkl`, `data/processed/test.csv`
- Loads the trained model, computes accuracy on the test set, and logs the metric to MLflow.
<img width="2544" height="1211" alt="Data_Pipeline" src="https://github.com/user-attachments/assets/9eba552a-f113-4e18-8acb-4e081c096942" />

---

## Getting Started

### Prerequisites
- Python 3.8+
- A [DagsHub](https://dagshub.com/) account (for remote storage and MLflow tracking)

### 1. Clone the repository
```bash
git clone https://github.com/Somesh-Salunkhe/ML_Flow_With_DVC_and_Dagshub.git
cd ML_Flow_With_DVC_and_Dagshub
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure DagsHub credentials
Set up your DagsHub token as environment variables (or add them to a `.env` file):
```bash
export DAGSHUB_USER_TOKEN=<your_dagshub_token>
export MLFLOW_TRACKING_URI=https://dagshub.com/<your_username>/ML_Flow_With_DVC_and_Dagshub.mlflow
```

### 5. Pull the DVC-tracked data
```bash
dvc pull
```

---

## Running the Pipeline

### Reproduce the full pipeline
```bash
dvc repro
```
DVC detects which stages are stale (based on changed inputs or parameters) and re-runs only those stages.

### Add or modify pipeline stages manually
```bash
# Preprocess stage
dvc stage add -n preprocess \
  -p preprocess.input,preprocess.train_output,preprocess.val_output,preprocess.test_output \
  -d src/preprocess.py -d data/raw/data.csv \
  -o data/processed/train.csv -o data/processed/val.csv -o data/processed/test.csv \
  python src/preprocess.py

# Train stage
dvc stage add -n train \
  -p train.train_data,train.val_data,train.test_data,train.model \
  -d src/train.py -d data/processed/train.csv -d data/processed/val.csv -d data/processed/test.csv \
  -o models/model.pkl \
  python src/train.py

# Evaluate stage
dvc stage add -n evaluate \
  -p train.test_data,train.model \
  -d src/evaluate.py -d models/model.pkl -d data/processed/test.csv \
  python src/evaluate.py
```

### Check pipeline status
```bash
dvc status       # Shows which stages are outdated
dvc dag          # Visualizes the pipeline DAG
```

---

## Experiment Tracking

All experiments are tracked via MLflow on DagsHub. After running the pipeline, open the MLflow UI to:

- Compare runs across different hyperparameter configurations
- View logged metrics (e.g., accuracy) and parameters (e.g., `n_estimators`, `max_depth`)
- Download saved model artifacts

```bash
# Launch local MLflow UI (if tracking locally)
mlflow ui
# Open: http://127.0.0.1:5000
```

Or navigate directly to your DagsHub repository's **Experiments** tab to view all tracked runs.
<img width="319" height="312" alt="accuracy" src="https://github.com/user-attachments/assets/d3f11b69-8ce5-4c15-9bdd-44ddc7417972" />
<img width="1086" height="450" alt="newplot" src="https://github.com/user-attachments/assets/89dbca2b-8487-41fb-8933-d32ec42c2fc8" />

---

## Configuration

All hyperparameters and file paths are centralized in `params.yaml`. Modify this file to experiment with different model configurations — DVC will automatically detect the change and re-run affected stages on the next `dvc repro`.

```yaml
# Example params.yaml structure
train:
  n_estimators: 100
  max_depth: 5
  train_data: data/processed/train.csv
  val_data: data/processed/val.csv
  test_data: data/processed/test.csv
  model: models/model.pkl
```

---

## License

This project is open-source and available under the [MIT License](LICENSE).
