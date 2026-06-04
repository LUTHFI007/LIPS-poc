# LIPS Power Grid Benchmark — POC

A Streamlit web app that lets you browse power-grid datasets and models on HuggingFace, select them, and run LIPS benchmark evaluations against a ranked leaderboard. This is a Proof of Concept created as part of building a pipeline to automate LIPS benchmarks.

---

## Table of Contents

1. [Project overview](#1-project-overview)
2. [How the pipeline works](#2-how-the-pipeline-works)
3. [Prerequisites](#3-prerequisites)
4. [WSL environment setup](#4-wsl-environment-setup)
5. [LIPS library setup](#5-lips-library-setup)
6. [App setup](#6-app-setup)
7. [Running the app](#7-running-the-app)
8. [Project structure](#8-project-structure)

---

## 1. Project overview

The app has three tabs:

| Tab | What it does |
|---|---|
| **Data Hub** | Lists power-grid datasets from the `lips-poc` HuggingFace org. Click a row to select a dataset for evaluation. |
| **Model Hub** | Lists power-grid models from the `lips-poc` HuggingFace org. Click a row to select a model for evaluation. |
| **Scoreboard** | Shows the selected dataset and model, runs LIPS evaluation on click, and displays the ranked leaderboard. |

Supported model types (auto-detected from the HuggingFace repo name or model card tag):

| Model type tag | Architecture |
|---|---|
| `torch_fc` | PyTorch fully-connected |
| `tf_fc` | TensorFlow/Keras fully-connected |
| `tf_leapnet` | TensorFlow LeapNet |

---

## 2. How the pipeline works

```
User selects dataset + model in the UI, clicks Evaluate
        │
        ▼
  evaluation_runner.py
  ├── _resolve_model_type()   — reads HF model card tag, falls back to repo name
  ├── _download_model()       — snapshots model files into models/<repo-slug>_DEFAULT/
  ├── _load_simulator()       — builds the correct LIPS augmented simulator and restores weights
  └── run_evaluation()        — runs benchmark.evaluate_simulator() for test + test_ood_topo splits
        │
        ▼
  extract_scores()            — pulls MSE / MAE / MAPE_90 and Physics Violation % from raw results
        │
        ▼
  scoreboard.json             — new row appended, leaderboard re-rendered in the UI
```

**Scoreboard columns**

| Column | Description |
|---|---|
| MSE / MAE / MAPE_90 | ML metrics on the in-distribution test set |
| MSE (ood) / MAE (ood) / MAPE_90 (ood) | ML metrics on the out-of-distribution topology test set |
| Physics Viol. % | Average current-constraint violation percentage |

**Data Hub / Model Hub**

```
App loads on startup (cached 5 min)
        │
        ▼
  HfApi().list_datasets(author="lips-poc", limit=50)
  HfApi().list_models(author="lips-poc", limit=50)
        │
        ▼
  Results rendered in interactive Streamlit dataframe (click to select)
```

---

## 3. Prerequisites

- Windows 10 (build 19041+) or Windows 11
- WSL 2 enabled
- Ubuntu 22.04 (or 24.04) from the Microsoft Store
- Miniconda or Python 3.10 inside WSL
- ~4 GB free disk space for LIPS reference data and model downloads

---

## 4. WSL environment setup

### 4.1 Enable WSL 2

Open **PowerShell as Administrator** and run:

```powershell
wsl --install
```

Restart your machine when prompted. This installs WSL 2 and Ubuntu by default.

If WSL was already installed but on version 1, upgrade it:

```powershell
wsl --set-default-version 2
```

### 4.2 Launch Ubuntu and create a user

After the restart, open Ubuntu from the Start menu. Create your UNIX username and password when prompted.

### 4.3 Update packages

```bash
sudo apt update && sudo apt upgrade -y
```

### 4.4 Install Python and pip

Ubuntu 22.04 ships with Python 3.10. For a newer version:

```bash
sudo apt install -y python3 python3-pip python3-venv
```

Verify:

```bash
python3 --version
pip3 --version
```

### 4.5 Install Git

```bash
sudo apt install -y git
```

---

## 5. LIPS library setup

The LIPS library and its heavy dependencies (TensorFlow, PyTorch, Grid2Op, leap_net) must be installed separately into the conda/venv environment before running the app.

```bash
# recommended: use a conda env with Python 3.10
conda create -n venv_poc python=3.10 -y
conda activate venv_poc

# install LIPS and all power-grid extras
pip install lips[powergrid]

# leap_net is required for tf_leapnet models
pip install leap_net
```

The benchmark reference datasets must be downloaded once and placed at the path configured in `dataset_registry.json`:

```
datasets/powergrid/l2rpn_case14_sandbox/
```

---

## 6. App setup

### 6.1 Clone this repository

```bash
cd ~
git clone https://github.com/LUTHFI007/lips-poc
cd lips-poc
```

### 6.2 Activate the environment

```bash
conda activate venv_poc
```

### 6.3 Install app dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

```
huggingface_hub
datasets
streamlit
pandas
```

---

## 7. Running the app

```bash
cd ~/lips-poc
conda activate venv_poc
streamlit run main.py
```

Streamlit will print a local URL (e.g. `http://localhost:8501`). Open it in your Windows browser — WSL networking is bridged automatically.

---

## 8. Project structure

```
lips-poc/
├── main.py                      # Streamlit UI and event handlers
├── evaluation_runner.py         # Model download, loading, and LIPS evaluation logic
├── dataset_registry.json        # Maps HF dataset IDs to local benchmark paths/configs
├── requirements.txt             # App-level Python dependencies
├── scoreboard.json              # Auto-created on first evaluation; stores all results
├── configurations/
│   └── powergrid/
│       ├── benchmarks/
│       │   └── l2rpn_case14_sandbox.ini   # LIPS benchmark config
│       └── simulators/
│           ├── torch_fc.ini               # Simulator config for torch_fc models
│           ├── tf_fc.ini                  # Simulator config for tf_fc models
│           └── tf_leapnet.ini             # Simulator config for tf_leapnet models
├── datasets/
│   └── powergrid/
│       └── l2rpn_case14_sandbox/          # LIPS reference data (downloaded separately)
├── models/                                # Auto-populated by evaluation_runner on first run
└── lips_poc/
    ├── __init__.py
    ├── data_hub.py                        # HuggingFace dataset search
    ├── model_hub.py                       # HuggingFace model search
    └── scoreboard.py                      # Legacy scoring helpers (superseded by evaluation_runner)
```
