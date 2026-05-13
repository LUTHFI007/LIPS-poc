# LIPS Power Grid Benchmark — POC

A Gradio web app that lets you browse power-grid datasets and models on HuggingFace and evaluate surrogate models against the [LIPS benchmark](https://github.com/IRT-SystemX/LIPS). This is a Proof of Concept(PoC) created as part of building a Pipeline to Automate LIPS Banchmarks.

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
| **Data Hub** | Search HuggingFace for power-grid datasets. Click any URL to open it in a new tab. |
| **Model Hub** | Search HuggingFace for power-grid models. Click any URL to open it in a new tab. |
| **Scoreboard** | Upload a trained model + dataset, run LIPS evaluation, and view the ranked leaderboard. |

---

## 2. How the pipeline works

```
User uploads model + dataset
        │
        ▼
  scoreboard.py
        │
        ▼
  save_result() → scoreboard.json
        │
        ▼
  load_scoreboard() → sorted DataFrame → Leaderboard table
```

**Scoring formula**

```
final_score = α × ml_score + β × physics_score
              α = 0.5,  β = 0.5
```

- `ml_score` — derived from mean MAE across all output variables, normalized against a 200 A reference: `max(0, 1 − MAE/200)`
- `physics_score` — derived from current constraint violation proportion: `max(0, 1 − avg_violation)`

**Data Hub / Model Hub**

```
User types keyword (default: "powergrid")
        │
        ▼
  HfApi().list_datasets(search=keyword, limit=50)
  HfApi().list_models(search=keyword, limit=50)
        │
        ▼
  Results rendered in DataFrame with clickable HuggingFace URLs
```

---

## 3. Prerequisites

- Windows 10 (build 19041+) or Windows 11
- WSL 2 enabled
- Ubuntu 22.04 (or 24.04) from the Microsoft Store
- Python 3.10+ (Python 3.13 was used in development)
- ~4 GB free disk space for LIPS reference data

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

## 5. App setup

### 5.1 Clone this repository

```bash
cd ~
git clone https://github.com/LUTHFI007/lips-poc
cd lips-poc
```

### 5.2 Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 5.3 Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

```
huggingface_hub
datasets
gradio
pandas
```

---

## 6. Running the app

```bash
cd ~/lips-poc
source .venv/bin/activate   # if using a venv
python main.py
```

Gradio will print a local URL (e.g. `http://127.0.0.1:7860`). Open it in your Windows browser — WSL networking is bridged automatically.

---

## 7. Project structure

```
lips-poc/
├── main.py                  # Gradio UI and event handlers
├── requirements.txt
├── scoreboard.json          # Created automatically on first evaluation
└── lips_poc/
    ├── __init__.py
    ├── config.py            # HuggingFace org/repo IDs
    ├── data_hub.py          # HuggingFace dataset search
    ├── model_hub.py         # HuggingFace model search
    └── scoreboard.py        # LIPS evaluation logic and leaderboard I/O
```
