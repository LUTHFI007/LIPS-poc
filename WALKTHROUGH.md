# LIPS Power Grid Benchmark — Technical Walkthrough

---

## 1. What This Project Does

This is a Gradio web application that lets researchers train machine learning models on power grid simulation data, benchmark them against the LIPS framework, and publish scores to a persistent leaderboard.

The system has three concerns:

**Data Hub** — Browse, pull, and push power-grid datasets stored on Hugging Face. Internally this calls `huggingface_hub.HfApi.list_datasets`, `datasets.load_dataset`, and `DatasetDict.push_to_hub`.

**Model Hub** — Browse, pull, and push trained model checkpoints. Uses `HfApi.list_models`, `huggingface_hub.snapshot_download`, and per-file `HfApi.upload_file` with a generated `ModelCard`.

**Scoreboard** — Runs the LIPS `PowerGridBenchmark` against a submitted `TorchSimulator` checkpoint. Extracts two metrics (`ml_score`, `physics_score`), combines them into a weighted `final_score`, appends the record to a local `scoreboard.json`, and re-renders the leaderboard.

---

## 2. Architecture and Code Layout

```
lips-poc/
├── main.py                  # Gradio UI: event wiring and handler functions
├── requirements.txt         # Runtime Python dependencies
├── scoreboard.json          # Append-only leaderboard (auto-created on first run)
└── lips_poc/
    ├── __init__.py          # Empty package marker
    ├── config.py            # Module-level constants: ORG_NAME, dataset/model IDs
    ├── data_hub.py          # HF dataset operations (list / pull / push)
    ├── model_hub.py         # HF model operations (list / pull / push)
    └── scoreboard.py        # LIPS evaluation, score math, JSON persistence
```

### External dependencies (outside this repo)

| Path | Role |
|------|------|
| `/home/luthfi/LIPS/` | LIPS benchmark library — installed via `pip install -e .` |
| `/home/luthfi/LIPS/reference_data/powergrid/l2rpn_case14_sandbox/` | Pre-generated reference dataset used by `PowerGridBenchmark(load_data_set=True)` |
| `/home/luthfi/LIPS/lips/tests/configs/powergrid/benchmarks/l2rpn_case14_sandbox.ini` | INI config for the benchmark (variable list, eval splits, thresholds) |
| `/home/luthfi/LIPS/lips/tests/configs/powergrid/simulators/torch_fc.ini` | INI config for `TorchFullyConnected` (hidden layers, activation, learning rate) |
| `~/.cache/huggingface/hub/` | Hugging Face local cache — where `load_dataset` / `snapshot_download` write files |

### config.py — central constants

```python
ORG_NAME        = "lips-poc"
LIPS_DATASET_ID = "lips-poc/power-grid-benchmark"
LIPS_MODEL_ID   = "lips-poc/baseline-fc"
```

All three modules import from here; change the org or default IDs in one place.

---

## 3. Module Reference

### 3.1 `lips_poc/scoreboard.py`

This is the core evaluation module.

#### Score weights

```python
_ALPHA = 0.5   # ML weight
_BETA  = 0.5   # Physics weight
```

`final_score = _ALPHA * ml_score + _BETA * physics_score`

Currently equal weighting. Adjusting these constants re-balances the composite score without touching the evaluation logic.

#### `evaluate_model(model_path, dataset_path, username, model_name) → dict`

1. Calls `_run_lips_benchmark(model_path, dataset_path)` inside a `try/except`.
2. On any exception, falls back to `_mock_scores(model_path, dataset_path)` and logs a warning.
3. Computes `final_score` and returns a dict with six fields: `username`, `model_name`, `ml_score`, `physics_score`, `final_score`, `timestamp` (ISO-8601 UTC).

#### `_run_lips_benchmark(model_path, dataset_path) → tuple[float, float]`

Hardcoded config paths (all under `/home/luthfi/LIPS/`):

```python
BENCH_CONFIG_PATH = ".../benchmarks/l2rpn_case14_sandbox.ini"
SIM_CONFIG_PATH   = ".../simulators/torch_fc.ini"
BENCHMARK_PATH    = ".../reference_data/powergrid/l2rpn_case14_sandbox"
```

**Benchmark instantiation:**
```python
benchmark = PowerGridBenchmark(
    benchmark_path=BENCHMARK_PATH,
    config_path=BENCH_CONFIG_PATH,
    benchmark_name="Benchmark1",
    load_data_set=True,          # reads train/val/test splits from disk
    log_path=None,
    eval_dict={
        "ML":      ["MSE_avg", "MAE_avg"],
        "Physics": ["CURRENT_POS"],
        "IndRed":  [],
        "OOD":     []
    }
)
```

`eval_dict` controls which metric families are computed. `CURRENT_POS` is the overload detection metric.

**Simulator instantiation and restore:**
```python
augmented_simulator = TorchSimulator(
    model=TorchFullyConnected,
    sim_config_path=SIM_CONFIG_PATH,
    name="torch_fc",
    bench_config_path=BENCH_CONFIG_PATH,
    bench_config_name="Benchmark1",
    log_path=None,
)
augmented_simulator.restore(model_path)   # loads weights from the uploaded checkpoint
```

**Evaluation call:**
```python
results = benchmark.evaluate_simulator(
    dataset="test",
    augmented_simulator=augmented_simulator,
    save_path=None,
    save_predictions=False,
)
```

`results` is a nested dict: `results["test"]["ML"]` and `results["test"]["Physics"]`.

**ML score derivation (MAE-based):**

`MAE_avg` is `{variable_name: float}` — mean absolute error averaged per output variable.

```python
mae_dict = ml_metrics.get("MAE_avg", {})
mae = sum(mae_dict.values()) / len(mae_dict)   # average across variables
ml_score = max(0.0, 1.0 - mae / 200.0)
```

Reference scale is **200 A** (amperes). A model with 0 A average MAE scores 1.0; a model with ≥200 A MAE scores 0.0.

**Physics score derivation (CURRENT_POS):**

`CURRENT_POS` is `{variable_name: {"Violation_proportion": float}}` — fraction of test samples where the predicted current exceeds the thermal limit for each line.

```python
current_pos = physics_metrics.get("CURRENT_POS", {})
violations = [v["Violation_proportion"] for v in current_pos.values()]
physics_score = max(0.0, 1.0 - sum(violations) / len(violations))
```

0 violations → `physics_score = 1.0`. 100% violations on all lines → `physics_score = 0.0`.

**The l2rpn_case14_sandbox grid:**

This is a 14-bus IEEE power network used in the L2RPN reinforcement learning challenge. It has 20 power lines. The LIPS benchmark trains surrogate models to replace the physics simulator (Pandapower) and predict line flows (in amperes) from grid topology + injection inputs. The `TorchFullyConnected` model is a feed-forward neural net whose hyperparameters come from `torch_fc.ini`.

#### `_mock_scores(model_path, dataset_path) → tuple[float, float]`

Deterministic fallback using a hash seed:

```python
seed = hash(model_path + dataset_path) & 0xFFFF
ml_score      = 0.5 + (seed % 1000) / 2000.0   # range [0.5, 1.0)
physics_score = 0.4 + (seed % 800)  / 2000.0   # range [0.4, 0.8)
```

The same path combination always produces the same fake scores. These scores are not physically meaningful and should not be used for real comparisons.

#### `save_result(result_dict) → None`

Reads the entire `scoreboard.json` into memory, appends the new record, and writes the full list back. Not safe for concurrent writes — this is a single-user POC.

#### `load_scoreboard() → pd.DataFrame`

Reads `scoreboard.json`, constructs a DataFrame, and sorts descending by `final_score`. Returns an empty DataFrame with the correct columns if no file exists.

---

### 3.2 `lips_poc/data_hub.py`

A thin wrapper over `huggingface_hub` and `datasets`.

```python
_api = HfApi()   # shared singleton; reads HF_TOKEN from env or ~/.cache/huggingface/token
```

#### `list_power_grid_datasets() → list[dict]`

```python
_api.list_datasets(author=ORG_NAME, tags=["power-grid"])
```

Returns a generator of `DatasetInfo` objects. Filtered to the `lips-poc` org with the `power-grid` tag. Each is flattened to `{id, tags, last_modified}`.

#### `pull_dataset(dataset_id, revision="main") → DatasetDict`

```python
load_dataset(dataset_id, revision=revision)
```

Downloads to `~/.cache/huggingface/hub/datasets--{owner}--{name}/`. Returns a `DatasetDict` keyed by split name.

#### `push_dataset(local_dataset, username, base_dataset_id, description) → str`

Versioned repo naming: `{username}/{base_dataset_id}-{username}-v{timestamp}` where timestamp is UTC in `%Y%m%d%H%M%S` format. A new repo is always created (`exist_ok=False`) — no overwriting.

If `description` is provided, a minimal `README.md` is uploaded via `upload_file`. Then `DatasetDict.push_to_hub(repo_id, private=True)` streams all parquet shards.

Returns the canonical HF URL: `https://huggingface.co/datasets/{repo_id}`.

---

### 3.3 `lips_poc/model_hub.py`

Same pattern as `data_hub.py`, but for model repositories.

#### `list_power_grid_models() → list[dict]`

```python
_api.list_models(author=ORG_NAME, tags=["power-grid"])
```

#### `pull_model(model_id, revision="main") → str`

```python
snapshot_download(repo_id=model_id, revision=revision)
```

Unlike `hf_hub_download` (single file), `snapshot_download` mirrors the full repository tree to a single local directory and returns that path. This is what the Scoreboard tab passes to `augmented_simulator.restore(model_path)`.

#### `push_model(local_model_path, username, base_model_id, framework, description) → str`

1. Creates a private repo with `HfApi.create_repo(repo_type="model")`.
2. Builds a `ModelCard` from a Jinja template:
   ```python
   ModelCard.from_template(
       card_data=ModelCardData(tags=["power-grid", framework], library_name=framework),
       template_str=_model_card_template(...)
   )
   ```
   and pushes it with `card.push_to_hub(repo_id)` so the README appears before weights.
3. Walks `local_model_path` with `Path.rglob("*")` and uploads every file individually via `HfApi.upload_file`, preserving the subdirectory structure relative to `local_model_path`.

Returns `https://huggingface.co/models/{repo_id}`.

---

### 3.4 `main.py` — UI and Event Wiring

Built with **Gradio Blocks** API. Three tabs, each following the same pattern: display widgets → declare event handlers → wire them with `.click()` / `app.load()`.

```python
app.load(on_load_datasets, outputs=datasets_table)     # fires on page load
app.load(on_load_models,   outputs=models_table)
app.load(_scoreboard_df,   outputs=scoreboard_table)
```

`app.load` runs all three handlers once when the browser connects, pre-populating the tables without requiring the user to click a button.

Key evaluation handler:

```python
def on_evaluate(model_file, dataset_file, username, model_name) -> tuple[str, pd.DataFrame]:
    model_path   = model_file.name   # Gradio writes uploads to a tempfile
    dataset_path = dataset_file.name
    result = evaluate_model(model_path, dataset_path, username.strip(), model_name.strip())
    save_result(result)
    return status_string, _scoreboard_df()   # both outputs update atomically
```

`_scoreboard_df()` re-reads `scoreboard.json` from disk and injects a `Rank` column (1-indexed) before returning.

---

## 4. Data Flow: End-to-End Evaluation

```
User uploads checkpoint + dataset folder
        │
        ▼
on_evaluate() in main.py
        │
        ▼
evaluate_model() in scoreboard.py
        │
        ├──▶ _run_lips_benchmark()
        │        │
        │        ├── PowerGridBenchmark.__init__  (loads test split from BENCHMARK_PATH)
        │        ├── TorchSimulator.__init__       (builds model graph from torch_fc.ini)
        │        ├── augmented_simulator.restore() (loads .pt weights from uploaded path)
        │        └── benchmark.evaluate_simulator()
        │                │
        │                ├── model.predict(test_inputs) → predicted ampere values
        │                ├── compute MAE_avg vs. ground truth
        │                └── check CURRENT_POS violations vs. thermal limits
        │
        ├── ml_score      = max(0, 1 - mae / 200)
        ├── physics_score = max(0, 1 - mean(violation_proportions))
        └── final_score   = 0.5 * ml_score + 0.5 * physics_score
                │
                ▼
        save_result() → appends to scoreboard.json
                │
                ▼
        _scoreboard_df() → reads JSON, sorts by final_score desc, adds Rank column
                │
                ▼
        Gradio table component re-renders in browser
```

---

## 5. How to Run

```bash
conda activate venv_poc
cd /home/luthfi/lips-poc
python main.py
```

Open **http://127.0.0.1:7860** in your browser. Press `Ctrl + C` to stop.

`venv_poc` is the only environment with `lips-benchmark` (editable install of `/home/luthfi/LIPS`) and all of `requirements.txt` installed. Running with a different environment will raise `ModuleNotFoundError`.

---

## 6. How to Use Each Tab

### Data Hub Tab

**List datasets** — table auto-loads on page open via `app.load`. Queries `lips-poc` org on HF with the `power-grid` tag. Click **Refresh list** to re-query.

**Pull a dataset**
1. Copy a dataset ID from the table (e.g. `lips-poc/power-grid-benchmark`).
2. Paste into "Dataset ID" and click **Pull**.
3. `load_dataset` downloads Parquet shards to `~/.cache/huggingface/hub/`. Status shows the dataset ID on success.

**Push a dataset**
1. Select files saved with `DatasetDict.save_to_disk()` (Arrow/Parquet format).
2. Fill in Username and Base dataset ID.
3. Click **Push** — new private repo is created as `{username}/{base_dataset_id}-{username}-v{timestamp}`.

---

### Model Hub Tab

Mirrors Data Hub layout.

**Pull a model**
1. Paste a model ID (e.g. `lips-poc/baseline-fc`) and click **Pull**.
2. `snapshot_download` clones the full repo to `~/.cache/huggingface/hub/`. The returned local path is what you pass to the Scoreboard.

**Push a model**
1. Select the local model directory (must contain the checkpoint files `TorchSimulator.restore` expects).
2. Fill in Username, Base model ID, Framework.
3. Click **Push** — files are uploaded individually, preserving directory structure. A `ModelCard` README is written first.

---

### Scoreboard Tab

**Submit a model**
1. Upload the model checkpoint directory (the local path from "Pull a model").
2. Upload the dataset folder.
3. Enter a Username and Model name.
4. Click **Submit & Evaluate**.

The LIPS benchmark runs in the Gradio server process (blocking, 30–90 s). When done, the Result box shows:

```
Evaluation complete — final score: 0.7668  (ML: 0.5335, Physics: 1.0000)
```

The leaderboard updates atomically with the new row.

**Leaderboard columns**

| Column | Source | Range |
|--------|--------|-------|
| Rank | Computed from sort position | 1 = best |
| Username | User input at submission | — |
| Model Name | User input at submission | — |
| ML Score | `1 - mean(MAE_avg) / 200` | 0–1 |
| Physics Score | `1 - mean(CURRENT_POS violation proportions)` | 0–1 |
| Final Score | `0.5 * ML + 0.5 * Physics` | 0–1 |
| Timestamp | UTC ISO-8601 from `datetime.now(timezone.utc)` | — |

Sorted descending by Final Score. All submissions persisted in `scoreboard.json`.

---

## 7. Example Walkthrough

```bash
conda activate venv_poc
cd /home/luthfi/lips-poc
python main.py
```

Open **http://127.0.0.1:7860**.

**Pull a dataset**
1. Data Hub tab → find `lips-poc/power-grid-benchmark` → paste into "Dataset ID" → **Pull**.
2. Files land in `~/.cache/huggingface/hub/datasets--lips-poc--power-grid-benchmark/`.

**Pull a model**
1. Model Hub tab → find `lips-poc/baseline-fc` → paste into "Model ID" → **Pull**.
2. Note the full local path from the Status box (e.g. `~/.cache/huggingface/hub/models--lips-poc--baseline-fc/snapshots/<sha>/`).

**Submit for evaluation**
1. Scoreboard tab → upload the model folder and dataset folder → enter Username + Model name → **Submit & Evaluate**.
2. Expected result for baseline-fc:
   ```
   Evaluation complete — final score: 0.7668  (ML: 0.5335, Physics: 1.0000)
   ```
   Physics 1.0 means zero thermal limit violations on all test samples.

**Check the leaderboard**
Scroll down — your row appears at the top, ranked 1.

---

## 8. File Locations Reference

| What | Where |
|------|-------|
| Leaderboard records | `/home/luthfi/lips-poc/scoreboard.json` |
| HF dataset cache | `~/.cache/huggingface/hub/datasets--*` |
| HF model cache | `~/.cache/huggingface/hub/models--*` |
| Reference grid data | `/home/luthfi/LIPS/reference_data/powergrid/l2rpn_case14_sandbox/` |
| Benchmark INI config | `/home/luthfi/LIPS/lips/tests/configs/powergrid/benchmarks/l2rpn_case14_sandbox.ini` |
| Simulator INI config | `/home/luthfi/LIPS/lips/tests/configs/powergrid/simulators/torch_fc.ini` |
| LIPS source | `/home/luthfi/LIPS/` (editable install) |

To inspect all past scores:
```bash
cat /home/luthfi/lips-poc/scoreboard.json
```

---

## 9. Common Errors

**`ModuleNotFoundError: No module named 'lips_poc'`**

Wrong directory or wrong environment.

```bash
conda activate venv_poc
cd /home/luthfi/lips-poc
python main.py
```

---

**`ModuleNotFoundError: No module named 'gradio'` (or any package from `requirements.txt`)**

The `venv_poc` environment is not active.

```bash
conda activate venv_poc
```

---

**`ModuleNotFoundError: No module named 'lips'`**

The LIPS library editable install is missing or broken. Reinstall it:

```bash
conda activate venv_poc
pip install -e /home/luthfi/LIPS
```

---

**`LIPS evaluation failed, using mock scores`**

`_run_lips_benchmark` threw an exception and the app fell back to `_mock_scores`. Mock scores are hash-deterministic but physically meaningless. Common root causes:

- `model_path` does not contain the expected checkpoint files (`TorchSimulator.restore` fails).
- `BENCH_CONFIG_PATH` or `SIM_CONFIG_PATH` not found — check `/home/luthfi/LIPS/lips/tests/configs/`.
- `BENCHMARK_PATH` missing — check `/home/luthfi/LIPS/reference_data/powergrid/l2rpn_case14_sandbox/`.
- `venv_poc` not activated — `lips` module not importable.

To see the exact exception, look at the terminal where `python main.py` is running.

---

**`numpy` version conflict on `pip install`**

`lips-benchmark` pins `numpy==1.25.2`; `pandapower` requests `>=1.26`. The conflict is benign for inference — the benchmark still runs. If numpy was upgraded, restore it:

```bash
conda activate venv_poc
pip install "numpy==1.25.2"
```

---

**Browser shows blank page or "Connection refused"**

The Gradio server is not running. Start it:

```bash
python main.py
```

Confirm you see `Running on local URL: http://127.0.0.1:7860` before opening the browser.

---

**Hugging Face upload fails with authentication error**

```bash
huggingface-cli login
```

Paste a write-access token from https://huggingface.co/settings/tokens. The token is stored in `~/.cache/huggingface/token` and picked up automatically by `HfApi`.

---

**`exist_ok=False` error on push**

The repo name collision guard uses a UTC timestamp, so duplicates are extremely unlikely. If you see this, the clock may have been the same second for a previous push. Wait one second and retry.
