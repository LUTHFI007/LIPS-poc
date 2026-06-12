import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from lips_poc.data_hub import search_datasets
from lips_poc.model_hub import search_models, validate_upload_inputs, upload_model
from evaluation_runner import run_evaluation, extract_scores

_ROOT = Path(__file__).parent

_SB_BASE_COLS = ["Model", "Dataset", "Benchmark", "Physics Viol. %", "Timestamp"]

SCOREBOARD_FILE = _ROOT / "scoreboard.json"
_DOCS_DIR = _ROOT / "docs"

# The exact benchmark config the scoreboard evaluates against (see
# lips_poc/scoreboard.py). Served for download so users train against a
# byte-identical file — never copied/edited, so train-time == eval-time.
_BENCH_CONFIG_PATH = _ROOT / "configurations/powergrid/benchmarks/benchmark.ini"


def _doc_bytes(filename: str) -> bytes:
    """Read a file from docs/ as bytes for st.download_button. Passing bytes
    (not an open handle) makes browsers honour the exact download file_name."""
    return (_DOCS_DIR / filename).read_bytes()

with (_ROOT / "dataset_registry.json").open() as f:
    DATASET_REGISTRY = json.load(f)


@st.cache_data(ttl=300)
def _fetch_datasets() -> list[dict]:
    return search_datasets("")


@st.cache_data(ttl=300)
def _fetch_models() -> list[dict]:
    return search_models("")


def _load_scoreboard() -> pd.DataFrame:
    try:
        with SCOREBOARD_FILE.open() as f:
            rows = json.load(f)
        if not rows:
            return pd.DataFrame(columns=_SB_BASE_COLS)
        return pd.DataFrame(rows)
    except FileNotFoundError:
        return pd.DataFrame(columns=_SB_BASE_COLS)


def _save_scoreboard(df: pd.DataFrame) -> None:
    with SCOREBOARD_FILE.open("w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)


# Clear stale selections on every new browser session
if "initialized" not in st.session_state:
    st.session_state.selected_dataset = None
    st.session_state.selected_model   = None
    st.session_state.initialized      = True

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="LIPS Power Grid Benchmark", layout="wide")
st.title("LIPS Power Grid Benchmark POC")

tab_data, tab_model, tab_scoreboard = st.tabs(["Data Hub", "Model Hub", "Scoreboard"])

# ── Data Hub ──────────────────────────────────────────────────────────────────

with tab_data:
    st.subheader("Data Hub")
    st.caption("Click a row to select it.")
    datasets = _fetch_datasets()
    if not datasets:
        st.warning("No datasets found on HuggingFace (lips-poc org).")
    else:
        ds_df = pd.DataFrame(datasets)
        ds_event = st.dataframe(
            ds_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="ds_table",
        )
        selected_ds_rows = ds_event.selection.rows
        if selected_ds_rows:
            st.session_state.selected_dataset = ds_df.iloc[selected_ds_rows[0]]["Dataset ID"]
            st.success(f"Selected: **{st.session_state.selected_dataset}**")

# ── Model Hub ─────────────────────────────────────────────────────────────────

with tab_model:
    st.subheader("Model Hub")
    st.caption("Click a row to select it.")
    models = _fetch_models()
    if not models:
        st.warning("No models found on HuggingFace (lips-poc org).")
    else:
        m_df = pd.DataFrame(models)
        m_event = st.dataframe(
            m_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="m_table",
        )
        selected_m_rows = m_event.selection.rows
        if selected_m_rows:
            st.session_state.selected_model = m_df.iloc[selected_m_rows[0]]["Model ID"]
            st.success(f"Selected: **{st.session_state.selected_model}**")

    st.divider()
    st.subheader("Build Your Own Model")

    with st.expander("How to build and submit a model — read the instructions carefully!"):
        st.markdown("""
        ### Step-by-step

        **Step 1 — Download the template for your framework**
        Use the buttons below. The template is a Python file with one class you
        fill in. Everything else (data loading, preprocessing, training loop,
        save/restore) is handled by LIPS automatically.

        **Step 2 — Fill in `build_model()`**
        Open the template and replace the example architecture in `build_model()`
        with your own. Read the comments — they explain exactly what variables are
        available and what you must assign.

        **Step 3 — Download and adjust the `.ini` config**
        The `.ini` file controls your model's hyperparameters (layer sizes, learning
        rate, etc.). Download the one for your framework below, rename it (any name
        except `benchmark.ini`), and update the values to match what
        you put in `build_model()`.

        **Step 3b — Download the benchmark config**
        Also download `benchmark.ini` (button below). This is the fixed
        benchmark every model is scored against. Point `BENCH_CONFIG` in the template
        at it and train against it **as-is** — do not edit or rename it. It is *not*
        part of your ZIP; it stays on your machine for training only. Training against
        a different benchmark config produces invalid scores.

        **Step 4 — Train your model**
        Run the training script at the bottom of the template file:
        ```
        python augmented_simulator.py
        ```
        After training, LIPS saves your model into a folder. That folder will contain:
        `model.weights.h5`, `config.json`, `scaler_params.json`, `losses.json`,
        `metadata.json`.

        **Step 5 — Assemble the ZIP**
        Copy two files into your saved model folder:
        - `augmented_simulator.py` (your filled-in template)
        - `simulator.ini` (your adjusted config file)

        Then ZIP the entire folder. Your ZIP must look exactly like this:
        """)

        st.code("""
your-model-name.zip
├── model.weights.h5          ← from sim.save()
├── config.json               ← from sim.save()
├── scaler_params.json        ← from sim.save()
├── losses.json               ← from sim.save()
├── metadata.json             ← from sim.save()
├── simulator.ini          ← you provide (downloaded and adjusted below)
└── augmented_simulator.py ← you provide (downloaded and filled in below)
        """)

        st.markdown("""
        **Step 6 — Upload**
        Use the upload form below. Select `custom_tf` or `custom_torch` as the
        model type. The system validates your ZIP before uploading.
        """)

    col_tf, col_torch = st.columns(2)

    with col_tf:
        st.markdown("**TensorFlow**")
        st.download_button(
            "Download TF template",
            data=_doc_bytes("custom_model_template.py"),
            file_name="augmented_simulator.py",
            mime="application/octet-stream",
        )
        st.download_button(
            "Download TF simulator config",
            data=_doc_bytes("tf_fc.ini"),
            file_name="simulator.ini",
            mime="application/octet-stream",
        )

    with col_torch:
        st.markdown("**PyTorch**")
        st.download_button(
            "Download PyTorch template",
            data=_doc_bytes("custom_model_template_torch.py"),
            file_name="augmented_simulator.py",
            mime="application/octet-stream",
        )
        st.download_button(
            "Download PyTorch simulator config",
            data=_doc_bytes("torch_fc.ini"),
            file_name="simulator.ini",
            mime="application/octet-stream",
        )

    st.markdown("**Benchmark config (required for training — same for both frameworks)**")
    st.download_button(
        "Download benchmark config",
        data=_BENCH_CONFIG_PATH.read_bytes(),
        file_name="benchmark.ini",
        mime="application/octet-stream",
    )
    st.caption(
        "Train against this exact file — point `BENCH_CONFIG` in the template at it. "
        "Do **not** edit or rename it, and do **not** include it in your ZIP. It is the "
        "fixed config the scoreboard evaluates every model against; a mismatch produces "
        "invalid scores."
    )

    st.subheader("Upload Your Model")

    repo_name = st.text_input("Repository name", placeholder="my-model-v1")
    if repo_name:
        st.caption(f"Will be uploaded as: `lips-poc/{repo_name}`")

    model_type = st.selectbox(
        "Model type",
        ["tf_fc", "tf_leapnet", "torch_fc", "custom_tf", "custom_torch"],
    )

    if model_type in ("custom_tf", "custom_torch"):
        framework = "TensorFlow" if model_type == "custom_tf" else "PyTorch"
        st.info(
            f"Custom {framework} model. Your ZIP must contain "
            "`augmented_simulator.py` and a `.ini` config file. "
            "See the instructions above."
        )

    description = st.text_area("Description (optional)")

    zip_bytes = None
    if model_type != "dc_approximation":
        uploaded = st.file_uploader("Model ZIP", type=["zip"])
        if uploaded:
            zip_bytes = uploaded.read()

    errors = validate_upload_inputs(model_type, repo_name, zip_bytes)
    for err in errors:
        st.error(err)

    if st.button("Confirm Upload", type="primary", disabled=bool(errors)):
        with st.spinner("Uploading and validating…"):
            try:
                repo_id = upload_model(repo_name, model_type, zip_bytes, description)
                st.success(f"Uploaded successfully as `{repo_id}`.")
                st.cache_data.clear()
                st.session_state.selected_model = repo_id
                st.rerun()
            except Exception as e:
                st.error(f"Upload failed: {e}")

# ── Scoreboard ────────────────────────────────────────────────────────────────

with tab_scoreboard:
    st.subheader("Scoreboard")

    sel_ds = st.session_state.get("selected_dataset")
    sel_m  = (st.session_state.get("selected_model") or "").removesuffix("_DEFAULT") or None

    col1, col2 = st.columns(2)
    col1.metric("Selected Dataset", sel_ds or "None")
    col2.metric("Selected Model",   sel_m  or "None")

    if st.button("Evaluate", type="primary"):
        if not sel_ds:
            st.error("Please select a dataset in the Data Hub tab first.")
        elif not sel_m:
            st.error("Please select a model in the Model Hub tab first.")
        else:
            ds_key = sel_ds if sel_ds in DATASET_REGISTRY else sel_ds.split("/")[-1]
            if ds_key not in DATASET_REGISTRY:
                st.error(f"'{ds_key}' not found in dataset_registry.json.")
            else:
                with st.spinner("Downloading model and running evaluation — this may take a few minutes…"):
                    try:
                        results = run_evaluation(
                            dataset_info=DATASET_REGISTRY[ds_key],
                            model_repo_id=sel_m,
                        )
                        scores = extract_scores(results)
                    except Exception as e:
                        import traceback
                        st.error(f"Evaluation failed: {e}\n\n```\n{traceback.format_exc()}\n```")
                        st.stop()

                new_row = {
                    "Model":     sel_m.split("/")[-1],
                    "Dataset":   ds_key,
                    "Benchmark": DATASET_REGISTRY[ds_key]["benchmark_name"],
                    **scores,
                    "Timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                df = _load_scoreboard()
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                _save_scoreboard(df)
                st.success(f"Done — {new_row['Model']} on {ds_key} added to scoreboard.")
                st.rerun()

    st.dataframe(
        _load_scoreboard(),
        use_container_width=True,
        hide_index=True,
    )
