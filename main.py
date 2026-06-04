import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from lips_poc.data_hub import search_datasets
from lips_poc.model_hub import search_models
from evaluation_runner import run_evaluation, extract_scores

_ROOT = Path(__file__).parent

_SB_COLS = [
    "Model", "Dataset", "Benchmark",
    "MSE", "MAE", "MAPE_90",
    "MSE (ood)", "MAE (ood)", "MAPE_90 (ood)",
    "Physics Viol. %", "Timestamp",
]

SCOREBOARD_FILE = _ROOT / "scoreboard.json"

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
        return pd.DataFrame(rows, columns=_SB_COLS)
    except FileNotFoundError:
        return pd.DataFrame(columns=_SB_COLS)


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
