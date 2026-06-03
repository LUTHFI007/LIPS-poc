import json
from pathlib import Path

import gradio as gr
import pandas as pd

from lips_poc.data_hub import search_datasets
from lips_poc.model_hub import search_models

_ROOT = Path(__file__).parent

_DS_COLS = ["Select", "Dataset ID", "Last Modified", "URL"]
_M_COLS  = ["Select", "Model ID",   "Last Modified", "URL"]
_SB_COLS = [
    "Model", "Dataset", "Benchmark",
    "MSE", "MAE", "MAPE_90",
    "MSE (ood)", "MAE (ood)", "MAPE_90 (ood)",
    "Physics Viol. %", "Timestamp",
]

_EMPTY_DATASETS = pd.DataFrame(columns=_DS_COLS)
_EMPTY_MODELS   = pd.DataFrame(columns=_M_COLS)

_DS_DATATYPES = ["bool", "str", "str", "html"]
_M_DATATYPES  = ["bool", "str", "str", "html"]


def _linkify(url: str) -> str:
    return f'<a href="{url}" target="_blank">{url}</a>'


def _load_datasets() -> pd.DataFrame:
    try:
        rows = search_datasets("")
        if not rows:
            return _EMPTY_DATASETS
        df = pd.DataFrame(rows)
        df.insert(0, "Select", False)
        df["URL"] = df["URL"].apply(_linkify)
        return df
    except Exception:
        return _EMPTY_DATASETS


def _load_models() -> pd.DataFrame:
    try:
        rows = search_models("")
        if not rows:
            return _EMPTY_MODELS
        df = pd.DataFrame(rows)
        df.insert(0, "Select", False)
        df["URL"] = df["URL"].apply(_linkify)
        return df
    except Exception:
        return _EMPTY_MODELS


def _enforce_single(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    selected = df.index[df["Select"].astype(bool)].tolist()
    if len(selected) > 1:
        df = df.copy()
        df["Select"] = False
        df.loc[selected[-1], "Select"] = True
    return df


def build_app() -> gr.Blocks:
    with gr.Blocks(title="LIPS Power Grid Benchmark") as app:
        gr.Markdown("# LIPS Power Grid Benchmark POC")

        with gr.Tabs():

            with gr.Tab("Data Hub"):
                ds_table = gr.DataFrame(
                    value=_EMPTY_DATASETS,
                    datatype=_DS_DATATYPES,
                    interactive=True,
                    wrap=True,
                )
                ds_table.input(fn=_enforce_single, inputs=ds_table, outputs=ds_table)
                app.load(_load_datasets, outputs=ds_table)

            with gr.Tab("Model Hub"):
                m_table = gr.DataFrame(
                    value=_EMPTY_MODELS,
                    datatype=_M_DATATYPES,
                    interactive=True,
                    wrap=True,
                )
                m_table.input(fn=_enforce_single, inputs=m_table, outputs=m_table)
                app.load(_load_models, outputs=m_table)

            with gr.Tab("Scoreboard"):
                gr.Button("Evaluate", variant="primary")
                gr.DataFrame(
                    value=pd.DataFrame(columns=_SB_COLS),
                    interactive=False,
                    wrap=True,
                )

    return app


if __name__ == "__main__":
    build_app().queue().launch()
