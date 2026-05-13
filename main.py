import gradio as gr
import pandas as pd

from lips_poc.data_hub import search_datasets
from lips_poc.model_hub import search_models
from lips_poc.scoreboard import evaluate_model, load_scoreboard, save_result

_EMPTY_DATASETS = pd.DataFrame(columns=["Dataset ID", "Author", "Last Modified", "URL"])
_EMPTY_MODELS = pd.DataFrame(columns=["Model ID", "Author", "Last Modified", "URL"])

_DS_DATATYPES = ["str", "str", "str", "html"]
_M_DATATYPES = ["str", "str", "str", "html"]


def _linkify(url: str) -> str:
    return f'<a href="{url}" target="_blank">{url}</a>'


def _rows_to_datasets_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["URL"] = df["URL"].apply(_linkify)
    return df


def _rows_to_models_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["URL"] = df["URL"].apply(_linkify)
    return df


# ---------------------------------------------------------------------------
# Data Hub handlers
# ---------------------------------------------------------------------------

def on_search_datasets(keyword: str) -> pd.DataFrame:
    kw = keyword.strip() if keyword.strip() else "powergrid"
    try:
        rows = search_datasets(kw)
        return _rows_to_datasets_df(rows) if rows else _EMPTY_DATASETS
    except Exception:
        return _EMPTY_DATASETS


def on_refresh_datasets() -> pd.DataFrame:
    return on_search_datasets("powergrid")


# ---------------------------------------------------------------------------
# Model Hub handlers
# ---------------------------------------------------------------------------

def on_search_models(keyword: str) -> pd.DataFrame:
    kw = keyword.strip() if keyword.strip() else "powergrid"
    try:
        rows = search_models(kw)
        return _rows_to_models_df(rows) if rows else _EMPTY_MODELS
    except Exception:
        return _EMPTY_MODELS


def on_refresh_models() -> pd.DataFrame:
    return on_search_models("powergrid")


# ---------------------------------------------------------------------------
# Scoreboard handlers
# ---------------------------------------------------------------------------

def on_evaluate(
    model_file,
    dataset_file,
    username: str,
    model_name: str,
) -> tuple[str, pd.DataFrame]:
    if not model_file or not dataset_file:
        return "Please upload both model and dataset files.", _scoreboard_df()
    if not username.strip() or not model_name.strip():
        return "Please enter username and model name.", _scoreboard_df()

    model_path = model_file.name if hasattr(model_file, "name") else model_file
    dataset_path = dataset_file.name if hasattr(dataset_file, "name") else dataset_file

    try:
        result = evaluate_model(model_path, dataset_path, username.strip(), model_name.strip())
        save_result(result)
        status = (
            f"Evaluation complete — final score: {result['final_score']:.4f}  "
            f"(ML: {result['ml_score']:.4f}, Physics: {result['physics_score']:.4f})"
        )
        return status, _scoreboard_df()
    except Exception as e:
        return f"Error: {e}", _scoreboard_df()


def _scoreboard_df() -> pd.DataFrame:
    df = load_scoreboard()
    if df.empty:
        return df
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df.rename(columns={
        "username": "Username",
        "model_name": "Model Name",
        "ml_score": "ML Score",
        "physics_score": "Physics Score",
        "final_score": "Final Score",
        "timestamp": "Timestamp",
    })


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="LIPS Power Grid Benchmark") as app:
        gr.Markdown("# LIPS Power Grid Benchmark POC")

        with gr.Tabs():

            # ── Tab 1: Data Hub ───────────────────────────────────────────
            with gr.Tab("Data Hub"):
                gr.Markdown("### Search power-grid datasets on HuggingFace")
                with gr.Row():
                    ds_search_box = gr.Textbox(
                        label="Search keyword",
                        placeholder="e.g. powergrid, power grid, load flow …",
                        scale=4,
                    )
                    ds_search_btn = gr.Button("Search", variant="primary", scale=1)
                    ds_refresh_btn = gr.Button("Refresh", variant="secondary", scale=1)
                datasets_table = gr.DataFrame(
                    label="Datasets",
                    value=_EMPTY_DATASETS,
                    datatype=_DS_DATATYPES,
                    interactive=False,
                    wrap=True,
                )

                app.load(on_refresh_datasets, outputs=datasets_table)
                ds_search_btn.click(on_search_datasets, inputs=ds_search_box, outputs=datasets_table)
                ds_refresh_btn.click(on_refresh_datasets, outputs=datasets_table)

            # ── Tab 2: Model Hub ──────────────────────────────────────────
            with gr.Tab("Model Hub"):
                gr.Markdown("### Search power-grid models on HuggingFace")
                with gr.Row():
                    m_search_box = gr.Textbox(
                        label="Search keyword",
                        placeholder="e.g. powergrid, power grid, power flow …",
                        scale=4,
                    )
                    m_search_btn = gr.Button("Search", variant="primary", scale=1)
                    m_refresh_btn = gr.Button("Refresh", variant="secondary", scale=1)
                models_table = gr.DataFrame(
                    label="Models",
                    value=_EMPTY_MODELS,
                    datatype=_M_DATATYPES,
                    interactive=False,
                    wrap=True,
                )

                app.load(on_refresh_models, outputs=models_table)
                m_search_btn.click(on_search_models, inputs=m_search_box, outputs=models_table)
                m_refresh_btn.click(on_refresh_models, outputs=models_table)

            # ── Tab 3: Scoreboard ─────────────────────────────────────────
            with gr.Tab("Scoreboard"):
                gr.Markdown("### Submit a model for evaluation")
                with gr.Row():
                    eval_model_file = gr.File(label="Model file / directory")
                    eval_dataset_file = gr.File(label="Dataset file / directory")
                with gr.Row():
                    eval_username = gr.Textbox(label="Username")
                    eval_model_name = gr.Textbox(label="Model name")
                eval_btn = gr.Button("Submit & Evaluate", variant="primary")
                eval_status = gr.Textbox(label="Result", interactive=False)

                gr.Markdown("---")
                gr.Markdown("### Leaderboard")
                scoreboard_table = gr.DataFrame(label="Rankings")

                app.load(_scoreboard_df, outputs=scoreboard_table)
                eval_btn.click(
                    on_evaluate,
                    inputs=[eval_model_file, eval_dataset_file, eval_username, eval_model_name],
                    outputs=[eval_status, scoreboard_table],
                )

    return app


if __name__ == "__main__":
    build_app().launch()
