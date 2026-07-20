"""
MLflow tracking wrapper — the ONLY module that talks to MLflow.

Keeping every MLflow call here means the rest of the app stays clean and the
tracking server can move by setting one env var (MLFLOW_TRACKING_URI), with no
code edits elsewhere.

Design rules:
- The backend store (database) holds queryable run metadata: scores, params,
  tags. HF still holds the model files. The link between the two is the
  `hf_revision` param (the exact HF commit SHA).
- Tracking is OBSERVABILITY: a failure to log must never break an evaluation.
  Every public function swallows tracking errors and warns instead of raising.
- `import mlflow` is deferred into the functions that need it so importing this
  module (e.g. at Streamlit startup) stays cheap.

Tracking URI:
    MLFLOW_TRACKING_URI env var  ->  the MLflow tracking Space (Neon-backed).
    It is REQUIRED — there is no local sqlite fallback.
"""

import json
import logging
import os
import re
from datetime import datetime

_LOG = logging.getLogger(__name__)

# MLflow metric/param keys may only contain a limited character set; score keys
# like "ML (test)" or "Physics Viol. %" are not valid. Map anything to a safe
# name: collapse every run of non-alphanumerics to a single underscore.
_SAFE_KEY_RE = re.compile(r"[^0-9A-Za-z]+")


def tracking_uri() -> str:
    """The MLflow tracking URI from MLFLOW_TRACKING_URI (the tracking Space,
    Neon-backed). Required — raises if unset so misconfiguration is visible.
    Callers swallow it: log_evaluation warns, fetch_leaderboard returns []."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set — point it at the MLflow tracking Space."
        )
    return uri


def experiment_for(benchmark: str) -> str:
    """Experiment name for a benchmark. One experiment per benchmark keeps
    comparisons apples-to-apples (scores across benchmarks are not comparable)."""
    return f"powergrid-{benchmark.lower()}"


def make_run_name(model_name: str, version: str, author: str) -> str:
    """Human-readable run label, e.g. 'model-x@v1 (userB)'."""
    label = f"{model_name}@{version}" if version else model_name
    return f"{label} ({author})" if author else label


def sanitize_metric_key(key: str) -> str:
    """Convert an arbitrary score key into a valid MLflow metric name.
    'ML (test)' -> 'ML_test', 'Physics Viol. %' -> 'Physics_Viol'."""
    return _SAFE_KEY_RE.sub("_", key).strip("_") or "metric"


def flatten_config(cfg: dict, prefix: str = "cfg") -> dict:
    """Flatten a model config.json (its hyperparameters) into dotted MLflow
    params, e.g. {'cfg.activation': 'relu', 'cfg.layers': '[300, 300, 300, 300]',
    'cfg.optimizer.params.lr': 0.0003}. Nested dicts recurse; lists are
    JSON-stringified; None values are dropped later by log_evaluation."""
    out: dict = {}

    def _walk(obj, key):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{key}.{k}")
        elif isinstance(obj, (list, tuple)):
            out[key] = json.dumps(obj)
        else:
            out[key] = obj

    if cfg:
        _walk(cfg, prefix)
    return out


def log_evaluation(
    experiment: str,
    run_name: str,
    params: dict,
    metrics: dict,
    tags: "dict | None" = None,
) -> "str | None":
    """Record one evaluation as an MLflow run. Returns the run_id, or None if
    tracking failed (logging never raises — it must not break an evaluation).

    - params: string-valued inputs (None values are skipped).
    - metrics: numeric scores; keys are sanitized to valid MLflow names and
      non-numeric / None values are skipped.
    - tags: optional string tags (None values skipped).
    """
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri())
        mlflow.set_experiment(experiment)

        clean_params = {k: str(v) for k, v in params.items() if v is not None}
        clean_metrics = {
            sanitize_metric_key(k): float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        clean_tags = {k: str(v) for k, v in (tags or {}).items() if v is not None}

        with mlflow.start_run(run_name=run_name) as run:
            if clean_params:
                mlflow.log_params(clean_params)
            if clean_metrics:
                mlflow.log_metrics(clean_metrics)
            if clean_tags:
                mlflow.set_tags(clean_tags)
            return run.info.run_id

    except Exception as exc:  # tracking must never break evaluation
        _LOG.warning("MLflow logging failed (evaluation unaffected): %s", exc)
        return None


def fetch_leaderboard(benchmark: "str | None" = None) -> list:
    """Read scoreboard rows from the MLflow tracking store (the system of
    record). With `benchmark`, only that experiment; otherwise every
    `powergrid-*` experiment. Returns a list of flat dicts — identity columns
    first, then metrics, then revision/timestamp. Returns [] on any failure so
    callers can fall back to a local source."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri())
        client = MlflowClient()

        if benchmark:
            exp = client.get_experiment_by_name(experiment_for(benchmark))
            exps = [exp] if exp else []
        else:
            exps = [
                e for e in client.search_experiments()
                if e.name.startswith("powergrid-")
            ]

        rows = []
        for exp in exps:
            bench = exp.name[len("powergrid-"):]
            for r in client.search_runs([exp.experiment_id]):
                p, m = r.data.params, r.data.metrics
                repo = p.get("hf_repo_id", "")
                ts = (
                    datetime.fromtimestamp(r.info.start_time / 1000)
                    .isoformat(timespec="seconds")
                    if r.info.start_time else ""
                )
                rows.append({
                    "Model":     repo.split("/")[-1] if repo else r.data.tags.get("mlflow.runName", ""),
                    "Author":    p.get("author", ""),
                    "Version":   p.get("version", ""),
                    "Benchmark": bench,
                    "Dataset Version": p.get("ds_version", ""),
                    **m,
                    "Revision":  (p.get("hf_revision", "") or "")[:8],
                    "Timestamp": ts,
                    # Per-variable bulb colours (g/o/r strings, profile order) + the
                    # scoring version — logged as params, used by the Scoreboard to
                    # render the competition-style traffic lights. Absent on legacy
                    # runs (predating global scoring), which the Scoreboard skips.
                    "gs_test_ml":      p.get("gs_test_ml", ""),
                    "gs_test_phys":    p.get("gs_test_phys", ""),
                    "gs_ood_ml":       p.get("gs_ood_ml", ""),
                    "gs_ood_phys":     p.get("gs_ood_phys", ""),
                    "scoring_version": p.get("scoring_version", ""),
                    # Internal — used by the admin delete action, not for display.
                    "run_id":    r.info.run_id,
                })
        return rows
    except Exception as exc:
        _LOG.warning("MLflow leaderboard fetch failed: %s", exc)
        return []


def hard_delete_runs(run_ids: list) -> int:
    """PERMANENTLY delete runs and all their metrics/params/tags rows from the
    MLflow backend store, and return how many were deleted.

    Unlike the observability helpers above, this is an explicit admin action, so
    it RAISES on failure (the UI reports it) instead of swallowing errors.

    MLflow's REST API (the tracking Space) can only *soft*-delete, so a true wipe
    needs a DIRECT connection to the backend Postgres. This uses
    MLFLOW_BACKEND_STORE_URI if set, else APP_DB_URI — the same Neon database the
    MLflow tables live in. Our runs log no artifacts, so this is a clean DB-only
    delete (no artifact store cleanup needed)."""
    if not run_ids:
        return 0

    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

    backend = _hard_delete_backend_uri()
    # A direct store handle to the backend DB. default_artifact_root is required
    # by the constructor but unused here (no artifacts to remove).
    store = SqlAlchemyStore(backend, default_artifact_root="/tmp/mlartifacts")
    for run_id in run_ids:
        store._hard_delete_run(run_id)
    return len(run_ids)


def _hard_delete_backend_uri() -> str:
    """Resolve the DB to hard-delete from. It MUST be the same backend the
    scoreboard reads from, or we would delete from the wrong place (or fail to
    find the run — which is exactly what happens when it points at the auth DB).

    1) MLFLOW_BACKEND_STORE_URI if set (explicit — the tracking Space's backend).
    2) else, if MLFLOW_TRACKING_URI is itself a direct DB store (not an http
       server), it IS the backend, so use it.
    3) else raise — an http tracking server's backend is not reachable from here,
       so the operator must supply MLFLOW_BACKEND_STORE_URI. We deliberately do
       NOT fall back to APP_DB_URI (the auth DB, possibly a different database)."""
    backend = os.environ.get("MLFLOW_BACKEND_STORE_URI")
    if backend:
        return backend
    track = os.environ.get("MLFLOW_TRACKING_URI", "")
    if track and not track.startswith(("http://", "https://")):
        return track
    raise RuntimeError(
        "Hard delete needs the MLflow BACKEND store URI. Set MLFLOW_BACKEND_STORE_URI "
        "to the same Postgres your tracking Space uses (its MLFLOW_BACKEND_STORE_URI "
        "secret). MLFLOW_TRACKING_URI points at an HTTP server whose backend the app "
        "cannot reach directly."
    )
