# IMPORTANT: Do NOT add top-level LIPS/TF/Torch imports here.
# TensorFlow runs a full GPU/CUDA probe on import, crashing the app
# before Streamlit launches. All heavy imports are deferred into the
# functions that need them and only execute when Evaluate is clicked.

import json
import pathlib
from typing import Optional

from huggingface_hub import snapshot_download, ModelCard, HfApi

_MODELS_DIR = pathlib.Path(__file__).parent / "models"

# Versioned-dataset snapshots downloaded from lakeFS, keyed by commit id —
# the dataset counterpart of the models/ cache below.
_DATASETS_CACHE = pathlib.Path(__file__).parent / "datasets" / "cache"

# Cached AC-solver reference time per benchmark (for the speed-up sub-score).
# Computing it runs the physical solver (~20s), so it is memoized to disk and
# reused across evaluations of the same benchmark.
_SPEED_CACHE = pathlib.Path(__file__).parent / "speed_reference_cache.json"

# Fallback: infer model type from repo name if no HF tag is set.
# Keys are checked in order — more specific keys must come first.
NAME_FALLBACK = {
    "tf_leapnet": "tf_leapnet",
    "tf_fc":      "tf_fc",
    "torch_fc":   "torch_fc",
    "dc":         "dc_approximation",
}


def _resolve_model_type(repo_id: str) -> str:
    try:
        card = ModelCard.load(repo_id)
        for tag in (card.data.tags or []):
            if tag.startswith("lips_model_type:"):
                return tag.split(":", 1)[1]
    except Exception:
        pass
    name_lower = repo_id.lower()
    for keyword, model_type in NAME_FALLBACK.items():
        if keyword in name_lower:
            return model_type
    raise ValueError(
        f"Cannot determine model type for '{repo_id}'. "
        "Add a 'lips_model_type:<type>' tag to its HF model card."
    )


def _flatten_single_subdir(lips_dir: pathlib.Path) -> None:
    """If the model payload was uploaded inside a single nested subfolder
    (common when a folder is zipped as-is, e.g. model_1.zip -> model_1/...),
    move those files up to lips_dir so LIPS restore() and the loader find them
    at the top level. No-op if a config (.ini) is already at the top level."""
    import shutil

    if any(lips_dir.glob("*.ini")):
        return
    subdirs = [d for d in lips_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    nested = [d for d in subdirs if any(d.glob("*.ini"))]
    if len(nested) != 1:
        return
    src_dir = nested[0]
    for item in src_dir.iterdir():
        dest = lips_dir / item.name
        if not dest.exists():
            shutil.move(str(item), str(dest))
    try:
        src_dir.rmdir()
    except OSError:
        pass


def _download_model(repo_id: str, revision: "str | None" = None) -> tuple[str, str, str]:
    """
    Download a specific model revision into models/{repo_slug}__{sha}_DEFAULT/.

    The exact HF commit SHA is resolved first — even when revision is None
    (i.e. latest) — and used as the cache key, so a new commit never reuses a
    stale cache and every evaluation is tied to one concrete commit (the link
    to the MLflow tracking plane).

    LIPS appends _DEFAULT to the name it receives, so the folder name minus the
    _DEFAULT suffix becomes the LIPS name; that contract is kept intact.
    Returns (restore_base_path, model_files_path, hf_revision_sha).
    """
    sha = HfApi().model_info(repo_id=repo_id, revision=revision).sha

    repo_slug = repo_id.replace("/", "--")
    lips_dir  = _MODELS_DIR / f"{repo_slug}__{sha[:8]}_DEFAULT"

    if not lips_dir.exists():
        lips_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo_id, revision=sha, local_dir=str(lips_dir))

    # Flatten a single nested upload folder so the files sit directly in lips_dir.
    _flatten_single_subdir(lips_dir)

    # LIPS looks for weights.h5 but HF stores it as model.weights.h5
    src = lips_dir / "model.weights.h5"
    dst = lips_dir / "weights.h5"
    if src.exists() and not dst.exists():
        dst.symlink_to(src)

    return str(_MODELS_DIR), str(lips_dir), sha


def _download_dataset(lakefs_repo: str, commit_id: str, benchmark_name: str) -> str:
    """Download the FULL dataset snapshot (all four splits) at a lakeFS commit
    into datasets/cache/{repo}__{commit12}/{benchmark_name}/ and return the
    benchmark_path (the parent dir — PowerGridBenchmark expects
    benchmark_path/<benchmark_name>/<splits>).

    Mirrors _download_model: the immutable commit id is the cache key, so a
    version re-run downloads nothing and a new version never reuses a stale
    cache. The snapshot lands under a temp name and is renamed into place, so
    an interrupted download can't masquerade as a complete cache."""
    root   = _DATASETS_CACHE / f"{lakefs_repo}__{commit_id[:12]}"
    target = root / benchmark_name

    if not target.exists():
        from lips_poc import lakefs_store

        tmp = root / f".{benchmark_name}.partial"
        if tmp.exists():
            import shutil
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        lakefs_store.download_snapshot(lakefs_repo, commit_id, str(tmp))
        tmp.rename(target)

    return str(root)


def _load_custom_class(model_files_path, base_cls):
    """Load augmented_simulator.py from the model folder and return the
    class inside it that subclasses base_cls. Raises ValueError if the file or
    a matching subclass is missing."""
    import importlib.util
    import inspect

    loader_path = model_files_path / "augmented_simulator.py"
    if not loader_path.exists():
        raise ValueError(
            "Custom model type requires augmented_simulator.py in the model "
            "folder. See docs/custom_model_template.py for the required format."
        )

    spec = importlib.util.spec_from_file_location("augmented_simulator", str(loader_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if issubclass(obj, base_cls) and obj is not base_cls:
            return obj

    raise ValueError(
        f"augmented_simulator.py must define a subclass of {base_cls.__name__}."
    )


def _scaler_kwarg(model_files_path) -> dict:
    """Auto-detect normalization for custom uploads. A model trained WITH a scaler
    ships scaler_params.json; pass StandardScaler so restore() loads those params and
    predict() de-normalizes the outputs. A model trained without one ships no such
    file, so return {} and leave the simulator unscaled (scaler=None) — matching how
    it was trained. StandardScaler is the LIPS default the FC-derived templates use."""
    if (model_files_path / "scaler_params.json").exists():
        from lips.dataset.scaler import StandardScaler
        return {"scaler": StandardScaler}
    return {}


def _load_simulator(model_type: str, restore_base: str, model_files: str, dataset_info: dict):
    restore_base_path = pathlib.Path(restore_base)
    model_files_path  = pathlib.Path(model_files)

    if model_type == "dc_approximation":
        from lips.physical_simulator.dcApproximationAS import DCApproximationAS
        return DCApproximationAS(name="dc_approximation")

    # The model folder may bundle several .ini files: the benchmark config, the
    # grid/env config (e.g. l2rpn_case14_sandbox.ini) and the simulator config.
    # Pick the simulator config: skip the benchmark config, and skip any grid/env
    # config (identifiable by its "env_name" option, which model configs never
    # have). glob order is filesystem-dependent, so filtering by content — not
    # position — is what makes this deterministic.
    bench_ini_name = pathlib.Path(dataset_info["config_path"]).name

    def _is_env_config(ini_path: pathlib.Path) -> bool:
        try:
            return "env_name" in ini_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False

    sim_ini_files = [
        f for f in sorted(model_files_path.glob("*.ini"))
        if f.name != bench_ini_name and not _is_env_config(f)
    ]
    if not sim_ini_files:
        raise ValueError(f"No simulator .ini config found in {model_files_path}")
    sim_config_path = str(sim_ini_files[0])

    # LIPS appends "_DEFAULT" to name in __init__.
    # Pass the folder name without "_DEFAULT" so LIPS produces the exact folder name.
    lips_name = model_files_path.name.removesuffix("_DEFAULT")

    if model_type in ("tf_fc", "tf_leapnet"):
        import tensorflow as tf
        # tf_fc must use the GENERIC TfFullyConnected — the same class the notebook
        # trained/restored with. The powergrid subclass (TfFullyConnectedPowerGrid)
        # injects an extra topo-vector transformation into the inputs, so restoring
        # these weights into it feeds the network a different encoding than it was
        # trained on and predictions degrade badly.
        from lips.augmented_simulators.tensorflow_models import TfFullyConnected
        from lips.augmented_simulators.tensorflow_models.powergrid.leap_net import LeapNet
        # The models were trained WITH a scaler (see notebook): inputs/outputs are
        # normalized for training and de-normalized on predict. The fitted params
        # live in scaler_params.json. Without passing the scaler class here,
        # self.scaler is None and LIPS silently skips load/transform/inverse —
        # predictions stay in normalized space and metrics blow up.
        from lips.dataset.scaler import StandardScaler
        from lips.dataset.scaler.powergrid_scaler import PowerGridScaler

        base_cls   = TfFullyConnected if model_type == "tf_fc" else LeapNet
        scaler_cls = StandardScaler   if model_type == "tf_fc" else PowerGridScaler

        class _SimWrapper(base_cls):
            def _load_model(self, path):
                # On the Keras 2.8 / TF 2.8 env, weights.h5 is a native Keras 2
                # HDF5 file, so LIPS' standard load_weights restores it directly.
                tf.keras.backend.clear_session()
                super()._load_model(path)

        sim = _SimWrapper(
            name=lips_name,
            sim_config_path=sim_config_path,
            bench_config_path=dataset_info["config_path"],
            bench_config_name=dataset_info["benchmark_name"],
            scaler=scaler_cls,
            log_path=None,
        )

    elif model_type == "torch_fc":
        from lips.augmented_simulators.torch_simulator import TorchSimulator
        from lips.augmented_simulators.torch_models.fully_connected import TorchFullyConnected
        # Trained with StandardScaler (see notebook) — pass it so restore loads
        # scaler_params.json and predict de-normalizes the outputs.
        from lips.dataset.scaler import StandardScaler

        sim = TorchSimulator(
            model=TorchFullyConnected,
            sim_config_path=sim_config_path,
            name=lips_name,
            scaler=StandardScaler,
            bench_config_path=dataset_info["config_path"],
            bench_config_name=dataset_info["benchmark_name"],
            log_path=None,
        )

    elif model_type == "custom_tf":
        # The user's ZIP includes augmented_simulator.py defining a subclass
        # of TfFullyConnectedPowerGrid. Load it, find that subclass, and
        # instantiate it. restore() is called by the shared line below.
        from lips.augmented_simulators.tensorflow_models.powergrid.fully_connected import TfFullyConnectedPowerGrid

        custom_cls = _load_custom_class(model_files_path, TfFullyConnectedPowerGrid)
        custom_kwargs = dict(
            name=lips_name,
            sim_config_path=sim_config_path,
            bench_config_path=dataset_info["config_path"],
            bench_config_name=dataset_info["benchmark_name"],
            log_path=None,
        )
        custom_kwargs.update(_scaler_kwarg(model_files_path))
        sim = custom_cls(**custom_kwargs)

    elif model_type == "custom_torch":
        # The user's ZIP includes augmented_simulator.py defining a subclass
        # of TorchFullyConnected (the model). Wrap it in a TorchSimulator.
        from lips.augmented_simulators.torch_simulator import TorchSimulator
        from lips.augmented_simulators.torch_models.fully_connected import TorchFullyConnected

        custom_cls = _load_custom_class(model_files_path, TorchFullyConnected)
        sim = TorchSimulator(
            model=custom_cls,
            sim_config_path=sim_config_path,
            name=lips_name,
            bench_config_path=dataset_info["config_path"],
            bench_config_name=dataset_info["benchmark_name"],
            log_path=None,
            **_scaler_kwarg(model_files_path),
        )

    else:
        raise ValueError(f"Unknown model type: '{model_type}'")

    sim.restore(path=str(restore_base_path))
    return sim


def _read_model_config(model_files: str) -> dict:
    """Read the model's config.json (its hyperparameters) from the downloaded
    folder so they can be logged as MLflow params. Returns {} if absent."""
    import json
    cfg_path = pathlib.Path(model_files) / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        with cfg_path.open() as f:
            return json.load(f)
    except Exception:
        return {}


def run_evaluation(
    dataset_info: dict,
    model_repo_id: str,
    eval_splits: tuple = ("test", "test_ood_topo"),
    revision: "str | None" = None,
    dataset_revision: "str | None" = None,
) -> "tuple[dict, str, dict]":
    from lips.benchmark.powergridBenchmark import PowerGridBenchmark

    model_type   = _resolve_model_type(model_repo_id)
    restore_base, model_files, hf_revision = _download_model(model_repo_id, revision)
    model_config = _read_model_config(model_files)
    simulator    = _load_simulator(model_type, restore_base, model_files, dataset_info)

    # Versioned datasets: fetch the exact snapshot (lakeFS commit) the user
    # selected. Legacy datasets without a lakeFS repo (or without a selected
    # version) keep reading the fixed local dataset_root.
    lakefs_repo = dataset_info.get("lakefs_repo")
    if lakefs_repo and dataset_revision:
        benchmark_path = _download_dataset(
            lakefs_repo, dataset_revision, dataset_info["benchmark_name"]
        )
    else:
        benchmark_path = dataset_info["dataset_root"]

    benchmark = PowerGridBenchmark(
        benchmark_name=dataset_info["benchmark_name"],
        benchmark_path=benchmark_path,
        load_data_set=True,
        config_path=dataset_info["config_path"],
    )

    # _topo_vect_transformer is set only during training and never persisted.
    # The generic tf_fc model has no topo transformer, so it needs nothing here.
    if model_type == "custom_tf":
        # process_dataset(training=True) safely sets the transformer for fc models.
        simulator.process_dataset(benchmark._test_dataset, training=True)
    elif model_type == "tf_leapnet":
        # Set the transformer directly rather than via process_dataset(training=True),
        # which would also call _leap_net_model.init() and corrupt the loaded weights.
        from lips.augmented_simulators.tensorflow_models.powergrid.utils import TopoVectTransformation
        simulator._topo_vect_transformer = TopoVectTransformation(
            simulator.bench_config, simulator.params, benchmark._test_dataset
        )

    all_results = {}
    for split in eval_splits:
        res = benchmark.evaluate_simulator(
            augmented_simulator=simulator,
            dataset=split,
            eval_batch_size=128,
            shuffle=False,
        )
        all_results.update(res)

    return all_results, hf_revision, model_config


def _get_metric(split_results: dict, metric_key: str) -> dict:
    return split_results.get("ML", {}).get(metric_key, {})


def _physics_violation_pct(split_results: dict) -> Optional[float]:
    physics = split_results.get("Physics", {})
    pcts = []
    for check_val in physics.values():
        if isinstance(check_val, dict):
            if "violation_percentage" in check_val:
                pcts.append(check_val["violation_percentage"])
            else:
                for v in check_val.values():
                    if isinstance(v, dict) and "Violation_proportion" in v:
                        pcts.append(v["Violation_proportion"] * 100)
    return round(sum(pcts) / len(pcts), 2) if pcts else None


def _flatten_metric(values: dict, label: str, out: dict) -> None:
    for var, val in values.items():
        if isinstance(val, (int, float)):
            out[f"{label} ({var})"] = round(val, 4)


def _reference_solver_time(benchmark_name: str, config_path: str) -> Optional[float]:
    """AC (LightSim2Grid/Grid2Op) solver time used as the speed-up reference,
    mirroring utils/compute_score.compute_speed_up (nb_samples=1e5). Cached to
    disk per benchmark; returns None if the solver time can't be computed."""
    cache: dict = {}
    if _SPEED_CACHE.exists():
        try:
            cache = json.loads(_SPEED_CACHE.read_text())
        except Exception:
            cache = {}
    if benchmark_name in cache:
        return cache[benchmark_name]

    try:
        from lips.metrics.power_grid.compute_solver_time_grid2op import (
            compute_solver_time_grid2op,
        )
        t = compute_solver_time_grid2op(
            config_path=config_path,
            benchmark_name=benchmark_name,
            nb_samples=int(1e5),
        )
    except Exception as e:
        print(f"Reference solver time unavailable for {benchmark_name}: {e}")
        return None

    cache[benchmark_name] = t
    try:
        _SPEED_CACHE.write_text(json.dumps(cache, indent=2))
    except Exception:
        pass
    return t


def _speed_up(results: dict, benchmark_name: str, config_path: str) -> Optional[float]:
    """speed_up = reference_solver_time / model_inference_time (TIME_INF), exactly
    as the competition computes it (on the test split only)."""
    time_inf = results.get("test", {}).get("ML", {}).get("TIME_INF")
    if not time_inf:
        return None
    ref = _reference_solver_time(benchmark_name, config_path)
    if not ref:
        return None
    return ref / time_inf


def compute_global_scores(results: dict, benchmark_name: str, config_path: str) -> tuple:
    """Turn a raw evaluation into (numbers, color_params) for the scoreboard.

    numbers: the six persisted metrics — Global Score, ML/Physics (test), ML/Physics
      (ood), Speed-up. Logged to MLflow as metrics (the Evaluation Runs numbers and
      the Scoreboard's ranking key).
    color_params: per-variable traffic-light strings (g/o/r, in profile order) for
      the four blocks + the scoring version — logged as MLflow params so the
      Scoreboard can render the competition-style bulbs without re-deriving.

    Returns ({}, {}) if the benchmark has no scoring profile defined yet."""
    from lips_poc.scoring_profiles import profile_for, SCORING_VERSION
    from lips_poc.global_score import (
        build_score_inputs, compute_global_score, variable_colors,
    )

    profile = profile_for(benchmark_name)
    if profile is None:
        return {}, {}

    speed_up = _speed_up(results, benchmark_name, config_path)
    numbers = compute_global_score(*build_score_inputs(results, profile), speed_up=speed_up)
    colors = variable_colors(results, profile)

    def _enc(split: str, group: str, keys: list) -> str:
        block = colors.get(split, {}).get(group, {})
        return "".join(block.get(k, "") for k in keys)

    ml_keys = list(profile["ml"].keys())
    ph_keys = list(profile["physics"].keys())
    color_params = {
        "gs_test_ml":      _enc("test", "ML", ml_keys),
        "gs_test_phys":    _enc("test", "Physics", ph_keys),
        "gs_ood_ml":       _enc("ood", "ML", ml_keys),
        "gs_ood_phys":     _enc("ood", "Physics", ph_keys),
        "scoring_version": SCORING_VERSION,
    }
    return numbers, color_params


def extract_scores(results: dict) -> dict:
    test = results.get("test", {})
    ood  = results.get("test_ood_topo", {})
    scores: dict = {}
    _flatten_metric(_get_metric(test, "MSE_avg"),    "MSE",         scores)
    _flatten_metric(_get_metric(test, "MAE_avg"),    "MAE",         scores)
    _flatten_metric(_get_metric(test, "MAPE_90_avg"),"MAPE_90",     scores)
    _flatten_metric(_get_metric(ood,  "MSE_avg"),    "MSE_ood",     scores)
    _flatten_metric(_get_metric(ood,  "MAE_avg"),    "MAE_ood",     scores)
    _flatten_metric(_get_metric(ood,  "MAPE_90_avg"),"MAPE_90_ood", scores)
    scores["Physics Viol. %"] = _physics_violation_pct(test)
    return scores
