# IMPORTANT: Do NOT add top-level LIPS/TF/Torch imports here.
# TensorFlow runs a full GPU/CUDA probe on import, crashing the app
# before Streamlit launches. All heavy imports are deferred into the
# functions that need them and only execute when Evaluate is clicked.

import pathlib
from typing import Optional

from huggingface_hub import snapshot_download, ModelCard

_MODELS_DIR = pathlib.Path(__file__).parent / "models"

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


def _download_model(repo_id: str) -> tuple[str, str]:
    """
    Download model files into models/{repo_slug}_DEFAULT/.
    LIPS appends _DEFAULT to the name it receives, so we pass repo_slug as the
    name and LIPS constructs repo_slug_DEFAULT — which is exactly the folder.
    Returns (restore_base_path, model_files_path).
    """
    repo_slug = repo_id.replace("/", "--")
    lips_dir  = _MODELS_DIR / (repo_slug + "_DEFAULT")

    if not lips_dir.exists():
        lips_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=str(lips_dir))

    # LIPS looks for weights.h5 but HF stores it as model.weights.h5
    src = lips_dir / "model.weights.h5"
    dst = lips_dir / "weights.h5"
    if src.exists() and not dst.exists():
        dst.symlink_to(src)

    return str(_MODELS_DIR), str(lips_dir)


def _load_simulator(model_type: str, restore_base: str, model_files: str, dataset_info: dict):
    restore_base_path = pathlib.Path(restore_base)
    model_files_path  = pathlib.Path(model_files)

    if model_type == "dc_approximation":
        from lips.physical_simulator.dcApproximationAS import DCApproximationAS
        return DCApproximationAS(name="dc_approximation")

    bench_ini_name = pathlib.Path(dataset_info["config_path"]).name
    sim_ini_files  = [f for f in model_files_path.glob("*.ini") if f.name != bench_ini_name]
    if not sim_ini_files:
        raise ValueError(f"No simulator .ini config found in {model_files_path}")
    sim_config_path = str(sim_ini_files[0])

    # LIPS appends "_DEFAULT" to name in __init__.
    # Pass the folder name without "_DEFAULT" so LIPS produces the exact folder name.
    lips_name = model_files_path.name.removesuffix("_DEFAULT")

    if model_type in ("tf_fc", "tf_leapnet"):
        import re
        import h5py
        import tensorflow as tf
        from lips.augmented_simulators.tensorflow_models.powergrid.fully_connected import TfFullyConnectedPowerGrid
        from lips.augmented_simulators.tensorflow_models.powergrid.leap_net import LeapNet

        base_cls = TfFullyConnectedPowerGrid if model_type == "tf_fc" else LeapNet

        _mtype = model_type  # capture for use inside the class

        class _SimWrapper(base_cls):
            def _post_process(self, dataset, predictions):
                if _mtype != "tf_leapnet":
                    return super()._post_process(dataset, predictions)
                # LeapNet's Keras model returns a list of arrays (one per attr_y).
                # Convert to the {attr_name: array} dict the evaluator expects.
                if self.scaler is not None:
                    predictions = self.scaler.inverse_transform(predictions)
                if isinstance(predictions, list):
                    return dict(zip(self._leap_net_model.attr_y, predictions))
                return dataset.reconstruct_output(predictions)

            def _load_model(self, path):
                tf.keras.backend.clear_session()

                if _mtype != "tf_fc":
                    # LeapNet uses ProxyLeapNet.load_data — no Keras load_weights involved
                    super()._load_model(path)
                    return

                # tf_fc: Keras 3 legacy loader misreads the .weights.h5 format.
                # Load weights directly from h5py by index to bypass it.
                path = pathlib.Path(path)
                weights_file = path / "model.weights.h5"
                if not weights_file.exists():
                    weights_file = path / "weights.h5"
                if not weights_file.exists():
                    raise FileNotFoundError(f"No weights file found in {path}")

                self.build_model()

                def _layer_sort_key(name):
                    m = re.search(r'_(\d+)$', name)
                    return int(m.group(1)) if m else -1

                with h5py.File(str(weights_file), 'r') as f:
                    if 'layers' not in f:
                        raise ValueError(f"Unrecognised weights format in {weights_file}")
                    saved_keys = sorted(
                        [k for k in f['layers'] if len(f['layers'][k].get('vars', {})) > 0],
                        key=_layer_sort_key,
                    )
                    model_layers = [l for l in self._model.layers if l.weights]
                    for layer, key in zip(model_layers, saved_keys):
                        g = f['layers'][key]['vars']
                        layer.set_weights([g[str(i)][:] for i in range(len(g))])

        sim = _SimWrapper(
            name=lips_name,
            sim_config_path=sim_config_path,
            bench_config_path=dataset_info["config_path"],
            bench_config_name=dataset_info["benchmark_name"],
            log_path=None,
        )

    elif model_type == "torch_fc":
        from lips.augmented_simulators.torch_simulator import TorchSimulator
        from lips.augmented_simulators.torch_models.fully_connected import TorchFullyConnected

        sim = TorchSimulator(
            model=TorchFullyConnected,
            sim_config_path=sim_config_path,
            name=lips_name,
            bench_config_path=dataset_info["config_path"],
            bench_config_name=dataset_info["benchmark_name"],
            log_path=None,
        )

    else:
        raise ValueError(f"Unknown model type: '{model_type}'")

    sim.restore(path=str(restore_base_path))
    return sim


def run_evaluation(
    dataset_info: dict,
    model_repo_id: str,
    eval_splits: tuple = ("test", "test_ood_topo"),
) -> dict:
    from lips.benchmark.powergridBenchmark import PowerGridBenchmark

    model_type   = _resolve_model_type(model_repo_id)
    restore_base, model_files = _download_model(model_repo_id)
    simulator    = _load_simulator(model_type, restore_base, model_files, dataset_info)

    benchmark = PowerGridBenchmark(
        benchmark_name=dataset_info["benchmark_name"],
        benchmark_path=dataset_info["dataset_root"],
        load_data_set=True,
        config_path=dataset_info["config_path"],
    )

    # _topo_vect_transformer is set only during training and never persisted.
    if model_type == "tf_fc":
        # process_dataset(training=True) safely sets the transformer for fc models.
        simulator.process_dataset(benchmark._test_dataset, training=True)
    elif model_type == "tf_leapnet":
        # For LeapNet, process_dataset(training=True) also calls
        # _leap_net_model.init() which corrupts the loaded weights.
        # Set the transformer directly instead.
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

    return all_results


def _avg_metric(split_results: dict, metric_key: str) -> Optional[float]:
    values = split_results.get("ML", {}).get(metric_key, {})
    nums = [v for v in values.values() if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 4) if nums else None


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


def extract_scores(results: dict) -> dict:
    test = results.get("test", {})
    ood  = results.get("test_ood_topo", {})
    return {
        "MSE":             _avg_metric(test, "MSE_avg"),
        "MAE":             _avg_metric(test, "MAE_avg"),
        "MAPE_90":         _avg_metric(test, "MAPE_90_avg"),
        "MSE (ood)":       _avg_metric(ood,  "MSE_avg"),
        "MAE (ood)":       _avg_metric(ood,  "MAE_avg"),
        "MAPE_90 (ood)":   _avg_metric(ood,  "MAPE_90_avg"),
        "Physics Viol. %": _physics_violation_pct(test),
    }
