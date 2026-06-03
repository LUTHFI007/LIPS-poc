# IMPORTANT: Do NOT add top-level LIPS/TF/Torch imports here.
# TensorFlow runs a full GPU/CUDA probe on import, crashing the app
# before Gradio launches. All heavy imports are deferred into the
# functions that need them and only execute when Evaluate is clicked.

import pathlib
from typing import Optional

from huggingface_hub import snapshot_download, ModelCard

# Fallback: infer model type from repo name if no HF tag is set.
# Keys are checked in order — more specific keys must come first.
NAME_FALLBACK = {
    "tf_leapnet": "tf_leapnet",
    "tf_fc":      "tf_fc",
    "torch_fc":   "torch_fc",
    "dc":         "dc_approximation",
}


def _get_model_class_map() -> dict:
    from lips.augmented_simulators.tensorflow_simulator import TensorflowSimulator
    from lips.augmented_simulators.tensorflow_models.powergrid.fully_connected import TfFullyConnectedPowerGrid
    from lips.augmented_simulators.tensorflow_models.powergrid.leap_net import LeapNet
    from lips.augmented_simulators.torch_simulator import TorchSimulator
    from lips.augmented_simulators.torch_models.fully_connected import TorchFullyConnected
    return {
        "tf_fc":            (TensorflowSimulator, TfFullyConnectedPowerGrid),
        "tf_leapnet":       (TensorflowSimulator, LeapNet),
        "torch_fc":         (TorchSimulator,      TorchFullyConnected),
        "dc_approximation": None,
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


def _download_model(repo_id: str) -> str:
    return snapshot_download(repo_id=repo_id)


def _load_simulator(repo_id: str, local_path: str, attr_x, attr_y, attr_tau):
    model_type = _resolve_model_type(repo_id)
    name = pathlib.Path(local_path).name

    if model_type == "dc_approximation":
        from lips.physical_simulator.dcApproximationAS import DCApproximationAS
        return DCApproximationAS(name="dc_approximation")

    model_class_map = _get_model_class_map()
    entry = model_class_map.get(model_type)
    if entry is None:
        raise ValueError(f"Unknown model type: '{model_type}'")

    simulator_cls, model_cls = entry
    sim = simulator_cls(
        name=name,
        model=model_cls,
        scaler=None,
        attr_x=attr_x,
        attr_y=attr_y,
        attr_tau=attr_tau,
    )
    sim.restore(path=local_path)
    return sim


def run_evaluation(
    dataset_info: dict,
    model_repo_id: str,
    eval_splits: tuple = ("test", "test_ood_topo"),
) -> dict:
    from lips.benchmark.powergridBenchmark import PowerGridBenchmark

    benchmark = PowerGridBenchmark(
        benchmark_name=dataset_info["benchmark_name"],
        benchmark_path=dataset_info["dataset_root"],
        load_data_set=True,
        config_path=dataset_info["config_path"],
    )
    attr_x   = benchmark.config.get_option("attr_x")
    attr_y   = benchmark.config.get_option("attr_y")
    attr_tau = benchmark.config.get_option("attr_tau")

    local_path = _download_model(model_repo_id)
    simulator  = _load_simulator(model_repo_id, local_path, attr_x, attr_y, attr_tau)

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
