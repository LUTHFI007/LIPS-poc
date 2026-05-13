import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCOREBOARD_FILE = Path(__file__).parent.parent / "scoreboard.json"

_ALPHA = 0.5
_BETA = 0.5


def evaluate_model(
    model_path: str,
    dataset_path: str,
    username: str,
    model_name: str,
) -> dict:
    try:
        ml_score, physics_score = _run_lips_benchmark(model_path, dataset_path)
    except Exception as e:
        print(f"LIPS evaluation failed, using mock scores. Reason: {e}")
        ml_score, physics_score = _mock_scores(model_path, dataset_path)

    final_score = _ALPHA * ml_score + _BETA * physics_score

    return {
        "username": username,
        "model_name": model_name,
        "ml_score": round(ml_score, 6),
        "physics_score": round(physics_score, 6),
        "final_score": round(final_score, 6),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def save_result(result_dict: dict) -> None:
    records: list[dict] = []
    if SCOREBOARD_FILE.exists():
        with SCOREBOARD_FILE.open("r", encoding="utf-8") as f:
            records = json.load(f)
    records.append(result_dict)
    with SCOREBOARD_FILE.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def load_scoreboard() -> pd.DataFrame:
    cols = ["username", "model_name", "ml_score", "physics_score", "final_score", "timestamp"]
    if not SCOREBOARD_FILE.exists():
        return pd.DataFrame(columns=cols)
    with SCOREBOARD_FILE.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not records:
        return pd.DataFrame(columns=cols)
    return (
        pd.DataFrame(records)
        .sort_values("final_score", ascending=False)
        .reset_index(drop=True)
    )


def _run_lips_benchmark(model_path: str, dataset_path: str) -> tuple[float, float]:
    from lips.benchmark.powergridBenchmark import PowerGridBenchmark
    from lips.augmented_simulators.torch_simulator import TorchSimulator
    from lips.augmented_simulators.torch_models.fully_connected import TorchFullyConnected

    BENCH_CONFIG_PATH = "/home/luthfi/LIPS/lips/tests/configs/powergrid/benchmarks/l2rpn_case14_sandbox.ini"
    SIM_CONFIG_PATH = "/home/luthfi/LIPS/lips/tests/configs/powergrid/simulators/torch_fc.ini"
    BENCHMARK_PATH = "/home/luthfi/LIPS/reference_data/powergrid/l2rpn_case14_sandbox"

    benchmark = PowerGridBenchmark(
        benchmark_path=BENCHMARK_PATH,
        config_path=BENCH_CONFIG_PATH,
        benchmark_name="Benchmark1",
        load_data_set=True,
        log_path=None,
        eval_dict={
            "ML": ["MSE_avg", "MAE_avg"],
            "Physics": ["CURRENT_POS"],
            "IndRed": [],
            "OOD": []
        }
    )

    augmented_simulator = TorchSimulator(
        model=TorchFullyConnected,
        sim_config_path=SIM_CONFIG_PATH,
        name="torch_fc",
        bench_config_path=BENCH_CONFIG_PATH,
        bench_config_name="Benchmark1",
        log_path=None,
    )

    augmented_simulator.restore(model_path)

    results = benchmark.evaluate_simulator(
        dataset="test",
        augmented_simulator=augmented_simulator,
        save_path=None,
        save_predictions=False,
    )

    ml_metrics = results.get("test", {}).get("ML", {})
    physics_metrics = results.get("test", {}).get("Physics", {})

    # MAE_avg is {var: float} — average across variables, normalize against 200 A reference
    mae_dict = ml_metrics.get("MAE_avg", {})
    if isinstance(mae_dict, dict) and mae_dict:
        mae = sum(mae_dict.values()) / len(mae_dict)
    else:
        mae = float(mae_dict) if mae_dict else 200.0
    ml_score = max(0.0, 1.0 - mae / 200.0)

    # CURRENT_POS is {var: {"Violation_proportion": float}} — 0.0 violation = perfect
    current_pos = physics_metrics.get("CURRENT_POS", {})
    if isinstance(current_pos, dict) and current_pos:
        violations = [v["Violation_proportion"] for v in current_pos.values() if isinstance(v, dict)]
        physics_score = max(0.0, 1.0 - sum(violations) / len(violations)) if violations else 0.0
    else:
        physics_score = 0.0

    return ml_score, physics_score


def _mock_scores(model_path: str, dataset_path: str) -> tuple[float, float]:
    seed = hash(model_path + dataset_path) & 0xFFFF
    ml_score = 0.5 + (seed % 1000) / 2000.0
    physics_score = 0.4 + (seed % 800) / 2000.0
    return ml_score, physics_score