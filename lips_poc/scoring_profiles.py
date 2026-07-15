"""
Per-benchmark scoring profiles — the ONLY place that knows which quantities a
benchmark grades and against which thresholds. Adding a benchmark = adding a dict
entry here; no change to global_score.py or the app.

A profile has two keys:
  "ml":      {variable: (metric_key, threshold_great, threshold_acceptable)}
  "physics": {criterion: (threshold_great, threshold_acceptable)}

Thresholds are 'lower is better': value < great -> green, great < value < accept
-> orange, else red. Values are in the metric's own units — currents/powers as
MAPE fractions (0.02 == 2%), voltages as MAE, physics as violation percentages.

SAFETY NOTE: *Adding* a benchmark is fully additive and cannot affect stored
scores. *Editing an existing benchmark's thresholds* changes future scores only
(stored numbers are frozen), so it must be paired with a SCORING_VERSION bump and
a re-evaluation of that leaderboard to keep rankings apples-to-apples.
"""

# Bump when any existing profile's thresholds change (drift safeguard, Phase 3).
SCORING_VERSION = "gs-v1"


# The exact competition definition (files/5_Scoring.ipynb, cells 24 & 26).
# Reused by any benchmark that emits the full metric set.
COMPETITION_PROFILE = {
    "ml": {
        "a_or": ("MAPE_90_avg", 0.02, 0.05),
        "a_ex": ("MAPE_90_avg", 0.02, 0.05),
        "p_or": ("MAPE_10_avg", 0.02, 0.05),
        "p_ex": ("MAPE_10_avg", 0.02, 0.05),
        "v_or": ("MAE_avg",     0.20, 0.50),
        "v_ex": ("MAE_avg",     0.20, 0.50),
    },
    "physics": {
        "CURRENT_POS":     (1.0, 5.0),
        "VOLTAGE_POS":     (1.0, 5.0),
        "LOSS_POS":        (1.0, 5.0),
        "DISC_LINES":      (1.0, 5.0),
        "CHECK_LOSS":      (1.0, 5.0),
        "CHECK_GC":        (0.05, 0.10),
        "CHECK_LC":        (0.05, 0.10),
        "CHECK_JOULE_LAW": (1.0, 5.0),
    },
}


PROFILES = {
    # Benchmark1 emits only currents (a_or/a_ex) + CURRENT_POS today, so its
    # profile grades exactly that. The block fractions normalise over this reduced
    # set, so the Global Score is valid now — no retraining required.
    "Benchmark1": {
        "ml": {
            "a_or": ("MAPE_90_avg", 0.02, 0.05),
            "a_ex": ("MAPE_90_avg", 0.02, 0.05),
        },
        "physics": {
            "CURRENT_POS": (1.0, 5.0),
        },
    },

    # Drop-in for later: yields the exact competition Global Score with no code
    # change once that benchmark's dataset/models are wired in.
    "Benchmark_competition": COMPETITION_PROFILE,
}


def profile_for(benchmark: str) -> "dict | None":
    """Return the scoring profile for a benchmark, or None if none is defined
    (the caller then skips the Global Score and the row shows '-').

    Case-insensitive: MLflow experiments store the benchmark name lowercased
    (experiment_for -> 'powergrid-benchmark1'), so it round-trips as 'benchmark1'
    on read while the profile key is 'Benchmark1'. Match on lowercase so both the
    write side (capitalised) and the read side (lowercased) resolve the profile."""
    if not benchmark:
        return None
    for key, prof in PROFILES.items():
        if key.lower() == benchmark.lower():
            return prof
    return None
