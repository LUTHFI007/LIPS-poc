"""
Global Score — a faithful port of the ML4PhySim power grid competition's scoring
(the starting kit's utils/compute_score.py and files/5_Scoring.ipynb).

A single headline number in [0, 100]:

    Global = 100 * ( 0.30 * test_subscore
                   + 0.30 * ood_subscore
                   + 0.40 * speed_score )

    <split>_subscore = 0.66 * ml_fraction + 0.34 * physics_fraction

Every graded quantity (a_or, CHECK_GC, ...) gets a traffic light against two
thresholds — green=2, orange=1, red=0 points. A block's *fraction* is
earned_points / (2 * N), i.e. 1.0 when every quantity in the block is green.
The speed-up ratio (xN vs. the reference solver) is mapped to [0, 1] by the
competition's Weibull curve.

This module is deliberately dependency-free (plain dicts/lists only) so it can be
unit-tested against the notebook's published example. The mapping from raw LIPS
results -> graded quantities lives in `build_score_inputs`, driven by a
per-benchmark profile (see scoring_profiles.py).
"""
import math

# ── Exact competition constants (files/5_Scoring.ipynb, cell-29) ────────────────
COEFFICIENTS   = {"test": 0.30, "test_ood": 0.30, "speed_up": 0.40}
RATIOS         = {"ml": 0.66, "physics": 0.34}
VALUE_BY_COLOR = {"g": 2, "o": 1, "r": 0}
WEIBULL_C, WEIBULL_B = 5, 1.7


# ── Speed-up curve ──────────────────────────────────────────────────────────────
def weibull(c: float, b: float, x: float) -> float:
    a = c * ((-math.log(0.9)) ** (-1.0 / b))
    return 1.0 - math.exp(-((x / a) ** b))


def speed_metric(speed_up: "float | None") -> float:
    """Speed-up ratio (xN) -> [0, 1] via the competition's Weibull curve."""
    if speed_up is None or speed_up <= 0:
        return 0.0
    return max(min(weibull(WEIBULL_C, WEIBULL_B, speed_up), 1.0), 0.0)


# ── Traffic-light grading ────────────────────────────────────────────────────────
def _color(value: float, tmin: float, tmax: float) -> str:
    """Green/orange/red for a 'lower is better' quantity. Strict inequalities,
    matching the notebook exactly (a value == tmin lands in red)."""
    if value < tmin:
        return "g"
    if tmin < value < tmax:
        return "o"
    return "r"


def _fraction(colors: list) -> float:
    """Block fraction in [0, 1]: earned points / max possible points."""
    if not colors:
        return 0.0
    earned = sum(VALUE_BY_COLOR[c] for c in colors)
    return earned / (len(colors) * max(VALUE_BY_COLOR.values()))


def _block(ml_items: list, phys_items: list) -> tuple:
    """Grade one split. *_items are lists of (value, tmin, tmax).
    Returns (ml_fraction, physics_fraction, subscore)."""
    ml_colors   = [_color(v, lo, hi) for (v, lo, hi) in ml_items]
    phys_colors = [_color(v, lo, hi) for (v, lo, hi) in phys_items]
    ml_frac   = _fraction(ml_colors)
    phys_frac = _fraction(phys_colors)
    subscore  = RATIOS["ml"] * ml_frac + RATIOS["physics"] * phys_frac
    return ml_frac, phys_frac, subscore


# ── Public: the 6 stored numbers ─────────────────────────────────────────────────
def compute_global_score(test_ml: list, test_phys: list,
                         ood_ml: list, ood_phys: list,
                         speed_up: "float | None") -> dict:
    """Return the six numbers persisted per evaluation. Category values are block
    fractions in [0, 1] (1.0 = all green); Speed-up is the raw xN ratio; Global
    Score is the 0-100 headline. Colours are NOT stored — see category_colors()."""
    ml_t, phy_t, sub_t = _block(test_ml, test_phys)
    ml_o, phy_o, sub_o = _block(ood_ml, ood_phys)
    sp = speed_metric(speed_up)

    global_score = 100.0 * (
        COEFFICIENTS["test"] * sub_t
        + COEFFICIENTS["test_ood"] * sub_o
        + COEFFICIENTS["speed_up"] * sp
    )
    return {
        "Global Score":   round(global_score, 2),
        "ML (test)":      round(ml_t, 4),
        "Physics (test)": round(phy_t, 4),
        "ML (ood)":       round(ml_o, 4),
        "Physics (ood)":  round(phy_o, 4),
        "Speed-up":       round(speed_up, 4) if speed_up else None,
    }


def variable_colors(results: dict, profile: dict) -> dict:
    """Per-variable traffic lights, exactly as the notebook assigns them — the ONLY
    place the competition uses colours. Returns
        {split: {"ML": {var: 'g'/'o'/'r'}, "Physics": {criterion: 'g'/'o'/'r'}}}
    for the expanded 'View metrics' panel. Category/overall values stay numbers."""
    out = {}
    for split, res_split in (("test", "test"), ("ood", "test_ood_topo")):
        split_res = results.get(res_split, {})
        ml_res = split_res.get("ML", {})
        ml_cols = {}
        for var, (metric_key, lo, hi) in profile["ml"].items():
            val = ml_res.get(metric_key, {})
            val = val.get(var) if isinstance(val, dict) else None
            if val is not None:
                ml_cols[var] = _color(abs(val), lo, hi)

        phys_res = split_res.get("Physics", {})
        phys_cols = {}
        for crit, (lo, hi) in profile["physics"].items():
            val = _physics_value(phys_res.get(crit))
            if val is not None:
                phys_cols[crit] = _color(val, lo, hi)

        out[split] = {"ML": ml_cols, "Physics": phys_cols}
    return out


# ── Raw LIPS results -> graded inputs (validated end-to-end in Phase 2) ───────────
def _physics_value(raw) -> "float | None":
    """Extract one violation percentage from a LIPS physics-criterion result.
    Mirrors evaluation_runner._physics_violation_pct, but per single criterion."""
    if not isinstance(raw, dict):
        return None
    if "violation_percentage" in raw:
        return raw["violation_percentage"]
    props = [v["Violation_proportion"] * 100
             for v in raw.values()
             if isinstance(v, dict) and "Violation_proportion" in v]
    return sum(props) / len(props) if props else None


def build_score_inputs(results: dict, profile: dict) -> tuple:
    """Turn a raw LIPS results dict into the four graded lists compute_global_score
    expects, using `profile` to pick each quantity's metric key and thresholds.
    Quantities the benchmark did not produce are skipped (the block normalises
    over whatever is present)."""
    def block(split):
        split_res = results.get(split, {})
        ml_res = split_res.get("ML", {})
        ml_items = []
        for var, (metric_key, lo, hi) in profile["ml"].items():
            val = ml_res.get(metric_key, {})
            val = val.get(var) if isinstance(val, dict) else None
            if val is not None:
                ml_items.append((abs(val), lo, hi))

        phys_res = split_res.get("Physics", {})
        phys_items = []
        for crit, (lo, hi) in profile["physics"].items():
            val = _physics_value(phys_res.get(crit))
            if val is not None:
                phys_items.append((val, lo, hi))
        return ml_items, phys_items

    test_ml, test_phys = block("test")
    ood_ml, ood_phys   = block("test_ood_topo")
    return test_ml, test_phys, ood_ml, ood_phys
