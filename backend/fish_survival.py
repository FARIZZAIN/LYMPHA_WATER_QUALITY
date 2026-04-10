"""
fish_survival.py  —  Fuzzy threshold survival scorer
=====================================================
No ML training needed. Each parameter gets a sigmoid
membership function around its biological threshold.

Why sigmoid instead of hard cutoff:
  Hard cutoff: Goldfish dies instantly at pH 8.01 (wrong)
  Sigmoid:     Goldfish is at 90% survival at pH 7.9,
               50% at pH 8.0, 10% at pH 8.1 (realistic)

Final species score = min across all parameters
(Liebig's law of the minimum — the most limiting
factor determines survival, not the average).
"""

from math import exp

# ─────────────────────────────────────────────────────
# SPECIES SURVIVAL THRESHOLDS
# Source: aquaculture literature / limnology standards
# ─────────────────────────────────────────────────────
SPECIES_CONDITIONS = {
    "Goldfish": {
        "pH":          (6.5, 8.0),
        "Temperature": (10.0, 24.0),
        "Turbidity":   (0.0, 20.0),
        "DO":          (5.0, 15.0),
        "Conductivity": (100.0, 500.0),
    },
    "Tilapia": {
        "pH":          (6.0, 9.0),
        "Temperature": (22.0, 35.0),
        "Turbidity":   (0.0, 30.0),
        "DO":          (3.0, 15.0),
        "Conductivity": (100.0, 1000.0),
    },
    "Guppy": {
        "pH":          (6.5, 8.5),
        "Temperature": (18.0, 28.0),
        "Turbidity":   (0.0, 15.0),
        "DO":          (5.0, 15.0),
        "Conductivity": (100.0, 600.0),
    },
    "Mrigal": {
        "pH":          (6.5, 8.5),
        "Temperature": (18.0, 32.0),
        "Turbidity":   (0.0, 25.0),
        "DO":          (4.0, 15.0),
        "Conductivity": (100.0, 700.0),
    },
    "Silver Carp": {
        "pH":          (6.5, 8.5),
        "Temperature": (18.0, 32.0),
        "Turbidity":   (0.0, 30.0),
        "DO":          (3.0, 15.0),
        "Conductivity": (100.0, 800.0),
    },
    "Koi Carp": {
        "pH":          (6.5, 8.0),
        "Temperature": (10.0, 28.0),
        "Turbidity":   (0.0, 20.0),
        "DO":          (5.0, 15.0),
        "Conductivity": (100.0, 1200.0),
    },
}

# Half-width of the transition zone around each threshold.
# Score = 0.9 at (threshold + tolerance) going inward,
# Score = 0.5 at the threshold itself,
# Score = 0.1 at (threshold - tolerance) going outward.
PARAM_TOLERANCE = {
    "pH":          0.3,    # ±0.3 pH units
    "Temperature": 2.0,    # ±2 °C
    "Turbidity":   3.0,    # ±3 NTU
    "DO":          0.5,    # ±0.5 mg/L
    "Conductivity": 50.0,  # ±50 µS/cm
}


def _sigmoid(value: float, center: float, steepness: float) -> float:
    """Standard logistic sigmoid."""
    try:
        return 1.0 / (1.0 + exp(-steepness * (value - center)))
    except OverflowError:
        return 0.0 if steepness * (value - center) < 0 else 1.0


def _param_score(value: float, lo: float, hi: float, param: str) -> float:
    """
    Returns 0-1 survival probability for a single parameter.
    Uses steepness derived from PARAM_TOLERANCE so the
    transition zone is biologically calibrated per parameter.
    """
    tol = PARAM_TOLERANCE.get(param, 1.0)
    # k such that score = 0.9 at threshold ± tolerance
    k = 2.197 / tol  # ln(9) / tolerance

    score = 1.0
    if lo is not None:
        score *= _sigmoid(value, lo, k)    # drops off below lo
    if hi is not None:
        score *= _sigmoid(value, hi, -k)   # drops off above hi
    return score


def predict_fish_survival(water_params: dict) -> list:
    """
    Score each species against the provided water parameters.
    Missing parameters are skipped (not penalised).

    Returns list of dicts sorted highest probability first:
      {
        species:         str,
        probability:     float (0-100),
        limiting_factor: str | None   — parameter with the lowest score,
        breakdown:       {param: score_0_to_100, ...}
      }
    """
    results = []
    for species, conditions in SPECIES_CONDITIONS.items():
        param_scores = {}
        for param, (lo, hi) in conditions.items():
            val = water_params.get(param)
            if val is not None:
                param_scores[param] = _param_score(val, lo, hi, param)

        if not param_scores:
            survival = 0.0
            limiting = None
        else:
            # Liebig's law: survival limited by worst parameter
            limiting = min(param_scores, key=param_scores.get)
            survival = param_scores[limiting]

        results.append({
            "species":         species,
            "probability":     round(survival * 100, 1),
            "limiting_factor": limiting,
            "breakdown":       {p: round(s * 100, 1) for p, s in param_scores.items()},
        })

    results.sort(key=lambda x: x["probability"], reverse=True)
    return results


# ─────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        {"label": "Warm tropical water",
         "params": {"pH": 7.5, "Temperature": 28.0, "Turbidity": 5.0,
                    "DO": 7.0, "Conductivity": 400.0}},
        {"label": "Cold clear pond",
         "params": {"pH": 7.2, "Temperature": 15.0, "Turbidity": 3.0,
                    "DO": 8.5, "Conductivity": 250.0}},
        {"label": "Polluted / hostile",
         "params": {"pH": 5.0, "Temperature": 38.0, "Turbidity": 35.0,
                    "DO": 1.5, "Conductivity": 1300.0}},
        {"label": "Near Tilapia boundary (pH just above limit)",
         "params": {"pH": 9.1, "Temperature": 28.0, "Turbidity": 5.0,
                    "DO": 6.0, "Conductivity": 400.0}},
    ]
    for case in test_cases:
        print(f"\n{case['label']}")
        for r in predict_fish_survival(case["params"]):
            bar = "█" * int(r["probability"] / 5)
            lim = f"  ← limited by {r['limiting_factor']}" if r["limiting_factor"] and r["probability"] < 80 else ""
            print(f"  {r['species']:15s} {r['probability']:5.1f}%  {bar}{lim}")
