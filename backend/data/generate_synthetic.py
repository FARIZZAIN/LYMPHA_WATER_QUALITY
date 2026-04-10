"""
generate_synthetic.py  —  Synthetic Water Quality Dataset Generator
====================================================================
Generates a multi-station, multi-variable time series with realistic
cross-node dependencies (temperature → conductivity → turbidity) and
random spike events.

Usage:
    python generate_synthetic.py                          # seed 42, default output
    python generate_synthetic.py --seed 999               # out-of-sample demo set
    python generate_synthetic.py --seed 999 --steps 20000 --out synthetic_demo.csv

Output columns (12 nodes, 3 stations × 4 variables):
    Temp_1, pH_1, Cond_1, Turb_1,
    Temp_2, pH_2, Cond_2, Turb_2,
    Temp_3, pH_3, Cond_3, Turb_3
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate(seed: int, n_steps: int, out_path: Path):
    PERIOD = 144
    stations = 3

    variables = []
    for s in range(1, stations + 1):
        variables += [f"Temp_{s}", f"pH_{s}", f"Cond_{s}", f"Turb_{s}"]

    n_vars = len(variables)
    print(f"[generate] seed={seed}  steps={n_steps}  nodes={n_vars}")

    np.random.seed(seed)

    X = np.zeros((n_steps, n_vars))
    X[:6] = np.random.normal(0, 0.5, (6, n_vars))

    for t in range(6, n_steps):
        noise = np.random.normal(0, 0.08, n_vars)

        for s in range(stations):
            iT  = s * 4
            iPH = s * 4 + 1
            iC  = s * 4 + 2
            iTu = s * 4 + 3

            X[t, iT] = (
                0.4 * X[t-1, iT] +
                0.2 * X[t-3, iT] +
                0.2 * np.sin(2 * np.pi * t / PERIOD) +
                noise[iT]
            )
            X[t, iPH] = (
                0.3 * X[t-1, iPH] +
                0.25 * np.tanh(X[t-2, iC]) +
                0.2 * np.sin(X[t-3, iT]) +
                noise[iPH]
            )
            X[t, iC] = (
                0.4 * X[t-1, iC] +
                0.35 * X[t-2, iT] +
                0.2 * X[t-3, iTu] +
                noise[iC]
            )
            X[t, iTu] = (
                0.3 * X[t-1, iTu] +
                0.3 * X[t-2, iC] +
                0.25 * X[t-3, iT] +
                0.2 * np.sin(2 * np.pi * t / PERIOD) +
                noise[iTu]
            )

        # Strong cross-station turbidity flow (upstream → downstream)
        for s in range(1, stations):
            upstream = (s - 1) * 4 + 3
            current  = s * 4 + 3
            X[t, current] += 0.5 * X[t-2, upstream]

        # Random spike events (1% chance per step)
        if np.random.rand() < 0.01:
            idx = np.random.randint(0, n_vars)
            X[t:t+3, idx] += np.random.normal(2.0, 0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X, columns=variables)
    df.to_csv(out_path, index=False)
    print(f"[generate] Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic water quality dataset")
    parser.add_argument("--seed",  type=int,   default=42,      help="Random seed (use 999 for out-of-sample demo)")
    parser.add_argument("--steps", type=int,   default=50000,   help="Number of timesteps")
    parser.add_argument("--out",   type=str,   default=None,    help="Output CSV filename (relative to this script's dir)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    if args.out:
        out_path = script_dir / args.out
    else:
        out_path = script_dir / f"synthetic_seed{args.seed}.csv"

    generate(seed=args.seed, n_steps=args.steps, out_path=out_path)
