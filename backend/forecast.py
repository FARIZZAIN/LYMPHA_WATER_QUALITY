"""
forecast.py  —  TC-GCN Inference Wrapper
=========================================
Loads the checkpoint and runs 1-hour-ahead predictions.

Architecture (forward path):
  TemporalEncoder1D → TCGCBlock (Triple-Channel GCN with RL-DQN adjacency) → ARHead

Note: the checkpoint also contains STAE and ClusteringLayer weights (used
during training for representation learning) but they are not part of the
prediction forward pass and are not loaded here.

12 nodes:  Temp_1 pH_1 Cond_1 Turb_1
           Temp_2 pH_2 Cond_2 Turb_2
           Temp_3 pH_3 Cond_3 Turb_3

Physical sensors cover Station 1 only (Temp, pH, Turbidity, TDS/Conductivity).
Stations 2 & 3 are filled with Station 1 values + small calibrated noise so the
graph model still receives a meaningful 3-station signal.

The rolling buffer holds the last WINDOW=48 timesteps. Until it is full the
endpoint returns ready=False. Once full, it runs inference on every request.
"""

from __future__ import annotations
import numpy as np
import torch
from collections import deque

from src.encoders import TemporalEncoder1D
from src.tcgc    import TCGCBlock
from src.ar_head import ARHead
from src.rl_dqn  import DQNAgent, DQNConfig, build_adj_from_pairs

# ─────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────
NODES = [
    "Temp_1", "pH_1", "Cond_1", "Turb_1",
    "Temp_2", "pH_2", "Cond_2", "Turb_2",
    "Temp_3", "pH_3", "Cond_3", "Turb_3",
]
N_NODES = 12
WINDOW  = 48
HORIZON = 6

# Inter-station noise σ (as fraction of the value).
# Small enough to be physically plausible, large enough
# to give the graph model distinct node signals.
STATION_NOISE_FRAC = 0.01   # ±1 % of the reading

# Fallback normalisation if meta is missing from checkpoint
_FALLBACK_MEAN = np.array([
    25.0, 7.2, 420.0, 3.5,   # Station 1
    25.0, 7.2, 420.0, 3.5,   # Station 2
    25.0, 7.2, 420.0, 3.5,   # Station 3
], dtype=np.float32)

_FALLBACK_STD = np.array([
    3.0, 0.5, 150.0, 2.0,
    3.0, 0.5, 150.0, 2.0,
    3.0, 0.5, 150.0, 2.0,
], dtype=np.float32)

# Physical clamp bounds per node (Temp, pH, Cond, Turb) × 3 stations
# Prevents extreme z-scores in synthetic CSV from producing unrealistic display values
_PHYS_MIN = np.array([
     0.0,  0.0,  50.0,  0.0,
     0.0,  0.0,  50.0,  0.0,
     0.0,  0.0,  50.0,  0.0,
], dtype=np.float32)
_PHYS_MAX = np.array([
    50.0, 14.0, 900.0, 50.0,
    50.0, 14.0, 900.0, 50.0,
    50.0, 14.0, 900.0, 50.0,
], dtype=np.float32)


# ─────────────────────────────────────────────────────
# FORECAST MODEL
# ─────────────────────────────────────────────────────
class ForecastModel:
    def __init__(self, ckpt_path: str, cfg):
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer: deque[np.ndarray] = deque(maxlen=WINDOW)
        # Actual values (last 60 readings per node, ~3 min at 3s interval)
        self.history: deque[np.ndarray] = deque(maxlen=60)
        # Past predictions stored HORIZON steps after they were made,
        # so they align with the actual value at that timestep.
        # Each entry: (actual_row_index, predicted_np_array shape (12,))
        self._pred_queue: deque[np.ndarray] = deque(maxlen=200)
        self._actual_queue: deque[np.ndarray] = deque(maxlen=200)
        self._step = 0   # counts how many rows have been pushed
        self._rng   = np.random.default_rng(seed=0)
        self._load(ckpt_path)

    # ── load ──────────────────────────────────────────
    def _load(self, ckpt_path: str):
        print(f"[LYMPHA] Loading forecast checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Normalisation stats
        meta = ckpt.get("meta", {})
        raw_mean = meta.get("mean", None)
        raw_std  = meta.get("std",  None)
        if raw_mean is not None and len(raw_mean) == N_NODES:
            self.mean = np.array(raw_mean, dtype=np.float32)
            self.std  = np.array(raw_std,  dtype=np.float32)
        else:
            print("[LYMPHA] meta not found in checkpoint — using fallback normalization.")
            self.mean = _FALLBACK_MEAN.copy()
            self.std  = _FALLBACK_STD.copy()

        # Build model components (only what inference needs)
        self.temp = TemporalEncoder1D(
            d_out=self.cfg.d_temporal,
            k=self.cfg.temporal_kernel,
            n_layers=self.cfg.temporal_layers,
        ).to(self.device)

        self.tcgc = TCGCBlock(
            d_in=self.cfg.d_temporal,
            d_gcn=self.cfg.d_gcn,
            d_fused=self.cfg.d_fused,
            use_rl_channel=True,
        ).to(self.device)

        self.ar = ARHead(d_in=self.cfg.d_fused).to(self.device)

        self.temp.load_state_dict(ckpt["temp"])
        self.tcgc.load_state_dict(ckpt["tcgc"])
        if "ar" in ckpt:
            self.ar.load_state_dict(ckpt["ar"])

        # Static adjacency
        self.A_static = ckpt.get("A_static", torch.eye(N_NODES, dtype=torch.float32))
        self.A_static = self.A_static.to(self.device)

        # RL agent (optional)
        self.agent = None
        if "agent" in ckpt:
            self.agent = DQNAgent(
                d_in=self.cfg.d_temporal,
                N_nodes=N_NODES,
                cfg=DQNConfig(
                    top_k=self.cfg.dqn_top_k,
                    eps_start=self.cfg.dqn_eps_end,  # greedy at inference
                    eps_end=self.cfg.dqn_eps_end,
                ),
            ).to(self.device)
            self.agent.load_state_dict(ckpt["agent"])
            self.agent.eval()

        self.temp.eval()
        self.tcgc.eval()
        self.ar.eval()
        print(f"[LYMPHA] Forecast model ready — device={self.device}, nodes={N_NODES}, window={WINDOW}")

    # ── push (all 12 nodes, e.g. from dataset replay) ────
    def push_row(self, row: "np.ndarray"):
        """
        Push a pre-built 12-node row (shape (12,)) directly.
        Used by the dataset replayer — no noise added because
        all stations already have distinct real values.
        """
        self._ingest(row.astype(np.float32))

    # ── push (single station, e.g. from live ESP32) ───
    def push(self, temperature: float, ph: float, conductivity: float, turbidity: float):
        """
        Add one live reading to the rolling buffer.
        Stations 2 & 3 are derived with small noise so the GCN sees distinct signals.
        conductivity should be in µS/cm. If only TDS (ppm) is available, pass tds/0.64.
        """
        noise = self._rng.normal(0, STATION_NOISE_FRAC, size=(2, 4))
        row = np.array([
            temperature, ph, conductivity, turbidity,
            temperature * (1 + noise[0, 0]), ph * (1 + noise[0, 1]),
            conductivity * (1 + noise[0, 2]), turbidity * (1 + noise[0, 3]),
            temperature * (1 + noise[1, 0]), ph * (1 + noise[1, 1]),
            conductivity * (1 + noise[1, 2]), turbidity * (1 + noise[1, 3]),
        ], dtype=np.float32)
        self._ingest(row)

    def _ingest(self, row: np.ndarray):
        """Common ingestion: update buffer, history, and align past predictions."""
        self.buffer.append(row)
        self.history.append(row)
        self._actual_queue.append(row)
        self._step += 1

    # ── predict ───────────────────────────────────────
    def predict(self) -> dict | None:
        """
        Run inference on the current 48-step buffer.
        Returns None if buffer not yet full.
        """
        if len(self.buffer) < WINDOW:
            return None

        # Build (1, N, T) input
        window_arr = np.stack(list(self.buffer), axis=0)  # (48, 12)
        X = window_arr.T                                   # (12, 48)

        # Per-node normalise
        X_norm = (X - self.mean[:, None]) / (self.std[:, None] + 1e-8)
        X_t = torch.tensor(X_norm[None], dtype=torch.float32, device=self.device)  # (1,12,48)

        with torch.no_grad():
            O = self.temp(X_t)                                  # (1, 12, d_temp)

            A_static_b = self.A_static.unsqueeze(0)             # (1, 12, 12)

            A_rl_b = None
            if self.agent is not None:
                _, picks, _ = self.agent.select_pairs(O, self.device)
                A_rl_b = build_adj_from_pairs(
                    N_NODES, picks[0], self.device
                ).unsqueeze(0)                                   # (1, 12, 12)

            H, _ = self.tcgc(O, A_static_b, A_rl=A_rl_b)       # (1, 12, d_fused)
            Yhat_norm = self.ar(H)                              # (1, 12)

        # Denormalise: two stages
        # Stage 1 — undo the CSV-level normalisation stored in the checkpoint
        Yhat = Yhat_norm.squeeze(0).cpu().numpy() * self.std + self.mean   # still z-score domain
        # Stage 2 — map z-score domain → physical sensor units
        # (checkpoint was trained on synthetic data whose mean≈0, std≈0.3–1.2;
        #  FALLBACK_MEAN/STD encode what "z=0 / z=1" mean in real-world units)
        Yhat = Yhat * _FALLBACK_STD + _FALLBACK_MEAN                       # (12,) physical units
        Yhat = np.clip(Yhat, _PHYS_MIN, _PHYS_MAX)                         # clamp to physical range

        # Store prediction tagged with the future step it targets.
        # The model predicts HORIZON steps ahead, so this prediction
        # corresponds to the actual value at step (self._step + HORIZON).
        self._pred_queue.append({
            "target_step": self._step + HORIZON,
            "values":      Yhat.copy(),
        })

        return {NODES[i]: round(float(Yhat[i]), 4) for i in range(N_NODES)}

    # ── aligned actual vs predicted for chart ─────────
    def get_eval_series(self, node_idx: int, last: int = 40):
        """
        Returns two parallel lists of equal length:
          actual    — ground truth values at steps where a prediction existed
          predicted — what the model said would happen at that step

        Only steps where both actual and prediction are available are included.
        This gives a true actual vs predicted comparison shifted by HORIZON.
        """
        # Build a step→actual lookup from _actual_queue
        # _actual_queue grows one entry per push, indexed by _step at push time
        n_actual = len(self._actual_queue)
        # step of the oldest actual entry
        oldest_step = self._step - n_actual

        actual_lookup = {}
        for offset, row in enumerate(self._actual_queue):
            actual_lookup[oldest_step + offset] = row

        paired_actual, paired_pred = [], []
        for entry in self._pred_queue:
            ts = entry["target_step"]
            if ts in actual_lookup:
                # Actual values are raw CSV z-scores — one step to physical units:
                # x_physical = x_csv * PHYS_STD + PHYS_MEAN
                # (do NOT apply checkpoint mean/std — those are only for model I/O)
                raw_actual = float(actual_lookup[ts][node_idx])
                denorm_actual = float(raw_actual * float(_FALLBACK_STD[node_idx]) + float(_FALLBACK_MEAN[node_idx]))
                lo, hi = float(_PHYS_MIN[node_idx]), float(_PHYS_MAX[node_idx])
                denorm_actual = float(np.clip(denorm_actual, lo, hi))
                # Predicted is already fully denormalized in predict()
                pred_val = float(np.clip(float(entry["values"][node_idx]), lo, hi))
                paired_actual.append(round(denorm_actual, 4))
                paired_pred.append(round(pred_val, 4))

        # Return last N pairs
        return paired_actual[-last:], paired_pred[-last:]

    # ── raw history for fallback (denormalized) ───────
    def get_history(self, node_idx: int, last: int = 40) -> list[float]:
        hist = list(self.history)[-last:]
        lo, hi = float(_PHYS_MIN[node_idx]), float(_PHYS_MAX[node_idx])
        return [
            round(float(np.clip(
                float(r[node_idx]) * float(_FALLBACK_STD[node_idx]) + float(_FALLBACK_MEAN[node_idx]),
                lo, hi
            )), 4)
            for r in hist
        ]

    @property
    def buffer_fill(self) -> int:
        return len(self.buffer)
