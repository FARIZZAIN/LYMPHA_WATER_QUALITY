from __future__ import annotations
import numpy as np
import torch
from collections import deque

from src.encoders import TemporalEncoder1D
from src.tcgc    import TCGCBlock
from src.ar_head import ARHead
from src.rl_dqn  import DQNAgent, DQNConfig, build_adj_from_pairs

NODES = [
    "Temp_1", "pH_1", "Cond_1", "Turb_1",
    "Temp_2", "pH_2", "Cond_2", "Turb_2",
    "Temp_3", "pH_3", "Cond_3", "Turb_3",
]
N_NODES = 12
WINDOW  = 48
HORIZON = 6

STATION_NOISE_FRAC = 0.01  # ±1% noise added to stations 2 & 3

# Used when the checkpoint has no normalisation metadata
_FALLBACK_MEAN = np.array([
    25.0, 7.2, 420.0, 3.5,
    25.0, 7.2, 420.0, 3.5,
    25.0, 7.2, 420.0, 3.5,
], dtype=np.float32)

_FALLBACK_STD = np.array([
    3.0, 0.5, 150.0, 2.0,
    3.0, 0.5, 150.0, 2.0,
    3.0, 0.5, 150.0, 2.0,
], dtype=np.float32)

# Physical clamp bounds per node: [Temp, pH, Cond, Turb] × 3 stations
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


class ForecastModel:
    def __init__(self, ckpt_path: str, cfg):
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer: deque[np.ndarray] = deque(maxlen=WINDOW)
        self.history: deque[np.ndarray] = deque(maxlen=60)
        self._pred_queue: deque[np.ndarray] = deque(maxlen=200)
        self._actual_queue: deque[np.ndarray] = deque(maxlen=200)
        self._step = 0
        self._rng  = np.random.default_rng(seed=0)
        self._load(ckpt_path)

    def _load(self, ckpt_path: str):
        print(f"[LYMPHA] Loading forecast checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

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

        self.A_static = ckpt.get("A_static", torch.eye(N_NODES, dtype=torch.float32))
        self.A_static = self.A_static.to(self.device)

        self.agent = None
        if "agent" in ckpt:
            self.agent = DQNAgent(
                d_in=self.cfg.d_temporal,
                N_nodes=N_NODES,
                cfg=DQNConfig(
                    top_k=self.cfg.dqn_top_k,
                    eps_start=self.cfg.dqn_eps_end,
                    eps_end=self.cfg.dqn_eps_end,
                ),
            ).to(self.device)
            self.agent.load_state_dict(ckpt["agent"])
            self.agent.eval()

        self.temp.eval()
        self.tcgc.eval()
        self.ar.eval()
        print(f"[LYMPHA] Forecast model ready — device={self.device}, nodes={N_NODES}, window={WINDOW}")

    def push_row(self, row: "np.ndarray"):
        """Push a pre-built 12-node row from the dataset replayer."""
        self._ingest(row.astype(np.float32))

    def push(self, temperature: float, ph: float, conductivity: float, turbidity: float):
        """Add one live ESP32 reading. Stations 2 & 3 get small noise for distinct GCN signals."""
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
        self.buffer.append(row)
        self.history.append(row)
        self._actual_queue.append(row)
        self._step += 1

    def predict(self) -> dict | None:
        if len(self.buffer) < WINDOW:
            return None

        window_arr = np.stack(list(self.buffer), axis=0)  # (48, 12)
        X = window_arr.T                                   # (12, 48)
        X_norm = (X - self.mean[:, None]) / (self.std[:, None] + 1e-8)
        X_t = torch.tensor(X_norm[None], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            O = self.temp(X_t)
            A_static_b = self.A_static.unsqueeze(0)

            A_rl_b = None
            if self.agent is not None:
                _, picks, _ = self.agent.select_pairs(O, self.device)
                A_rl_b = build_adj_from_pairs(N_NODES, picks[0], self.device).unsqueeze(0)

            H, _ = self.tcgc(O, A_static_b, A_rl=A_rl_b)
            Yhat_norm = self.ar(H)

        Yhat = Yhat_norm.squeeze(0).cpu().numpy() * self.std + self.mean
        Yhat = Yhat * _FALLBACK_STD + _FALLBACK_MEAN
        Yhat = np.clip(Yhat, _PHYS_MIN, _PHYS_MAX)

        self._pred_queue.append({
            "target_step": self._step + HORIZON,
            "values":      Yhat.copy(),
        })

        return {NODES[i]: round(float(Yhat[i]), 4) for i in range(N_NODES)}

    def get_eval_series(self, node_idx: int, last: int = 40):
        """Returns (actual, predicted) lists aligned by HORIZON offset."""
        n_actual = len(self._actual_queue)
        oldest_step = self._step - n_actual

        actual_lookup = {}
        for offset, row in enumerate(self._actual_queue):
            actual_lookup[oldest_step + offset] = row

        paired_actual, paired_pred = [], []
        lo, hi = float(_PHYS_MIN[node_idx]), float(_PHYS_MAX[node_idx])
        for entry in self._pred_queue:
            ts = entry["target_step"]
            if ts in actual_lookup:
                raw_actual = float(actual_lookup[ts][node_idx])
                denorm_actual = float(np.clip(
                    raw_actual * float(_FALLBACK_STD[node_idx]) + float(_FALLBACK_MEAN[node_idx]),
                    lo, hi
                ))
                pred_val = float(np.clip(float(entry["values"][node_idx]), lo, hi))
                paired_actual.append(round(denorm_actual, 4))
                paired_pred.append(round(pred_val, 4))

        return paired_actual[-last:], paired_pred[-last:]

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
