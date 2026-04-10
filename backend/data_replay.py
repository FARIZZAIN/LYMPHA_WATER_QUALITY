"""
data_replay.py  —  Synthetic Dataset Replay
============================================
Streams rows from a CSV file on a background thread, simulating
live sensor readings from 3 stations × 4 sensors = 12 nodes.

Row format expected (wide, one column per node):
  Temp_1, pH_1, Cond_1, Turb_1,
  Temp_2, pH_2, Cond_2, Turb_2,
  Temp_3, pH_3, Cond_3, Turb_3

Each row is pushed to:
  1. The forecast rolling buffer (all 12 nodes, exact values)
  2. The potability model (Station 1 only)
  3. latest_reading (dashboard display)
"""

from __future__ import annotations
import time
import threading
import numpy as np
import pandas as pd
from typing import Callable

# Node order must match checkpoint exactly
NODES = [
    "Temp_1", "pH_1", "Cond_1", "Turb_1",
    "Temp_2", "pH_2", "Cond_2", "Turb_2",
    "Temp_3", "pH_3", "Cond_3", "Turb_3",
]


class DatasetReplayer:
    """
    Cycles through a dataset CSV and fires a callback for each row.

    Args:
        csv_path:  Path to the wide-format CSV with 12 node columns.
        interval:  Seconds between rows.
        loop:      If True, restarts from row 0 when the dataset ends.
        col_map:   Optional rename map if CSV column names differ from NODES.
    """

    def __init__(
        self,
        csv_path: str,
        interval: float = 3.0,
        loop: bool = True,
        col_map: dict[str, str] | None = None,
    ):
        self.interval = interval
        self.loop     = loop
        self._idx     = 0
        self._running = False

        df = pd.read_csv(csv_path)

        # Rename columns if needed
        if col_map:
            df = df.rename(columns=col_map)

        # Verify all 12 node columns are present
        missing = [n for n in NODES if n not in df.columns]
        if missing:
            # Try case-insensitive match
            lower_map = {c.lower(): c for c in df.columns}
            for n in NODES:
                if n not in df.columns and n.lower() in lower_map:
                    df = df.rename(columns={lower_map[n.lower()]: n})

        still_missing = [n for n in NODES if n not in df.columns]
        if still_missing:
            raise ValueError(
                f"CSV is missing node columns: {still_missing}\n"
                f"Available columns: {list(df.columns)}"
            )

        self._data = df[NODES].to_numpy(dtype=np.float32)  # (T, 12)
        self._len  = len(self._data)
        print(f"[Replay] Loaded {self._len} rows from {csv_path}")

    # ── public ────────────────────────────────────────────────
    def start(self, callback: Callable[[np.ndarray], None]):
        """
        Start the replay thread.
        callback(row: np.ndarray of shape (12,)) is called for each row.
        """
        self._running = True
        t = threading.Thread(target=self._loop, args=(callback,), daemon=True)
        t.start()
        print(f"[Replay] Started — interval={self.interval}s, rows={self._len}, loop={self.loop}")

    def stop(self):
        self._running = False

    @property
    def current_index(self) -> int:
        return self._idx

    @property
    def total_rows(self) -> int:
        return self._len

    # ── internal ──────────────────────────────────────────────
    def _loop(self, callback: Callable):
        while self._running:
            if self._idx >= self._len:
                if self.loop:
                    self._idx = 0
                    print("[Replay] Dataset end reached — looping back to row 0.")
                else:
                    print("[Replay] Dataset exhausted — stopping.")
                    self._running = False
                    break

            row = self._data[self._idx]
            self._idx += 1

            try:
                callback(row)
            except Exception as e:
                print(f"[Replay] Callback error at row {self._idx}: {e}")

            time.sleep(self.interval)
