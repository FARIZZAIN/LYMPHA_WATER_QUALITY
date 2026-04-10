import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# =========================
# DATASET
# =========================

class TimeGraphDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# =========================
# HELPERS
# =========================

def train_val_test_split_idx(S: int, train_ratio: float, val_ratio: float):
    b = int(S * train_ratio)
    d = int(S * (train_ratio + val_ratio))
    return {"train": (0, b), "val": (b, d), "test": (d, S)}

def build_windows(values: np.ndarray, window: int, horizon: int):
    """
    values: (T, N) -> X: (S, N, window), Y: (S, N)
    """
    T, N = values.shape
    S = T - window - horizon + 1
    if S <= 0:
        raise ValueError("Not enough timesteps to build windows. Reduce window/horizon or provide more data.")
    X = np.zeros((S, N, window), dtype=np.float32)
    Y = np.zeros((S, N), dtype=np.float32)
    for i in range(S):
        X[i] = values[i:i+window].T
        Y[i] = values[i + window + horizon - 1]
    return X, Y

# =========================
# LONG-FORMAT LOADER
# =========================

def _parse_ts_col(df: pd.DataFrame) -> pd.Series:
    if 'Timestamp as DateTime' in df.columns:
        s = df['Timestamp as DateTime'].astype(str).str.replace(r'\s*0{4}$', '', regex=True)
        ts = pd.to_datetime(s, format='%Y/%m/%d %H:%M:%S', errors='coerce')
        if ts.notna().any():
            return ts
    if 'Timestamp' in df.columns:
        s = df['Timestamp'].astype(str).str.replace(r'0{4}$', '', regex=True)
        ts = pd.to_datetime(s, format='%Y%m%d%H%M%S', errors='coerce')
        return ts
    raise ValueError("No parseable timestamp column found.")

def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out

def _impute_df(df: pd.DataFrame, how: str = "ffill_bfill") -> pd.DataFrame:
    if how == "ffill_bfill":
        return df.ffill().bfill()
    return df

def read_long_and_pivot(csv_path: str,
                        option: str = "single_node",
                        node_id_filter: str | None = None,
                        resample_rule: str = "10min",
                        impute: str = "ffill_bfill") -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    df = df.assign(ts=_parse_ts_col(df)).dropna(subset=['ts']).sort_values('ts')

    vars_ = [c for c in ['Temperature', 'Conductivity', 'pH', 'DissolvedOxygen'] if c in df.columns]
    if not vars_:
        raise ValueError("No known sensor columns found.")
    df[vars_] = _coerce_numeric(df[vars_], vars_)[vars_]

    if option == "single_node":
        if 'Node Id' not in df.columns:
            raise ValueError("Missing 'Node Id' column.")
        if node_id_filter is None:
            node_id_filter = df['Node Id'].mode().iloc[0]
        df = df[df['Node Id'].astype(str) == str(node_id_filter)]
        wide = df.set_index('ts')[vars_].resample(resample_rule).mean()
        wide = _impute_df(wide, impute)
        return wide.to_numpy(), vars_

    if option == "all_pairs":
        if 'Node Id' not in df.columns:
            raise ValueError("Missing 'Node Id' column.")
        df['Node Id'] = df['Node Id'].astype(str)
        agg = df.groupby(['Node Id', 'ts'], as_index=False)[vars_].mean()
        frames, cols = [], []
        for v in vars_:
            tmp = agg.pivot(index='ts', columns='Node Id', values=v)
            tmp.columns = [f"{c}_{v}" for c in tmp.columns]
            frames.append(tmp)
            cols.extend(tmp.columns)
        wide = pd.concat(frames, axis=1).sort_index().resample(resample_rule).mean()
        wide = _impute_df(wide, impute)
        return wide.to_numpy(), cols

    raise ValueError("option must be 'single_node' or 'all_pairs'")

# =========================
# MAIN ENTRY
# =========================

def make_dataloaders(csv_path: str, window: int, horizon: int,
                     target_nodes: tuple[str, ...] | None,
                     batch_size=64, train_ratio=0.7, val_ratio=0.15,
                     is_long_format: bool = False,
                     long_option: str = "single_node",
                     node_id_filter: str | None = None,
                     resample_rule: str = "10min",
                     impute: str = "ffill_bfill"):

    if is_long_format:
        values, cols = read_long_and_pivot(
            csv_path, option=long_option, node_id_filter=node_id_filter,
            resample_rule=resample_rule, impute=impute
        )
    else:
        df = pd.read_csv(csv_path)
        if target_nodes:
            cols = list(target_nodes)
        else:
            drop_like = {"time", "date", "stamp"}
            cols = [c for c in df.columns
                    if (df[c].dtype != object) and not any(k in c.lower() for k in drop_like)]
        values = df[cols].to_numpy()

    X, Y = build_windows(values, window, horizon)
    S = X.shape[0]
    idx = train_val_test_split_idx(S, train_ratio, val_ratio)
    (a, b), (c, d), (e, f) = idx["train"], idx["val"], idx["test"]

    mean = X[a:b].mean(axis=(0, 2), keepdims=True)
    std  = X[a:b].std(axis=(0, 2), keepdims=True) + 1e-8
    X = (X - mean) / std
    Y = (Y - mean.squeeze((0, 2))) / std.squeeze((0, 2))

    ds_tr = TimeGraphDataset(X[a:b], Y[a:b])
    ds_va = TimeGraphDataset(X[c:d], Y[c:d])
    ds_te = TimeGraphDataset(X[e:f], Y[e:f])

    _pin = torch.cuda.is_available()
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  drop_last=False, pin_memory=_pin)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=_pin)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=_pin)

    meta = {
        "num_nodes": X.shape[1],
        "window": X.shape[2],
        "cols": cols,
        "train_samples": b - a,
        "val_samples": d - c,
        "test_samples": f - e,
        "mean": mean.squeeze((0, 2)).tolist(),
        "std":  std.squeeze((0, 2)).tolist(),
    }
    return dl_tr, dl_va, dl_te, meta
