from __future__ import annotations
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.metrics import mae, rmse, rse, corr, r2, r2_per_node
from src.data import make_dataloaders
from src.config import Config
from src.encoders import TemporalEncoder1D
from src.tcgc import TCGCBlock
from src.ar_head import ARHead
from src.rl_dqn import DQNAgent, DQNConfig, build_adj_from_pairs


@torch.no_grad()
def inverse_standardize(Y_std: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    mean = torch.tensor(mean, device=Y_std.device).view(1, -1)
    std  = torch.tensor(std,  device=Y_std.device).view(1, -1)
    Y = Y_std * std + mean
    return Y.cpu().numpy()


@torch.no_grad()
def evaluate_checkpoint(ckpt_path: str, cfg: Config, use_rl: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl_tr, dl_va, dl_te, meta = make_dataloaders(
        cfg.data_path, cfg.window, cfg.horizon, cfg.target_nodes,
        cfg.batch_size, cfg.train_ratio, cfg.val_ratio,
        is_long_format=cfg.is_long_format,
        long_option=cfg.long_option,
        node_id_filter=cfg.node_id_filter,
        resample_rule=cfg.resample_rule,
        impute=cfg.impute
    )

    mean, std = np.array(meta["mean"]), np.array(meta["std"])
    N = meta["num_nodes"]

    A_static = torch.eye(N, dtype=torch.float32, device=device)

    temp    = TemporalEncoder1D(d_out=cfg.d_temporal, k=cfg.temporal_kernel, n_layers=cfg.temporal_layers).to(device)
    tcgc    = TCGCBlock(d_in=cfg.d_temporal, d_gcn=cfg.d_gcn, d_fused=cfg.d_fused, use_rl_channel=True).to(device)
    ar      = ARHead(d_in=cfg.d_fused).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    temp.load_state_dict(ckpt["temp"])
    tcgc.load_state_dict(ckpt["tcgc"])
    if "ar" in ckpt:
        ar.load_state_dict(ckpt["ar"])
    A_static = ckpt.get("A_static", A_static).to(device)

    agent = None
    if use_rl and "agent" in ckpt:
        agent = DQNAgent(d_in=cfg.d_temporal, N_nodes=N, cfg=DQNConfig()).to(device)
        agent.load_state_dict(ckpt["agent"])
        agent.eval()

    temp.eval(); tcgc.eval(); ar.eval()

    Yhat_list, Y_list = [], []
    for X, Y in dl_te:
        X, Y = X.to(device), Y.to(device)
        O = temp(X)
        B = X.size(0)
        A_static_b = A_static.unsqueeze(0).expand(B, N, N)
        A_rl_b = None
        if agent is not None:
            _, picks, _ = agent.select_pairs(O, device)
            A_rl_b = torch.stack([build_adj_from_pairs(N, picks[b], device) for b in range(B)], dim=0)
        H, _ = tcgc(O, A_static_b, A_rl=A_rl_b)
        Yhat_list.append(ar(H))
        Y_list.append(Y)

    Yhat_std = torch.cat(Yhat_list, dim=0)
    Y_std    = torch.cat(Y_list,    dim=0)
    Yhat  = inverse_standardize(Yhat_std, mean, std)
    Ytrue = inverse_standardize(Y_std,    mean, std)

    Yhat_t  = torch.from_numpy(Yhat).float()
    Ytrue_t = torch.from_numpy(Ytrue).float()

    mae_val  = mae(Yhat_t, Ytrue_t)
    rmse_val = rmse(Yhat_t, Ytrue_t)
    rse_val  = rse(Yhat_t, Ytrue_t)
    corr_val = corr(Yhat_t, Ytrue_t)
    r2_val   = r2(Yhat_t, Ytrue_t)

    print(f"[TEST]  MAE={mae_val:.4f}  RMSE={rmse_val:.4f}  RSE={rse_val:.4f}  CORR={corr_val:.4f}  R2={r2_val:.4f}")
    node_names = meta["cols"]
    r2_nodes   = r2_per_node(Yhat_t, Ytrue_t)
    print()
    for i, name in enumerate(node_names):
        p, t = Yhat_t[:, i], Ytrue_t[:, i]
        print(f"{name:>18}: MAE={mae(p,t):.4f}, RMSE={rmse(p,t):.4f}, RSE={rse(p,t):.4f}, CORR={corr(p,t):.4f}, R2={float(r2_nodes[i]):.4f}")

    return {
        "cols": node_names, "Yhat": Yhat, "Ytrue": Ytrue,
        "mae": float(mae_val), "rmse": float(rmse_val),
        "r2": float(r2_val),   "rse": float(rse_val), "corr": float(corr_val),
    }


def plot_series(y_true: np.ndarray, y_pred: np.ndarray, title: str, savepath: str, last: int | None = 200):
    if last is not None and last < len(y_true):
        y_true = y_true[-last:]
        y_pred = y_pred[-last:]
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Pred")
    plt.title(title); plt.xlabel("Test time index"); plt.ylabel("Value")
    plt.legend(); plt.tight_layout(); plt.savefig(savepath); plt.close()
