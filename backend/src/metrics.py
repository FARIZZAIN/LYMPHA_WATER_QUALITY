# src/metrics.py
import torch

def mae(pred, true):
    return torch.mean(torch.abs(pred - true))

def rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2))

def rse(pred, true):
    """Relative Squared Error."""
    numerator   = torch.sqrt(torch.sum((pred - true) ** 2))
    denominator = torch.sqrt(torch.sum((true - torch.mean(true)) ** 2))
    return numerator / (denominator + 1e-8)

def corr(pred, true):
    """Pearson correlation over flattened tensors."""
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    pred_mean = torch.mean(pred_flat)
    true_mean = torch.mean(true_flat)
    num   = torch.sum((pred_flat - pred_mean) * (true_flat - true_mean))
    denom = torch.sqrt(torch.sum((pred_flat - pred_mean) ** 2) * torch.sum((true_flat - true_mean) ** 2))
    return num / (denom + 1e-8)

def r2(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """R² over all samples and nodes."""
    ss_res = ((pred - true) ** 2).sum()
    mean_y = true.mean()
    ss_tot = ((true - mean_y) ** 2).sum()
    return 1.0 - ss_res / (ss_tot + 1e-8)

def r2_per_node(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """R² per node. Returns (N,) tensor."""
    ss_res = ((pred - true) ** 2).sum(dim=0)
    mean_y = true.mean(dim=0, keepdim=True)
    ss_tot = ((true - mean_y) ** 2).sum(dim=0)
    return 1.0 - ss_res / (ss_tot + 1e-8)
