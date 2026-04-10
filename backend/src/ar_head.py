# src/ar_head.py
import torch
import torch.nn as nn

class ARHead(nn.Module):
    """
    Linear AR head per node: maps (B, N, d_in) -> (B, N)
    """
    def __init__(self, d_in: int):
        super().__init__()
        self.proj = nn.Linear(d_in, 1)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        H: (B, N, d_in)
        returns Yhat: (B, N)
        """
        y = self.proj(H)      # (B, N, 1)
        return y.squeeze(-1)  # (B, N)
