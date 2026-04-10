# src/encoders.py
import torch
import torch.nn as nn

class TemporalEncoder1D(nn.Module):
    """
    Per-node temporal encoder.
    Input:  X of shape (B, N, T)
    Output: O of shape (B, N, d_out)
    """
    def __init__(self, d_out: int = 32, k: int = 3, n_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        assert k % 2 == 1, "Use odd kernel size for 'same' padding"
        layers = []
        in_ch = 1
        out_ch = d_out
        pad = k // 2

        # First conv
        layers.append(nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, padding=pad))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))

        # Optional extra conv layers
        for _ in range(n_layers - 1):
            layers.append(nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=pad))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))

        # Pool across time to get one vector per node
        layers.append(nn.AdaptiveAvgPool1d(1))

        self.net = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, N, T)  ->  O: (B, N, d_out)
        """
        B, N, T = X.shape
        x = X.reshape(B * N, 1, T)     # (B*N, 1, T)
        h = self.net(x)                # (B*N, d_out, 1)
        h = h.squeeze(-1)              # (B*N, d_out)
        O = h.view(B, N, -1)           # (B, N, d_out)
        return O
