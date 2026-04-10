# src/gcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_adj(A: torch.Tensor):
    """
    Symmetrically normalize adjacency matrix A.
    A: (B, N, N) or (N, N)
    """
    if A.dim() == 2:
        D = A.sum(1)
        D_inv_sqrt = torch.pow(D + 1e-8, -0.5)
        D_inv_sqrt = torch.diag(D_inv_sqrt)
        return D_inv_sqrt @ A @ D_inv_sqrt
    elif A.dim() == 3:
        D = A.sum(-1)
        D_inv_sqrt = torch.pow(D + 1e-8, -0.5).unsqueeze(-1)
        return A * D_inv_sqrt * D_inv_sqrt.transpose(1, 2)
    else:
        raise ValueError("Adjacency must be (N,N) or (B,N,N)")

class GraphConv(nn.Module):
    """
    Simple Graph Convolution layer: O' = σ(Ã O W)
    """
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.b = nn.Parameter(torch.zeros(out_dim)) if bias else None
        nn.init.xavier_uniform_(self.W)

    def forward(self, O, A):
        """
        O: (B, N, d_in)
        A: (N, N) or (B, N, N)
        Output: (B, N, d_out)
        """
        A_hat = normalize_adj(A)
        if A_hat.dim() == 2:
            O_new = A_hat @ O
        else:
            O_new = torch.bmm(A_hat, O)
        O_new = O_new @ self.W
        if self.b is not None:
            O_new = O_new + self.b
        return F.relu(O_new)

class AttentionAdjacency(nn.Module):
    """
    Learns a soft adjacency from node features using attention.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.Wq = nn.Linear(in_dim, in_dim, bias=False)
        self.Wk = nn.Linear(in_dim, in_dim, bias=False)
        self.scale = in_dim ** 0.5

    def forward(self, O):
        """
        O: (B, N, d)
        returns adjacency: (B, N, N)
        """
        Q = self.Wq(O)
        K = self.Wk(O)
        attn = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        A = torch.softmax(attn, dim=-1)
        return A
