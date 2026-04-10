# src/tcgc.py
import torch
import torch.nn as nn
from src.gcn import GraphConv, AttentionAdjacency

class TCGCBlock(nn.Module):
    """
    Triple-Channel Graph Convolution with 1x1 fusion.
      - Channel 1: Static adjacency (e.g., identity/correlation)
      - Channel 2: Attention adjacency (learned from features)
      - Channel 3: RL-learned adjacency
    """
    def __init__(self, d_in: int, d_gcn: int = 32, d_fused: int = 64, use_rl_channel: bool = True):
        super().__init__()
        self.use_rl = use_rl_channel

        # One GCN per channel
        self.gcn_static = GraphConv(d_in, d_gcn)
        self.gcn_attn   = GraphConv(d_in, d_gcn)
        if self.use_rl:
            self.gcn_rl = GraphConv(d_in, d_gcn)

        # Attention adjacency builder
        self.attn_adj = AttentionAdjacency(d_in)

        # 1x1 fusion (implemented as a per-node linear mix across channels)
        in_fuse = d_gcn * (3 if self.use_rl else 2)
        self.fuse = nn.Linear(in_fuse, d_fused)

    def forward(self, O: torch.Tensor, A_static: torch.Tensor, A_rl: torch.Tensor | None = None):
        """
        O:        (B, N, d_in)
        A_static: (N, N) or (B, N, N)
        A_rl:     (N, N) or (B, N, N)  optional
        returns H_fused: (B, N, d_fused)
        """
        A_attn = self.attn_adj(O)             # (B, N, N)

        H_static = self.gcn_static(O, A_static)
        H_attn   = self.gcn_attn(O, A_attn)

        feats = [H_static, H_attn]

        if self.use_rl:
            if A_rl is None:
                A_rl = A_attn.detach()
            H_rl = self.gcn_rl(O, A_rl)
            feats.append(H_rl)

        H_concat = torch.cat(feats, dim=-1)
        H_fused  = self.fuse(H_concat)
        return H_fused, {"A_attn": A_attn, "A_rl": A_rl}
