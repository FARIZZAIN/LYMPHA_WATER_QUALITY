# src/rl_dqn.py
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def pair_indices(N: int) -> List[Tuple[int,int]]:
    """All undirected pairs i<j."""
    return [(i,j) for i in range(N) for j in range(i+1, N)]

def build_adj_from_pairs(N: int, pairs: List[Tuple[int,int]], device, sym=True):
    A = torch.zeros(N, N, device=device)
    for (i,j) in pairs:
        A[i,j] = 1.0
        if sym:
            A[j,i] = 1.0
    A = A + 1e-4 * torch.eye(N, device=device)
    return A

class PairwiseFeaturizer(nn.Module):
    """
    Turn per-node embeddings O (B,N,d) into per-pair features (B,M,f).
    """
    def __init__(self, d_in: int, d_hidden: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(4*d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, 1)
        )

    def forward(self, O: torch.Tensor) -> torch.Tensor:
        B, N, d = O.shape
        pairs = pair_indices(N)
        oi, oj = [], []
        for (i,j) in pairs:
            oi.append(O[:, i, :])
            oj.append(O[:, j, :])
        Oi = torch.stack(oi, dim=1)
        Oj = torch.stack(oj, dim=1)
        feats = torch.cat([Oi, Oj, (Oi - Oj).abs(), Oi * Oj], dim=-1)
        q = self.proj(feats).squeeze(-1)
        return q

@dataclass
class DQNConfig:
    eps_start: float = 0.2
    eps_end: float = 0.05
    eps_decay_steps: int = 1500
    top_k: int = 2
    gamma: float = 0.0
    lr: float = 1e-3
    target_update: int = 200
    buffer_size: int = 5000
    batch_size: int = 64

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = []
        self.idx = 0

    def push(self, s, a_idx, r, s2):
        self.data.append((s, a_idx, r, s2))
        if len(self.data) > self.capacity:
            self.data.pop(0)

    def sample(self, batch_size: int):
        return random.sample(self.data, min(batch_size, len(self.data)))

    def __len__(self):
        return len(self.data)

class DQNAgent(nn.Module):
    def __init__(self, d_in: int, N_nodes: int, cfg: DQNConfig):
        super().__init__()
        self.N = N_nodes
        self.cfg = cfg
        self.online = PairwiseFeaturizer(d_in)
        self.target = PairwiseFeaturizer(d_in)
        self.update_target()
        self.buf = ReplayBuffer(cfg.buffer_size)
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.steps = 0
        self.pairs = pair_indices(N_nodes)
        self.M = len(self.pairs)

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def epsilon(self):
        e0, e1, K = self.cfg.eps_start, self.cfg.eps_end, self.cfg.eps_decay_steps
        t = min(self.steps, K)
        return e0 + (e1 - e0) * (t / K)

    def select_pairs(self, O: torch.Tensor, device) -> Tuple[torch.Tensor, list, torch.Tensor]:
        with torch.no_grad():
            Q = self.online(O)

        B = O.size(0)
        eps = self.epsilon()
        picks = []
        mask_idx = torch.zeros(B, self.M, dtype=torch.bool, device=device)

        for b in range(B):
            if random.random() < eps:
                idx = torch.randperm(self.M, device=device)[:self.cfg.top_k]
            else:
                idx = torch.topk(Q[b], k=self.cfg.top_k, dim=-1).indices
            chosen = [self.pairs[int(i)] for i in idx]
            picks.append(chosen)
            mask_idx[b, idx] = True

        self.steps += 1
        return Q, picks, mask_idx

    def learn(self, O: torch.Tensor, mask_idx: torch.Tensor, reward: torch.Tensor):
        Q_all = self.online(O)
        Q_chosen = Q_all[mask_idx]
        r_rep = reward.unsqueeze(1).repeat(1, self.cfg.top_k).reshape(-1)
        loss = F.mse_loss(Q_chosen, r_rep.detach())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if self.steps % self.cfg.target_update == 0:
            self.update_target()
        return float(loss.item())
