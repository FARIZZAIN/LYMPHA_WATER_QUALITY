from dataclasses import dataclass

@dataclass
class Config:
    # Model hyperparameters — must match the checkpoint
    d_temporal: int = 64
    temporal_kernel: int = 3
    temporal_layers: int = 1
    d_gcn: int = 32
    d_fused: int = 96
    d_z: int = 8
    num_clusters: int = 3

    # DQN config
    dqn_top_k: int = 6
    dqn_eps_start: float = 0.2
    dqn_eps_end: float = 0.05
    dqn_eps_decay_steps: int = 1000
