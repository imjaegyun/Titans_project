# Titans_project/modules/trajectory_encoder.py

import torch.nn as nn
import torch

class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 768)
        )

    def forward(self, x):
        B,T,D = x.shape
        x = x.view(B, T*D)
        return self.net(x)
