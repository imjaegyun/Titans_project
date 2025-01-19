# Titans_project/modules/trajectory_encoder.py

import torch.nn as nn

class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128):  # input_dim=2로 수정
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 768)  # Output dimension to match embedding
        )

    def forward(self, x):
        return self.net(x)  # [B, 768]
