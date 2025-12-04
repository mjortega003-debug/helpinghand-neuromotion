
import torch
import torch.nn as nn

class BicepsMLP(nn.Module):
    def __init__(self, in_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()   # outputs: [0,1] for left, right
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_dim)
        return self.net(x)
