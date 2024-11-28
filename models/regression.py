import torch 
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, hidden_dim):
        super(Network, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(13, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.block(x)