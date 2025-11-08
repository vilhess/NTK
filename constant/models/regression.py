import torch 
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, hidden_dim):
        super(Network, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(13, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.init_weights()

    def forward(self, x):
        return self.block(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1/m.in_features**0.5)