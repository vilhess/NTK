import torch 
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(NN, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim, bias=False)
        )
        self.init_weights()

    def forward(self, x):
        if x.ndim >= 3:
            x = torch.flatten(x, start_dim=1)
        return self.block(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1/m.in_features**0.5)