import torch 
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, hidden_dim):
        super(NN, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(28*28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )
    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.block(x)
    
class CNN(nn.Module):
    def __init__(self, hidden_dim):
        super(CNN, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim*28*28, 10)
        )
    def forward(self, x):
        return self.block(x)