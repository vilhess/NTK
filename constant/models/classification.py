import torch 
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, hidden_dim):
        super(NN, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(28*28, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10, bias=False)
        )
        self.init_weights()

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.block(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1/m.in_features**0.5)

class CNN(nn.Module):
    def __init__(self, hidden_dim):
        super(CNN, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim*28*28, 10, bias=False)
        )
        self.init_weights()
        
    def forward(self, x):
        return self.block(x)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1/m.in_features**0.5)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=1/(m.in_channels * m.kernel_size[0] * m.kernel_size[1])**0.5)