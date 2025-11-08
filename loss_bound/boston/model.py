import torch.nn as nn 

class NN(nn.Module):
    def __init__(self, in_dim=13, hidden_dim=100):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.final_layer = nn.Linear(hidden_dim, 1, bias=False)
        self.init_weights()
        

    def forward(self, x):
        x = self.relu(self.fc2(self.relu(self.fc1(x))))
        return self.final_layer(x)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1/m.in_features**0.5)