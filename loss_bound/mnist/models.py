import torch 
import torch.nn as nn 

class NN(nn.Module):
    def __init__(self, hidden_dim=10):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(784, hidden_dim, bias=False) 
        self.fc3 = nn.Linear(hidden_dim, 10, bias=False) 
        self.init_weights()
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc3(x)
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1/m.in_features**0.5)
    
class CNN(nn.Module):
    def __init__(self, hidden_channels=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.final_layer = nn.Linear(in_features=hidden_channels*7*7, out_features=10, bias=False)
        self.act = nn.ReLU()
        self.init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = x.flatten(start_dim=1)
        x = self.final_layer(x)
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=1/(m.in_channels * m.kernel_size[0] * m.kernel_size[1])**0.5)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1/m.in_features**0.5)
