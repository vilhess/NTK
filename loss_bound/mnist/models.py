import torch 
import torch.nn as nn 

class NN(nn.Module):
    def __init__(self, hidden_dim=10):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(784, hidden_dim) 
        self.fc3 = nn.Linear(hidden_dim, 10)   
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc3(x)
        return x
    
class CNN(nn.Module):
    def __init__(self, hidden_channels=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1)
        self.final_layer = nn.Linear(in_features=hidden_channels*7*7, out_features=10)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = x.flatten(start_dim=1)
        x = self.final_layer(x)
        return x