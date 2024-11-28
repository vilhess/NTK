import torch 
from torch.utils.data import Dataset

import pandas as pd
from sklearn.preprocessing import StandardScaler

class BostonDataset(Dataset):
    
    def __init__(self, path="Boston.csv"):
        data = pd.read_csv(path)
        self.features = data.drop('medv', axis=1)
        self.targets = data['medv']
        self.features = StandardScaler().fit_transform(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.Tensor(self.features[idx]), torch.Tensor([self.targets[idx]]) 