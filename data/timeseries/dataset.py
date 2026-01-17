import torch
from torch.utils.data import Dataset
import pandas as pd


class GlobalTempDataset(Dataset):
    def __init__(self, path, seq_len=10):
        df = pd.read_csv(path)
        self.data = torch.tensor(df["data"].values, dtype=torch.float32)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len]
        return x, y
