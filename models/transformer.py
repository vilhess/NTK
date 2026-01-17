import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, hidden_dim, seq_len=10):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(1, hidden_dim, bias=False)
        self.block = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=1, batch_first=True, dropout=0.0),
            nn.Flatten(),
            nn.Linear(hidden_dim * seq_len, 1, bias=False)
        )
        self.init_weights()
        # for m in self.modules():
        #     if isinstance(m, nn.LayerNorm):
        #         m.weight.requires_grad = False
        #         m.bias.requires_grad = False
        
    def forward(self, x):
        # x: (batch, seq_len) -> (batch, seq_len, hidden_dim)
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        x = self.embedding(x)  # (batch, seq_len, hidden_dim)
        return self.block(x)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1/m.in_features**0.5)
            elif isinstance(m, nn.LayerNorm):
                nn.init.normal_(m.weight, mean=0.0, std=1/m.normalized_shape[0]**0.5)
