import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, in_features, embed_dim, out_features, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim // 2, out_features)
        )
    
    def forward(self, x):
        return self.layers(x)
    