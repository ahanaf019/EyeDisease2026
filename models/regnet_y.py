import torch
import torch.nn as nn
from models.modules.head import ClassificationHead
from torchvision.models import regnet_y_16gf, RegNet_Y_16GF_Weights

class RegNetY16GF(nn.Module):
    def __init__(self, feature_dim, num_classes, dropout=0.2):
        super().__init__()

        self.model = regnet_y_16gf(weights=RegNet_Y_16GF_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()
        # input_size = self.model.

        self.classifier = ClassificationHead(
            in_features=3024,
            embed_dim=feature_dim,
            out_features=num_classes,
            dropout_rate=dropout
        )

    def forward(self, x):
        features = self.model(x)   # (B, C)
        return self.classifier(features)