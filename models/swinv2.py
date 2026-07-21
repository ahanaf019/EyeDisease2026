from transformers import AutoModelForImageClassification
import torch.nn as nn
import torch
from models.modules.head import ClassificationHead
import timm

import torch.nn as nn
from torchvision.models import swin_v2_b, Swin_V2_B_Weights

class SwinV2B(nn.Module):
    def __init__(self, feature_dim, num_classes, dropout=0.2):
        super().__init__()

        # Load pretrained Swin V2 Base
        self.model = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)

        input_size = self.model.head.in_features  # 1024

        # Remove torchvision classifier
        self.model.head = nn.Identity()

        # Your custom classifier
        self.classifier = ClassificationHead(
            in_features=input_size,
            embed_dim=feature_dim,
            out_features=num_classes,
            dropout_rate=dropout
        )

    def forward(self, x):
        features = self.model(x)   # [B, 1024]
        return self.classifier(features)
