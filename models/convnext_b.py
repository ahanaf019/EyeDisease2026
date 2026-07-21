from transformers import AutoModelForImageClassification
import torch.nn as nn
import torch
from models.modules.head import ClassificationHead
import timm


class ConvNeXtB(nn.Module):
    def __init__(self, feature_dim, num_classes, dropout=0.2):
        super().__init__()

        self.model = timm.create_model(
            "convnext_base",
            pretrained=True,
            num_classes=0
        )

        input_size = self.model.num_features

        self.classifier = ClassificationHead(
            in_features=input_size,
            embed_dim=feature_dim,
            out_features=num_classes,
            dropout_rate=dropout
        )

    def forward(self, x):
        features = self.model(x)
        return self.classifier(features)
