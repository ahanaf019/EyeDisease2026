from transformers import AutoModelForImageClassification
import torch.nn as nn
import torch
from models.modules.head import ClassificationHead
from torchvision.models import vit_b_16, ViT_B_16_Weights

class DeiTB(nn.Module):
    def __init__(self, feature_dim, num_classes, dropout=0.2):
        super().__init__()
        # ViT_B_16_Weights.IMAGENET1K_V1 = DeiT weights
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        self.model.heads = nn.Identity()
        input_size = self.model.hidden_dim

        self.classifier = ClassificationHead(
            in_features=input_size,
            embed_dim=feature_dim,
            out_features=num_classes,
            dropout_rate=dropout
        )

    def forward(self, x):
        features = self.model(x)
        return self.classifier(features)