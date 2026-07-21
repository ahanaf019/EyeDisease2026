from transformers import AutoModelForImageClassification
import torch.nn as nn
import torch
from models.modules.head import ClassificationHead

class MambaVision(nn.Module):
    def __init__(self, feature_dim, num_classes, dropout=0.2):
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained(
            "nvidia/MambaVision-B-1K",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            trust_remote_code=True
        )
        
        # Remove original head
        input_size = self.model.model.head.in_features
        self.model.model.head = nn.Identity()
        # Enhanced classifier
        # Multi-layer classification head with regularization
        self.classifier = ClassificationHead(
            in_features=input_size, 
            embed_dim=feature_dim,
            out_features=num_classes, 
            dropout_rate=dropout
        )
    
    def forward(self, x):
        features = self.model(x)['logits']
        return self.classifier(features)