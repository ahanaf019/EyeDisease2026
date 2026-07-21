import torch
import torch.nn as nn
from models.models_mamba import vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2
from models.modules.head import ClassificationHead

class Vim(nn.Module):
    def __init__(self, feature_dim, num_classes, dropout=0.2, pretrained=True):
        super(Vim, self).__init__()
        self.model: nn.Module = vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2()
        
        
        if pretrained:
            weights_path = 'vim_weights/vim_b_midclstok_81p9acc.pth'
            checkpoint = torch.load(weights_path, weights_only=False)['model']
            self.model.load_state_dict(checkpoint)
            print(f'Weights Loaded from {weights_path}')
        # self.model.head = nn.Linear(self.model.num_features, 6)
        input_size = self.model.num_features
        self.model.head = nn.Identity()
        self.classifier = ClassificationHead(
            in_features=input_size,
            embed_dim=feature_dim,
            out_features=num_classes,
            dropout_rate=dropout
        )
        

    def forward(self, x):
        features = self.model(x)
        out = self.classifier(features)
        # out = torch.nan_to_num(out, nan=0.0)
        return out