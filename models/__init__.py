from models.mamba_vision import MambaVision
from models.swinv2 import SwinV2B
from models.swinv1 import SwinB
from models.convnext_b import ConvNeXtB
from models.regnet_y import RegNetY16GF
from models.deit import DeiTB

def get_model(model_name, embed_dim, num_classes):
    if model_name == 'MambaVision':
        return MambaVision(feature_dim=embed_dim, num_classes=num_classes)
    elif model_name == 'SwinV2B':
        return SwinV2B(feature_dim=embed_dim, num_classes=num_classes)
    elif model_name == 'SwinB':
        return SwinB(feature_dim=embed_dim, num_classes=num_classes)
    elif model_name == 'ConvNeXtB':
        return ConvNeXtB(feature_dim=embed_dim, num_classes=num_classes)
    elif model_name == 'RegNetY16GF':
        return RegNetY16GF(feature_dim=embed_dim, num_classes=num_classes)
    elif model_name == 'DeiTB':
        return DeiTB(feature_dim=embed_dim, num_classes=num_classes)
    elif model_name == 'Vim':
        from models.vim import Vim
        return Vim(feature_dim=embed_dim, num_classes=num_classes)