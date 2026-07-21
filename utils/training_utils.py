from config import Config
import torch

def get_optimizer(params, config: Config):
    if config.optimizer == 'adam':
        return torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == 'adamw':
        return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == 'sgd':
        return torch.optim.SGD(params, lr=config.lr, weight_decay=config.weight_decay)


def get_loss_fn(config: Config):
    if config.loss_fn == 'crossentropy':
        return torch.nn.CrossEntropyLoss()
    if config.loss_fn == 'weighted_crossentropy':
        class_counts = torch.tensor([
            71,    # CSCR
            1056,  # DR
            89,    # ODE
            944,   # GLC
            717,   # Healthy
            311,   # MS
            350,   # MYA
            87,    # RD
            97     # RP
        ], dtype=torch.float)

        # Inverse frequency
        weights = 1.0 / class_counts

        # Normalize (optional but recommended)
        weights = weights / weights.sum() * len(class_counts)
        return torch.nn.CrossEntropyLoss(weight=weights.to('cuda'))