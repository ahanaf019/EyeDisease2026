import os

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score
from utils.metrics import ClasswiseMetric, NamedMetric
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score
from torchmetrics import ConfusionMatrix
from utils.utils import save_state, load_state
from config import Config
from tqdm import tqdm
import numpy as np
from visualize.confusion_matrix import plot_normalized_confusion_matrix
from models import get_model
from utils.arg_parser import parse_ensenble_testing_arguments
from datasets import get_labeled_dataset
from utils.experiment_tracker import ExperimentTracker
from utils.training_utils import get_loss_fn

def evaluate_ensemble(
    models: List[nn.Module],
    loss_fn,
    metrics: List[NamedMetric],
    loader: DataLoader,
    config,
    use_progress_bar: bool = True,
    _type: str = "test",
    ensemble_mode: str = "mean",       # New: ensemble mode
    ensemble_weights: torch.Tensor = None  # New: for weighted_mean
):
    """
    Ensemble Evaluation Function with multiple ensemble modes.

    ensemble_mode: "mean", "sum", "weighted_mean", "max", "mean_plus_max"
    ensemble_weights: torch.Tensor of shape [num_models] for weighted_mean
    """
    device = config.device

    # Confusion matrix
    conf = ConfusionMatrix(
        task="multiclass",
        num_classes=config.num_classes
    ).to(device)
    conf.reset()

    # Reset metrics
    for metric in metrics:
        metric.to(device)
        metric.reset()

    # Set models to eval
    for model in models:
        model.eval()

    losses = []

    data_iter = tqdm(loader, leave=False) if use_progress_bar else loader

    def combine_logits(logits_list):
        probs = logits_list
        # probs = [F.softmax(logits, dim=1) for logits in logits_list]
        stacked = torch.stack(probs)  # shape: [num_models, batch_size, num_classes]
        if ensemble_mode == "mean":
            return stacked.mean(dim=0)
        elif ensemble_mode == "sum":
            return stacked.sum(dim=0)
        elif ensemble_mode == "weighted_mean":
            if ensemble_weights is None:
                raise ValueError("ensemble_weights mus\t be provided for weighted_mean")
            w = ensemble_weights.to(device).view(-1, 1, 1)
            return (stacked * w).sum(dim=0)
        elif ensemble_mode == "max":
            return stacked.max(dim=0).values
        elif ensemble_mode == "mean_plus_max":
            mean_logits = stacked.mean(dim=0)
            max_logits = stacked.max(dim=0).values
            return 0.5 * (mean_logits + max_logits)
        else:
            raise ValueError(f"Unknown ensemble_mode: {ensemble_mode}")

    with torch.inference_mode():
        for images, labels in data_iter:
            images = images.to(device)
            labels = labels.to(device)

            # Collect logits from all models
            logits_list = [model(images) for model in models]

            # Combine according to ensemble_mode
            ensemble_logits_out = combine_logits(logits_list)

            # Loss
            loss = loss_fn(ensemble_logits_out, labels)
            losses.append(loss.item())

            # Metrics
            for metric in metrics:
                metric.update(ensemble_logits_out, labels)

            conf.update(ensemble_logits_out, labels)

    mean_loss = float(np.mean(losses))
    print(f"Eval Loss: {mean_loss:.4f}", end=" | ")

    metrics_dict = {}
    for metric in metrics:
        metrics_dict[f"{_type}_{metric.name}"] = metric.compute().item()

    metrics_dict[f"{_type}_confusion_matrix"] = conf.compute().cpu().numpy()

    return mean_loss, metrics_dict


if __name__ == '__main__':
    args = parse_ensenble_testing_arguments()
    
    snapshots = [int(x) for x in args.snapshots.split(',')]
    ensemble_mode = args.ensemble_mode
    ensemble_weights = [float(x) for x in args.ensemble_weights.split(',')]
    
    print(snapshots)
    experiments_dir_prefix = 'experiment'
    models = []
    
    for snapshot in snapshots:
        path = f'{args.snapshot_dir}/{experiments_dir_prefix}-{snapshot:04d}'
        config = Config.load_from_json(path)
        print(path, os.path.exists(path), config.model)
        model = get_model(config.model, config.embed_dim, config.num_classes).to(config.device)
        load_state(path=f'{config.save_path}/{model.__class__.__name__}_best.pt', model=model, optim=None)
        models.append(model)
    
    tracker = ExperimentTracker(config.save_path, filename=f'test_ensemble_results-sn={args.snapshots}_mode={ensemble_mode}.txt', tensorboard=False)
    
    tracker.write_message(f'Using Models:')
    for model in models:
        tracker.write_message(f'{model.__class__.__name__}')
    
    _, _, test_loader = get_labeled_dataset(config, tracker)
    loss_fn = get_loss_fn(config)
    
    test_loss, metrics_dict = evaluate_ensemble(
        models=models,
        loss_fn=loss_fn,
        metrics=[
                NamedMetric(MulticlassAccuracy(num_classes=config.num_classes, average='micro'), name='micro_acc'),
                NamedMetric(MulticlassAccuracy(num_classes=config.num_classes, average='macro'), name='macro_acc'),
                NamedMetric(MulticlassF1Score(num_classes=config.num_classes, average='macro'), name='f1_score'),
                NamedMetric(MulticlassAUROC(num_classes=config.num_classes), name='auc_roc'),
                NamedMetric(ClasswiseMetric(config.num_classes, metric='accuracy', device=config.device), name='classwise_acc'),
                NamedMetric(ClasswiseMetric(config.num_classes, metric='precision', device=config.device), name='classwise_pr'),
                NamedMetric(ClasswiseMetric(config.num_classes, metric='f1', device=config.device), name='classwise_f1')
            ],
        loader=test_loader,
        config=config,
        ensemble_mode=ensemble_mode,
        ensemble_weights=torch.tensor(ensemble_weights)
        
    )
    
    tracker.write_message(f'ensemble_test_loss :\t{test_loss}')
    for key in metrics_dict.keys():
        tracker.write_message(f'{key} :\t{metrics_dict[key]}')
        
    plot_normalized_confusion_matrix(
        metrics_dict['test_confusion_matrix'], 
        save_path=f'{config.save_path}/ensemble_cm-sn={args.snapshots}_mode={ensemble_mode}.svg',
        class_labels=list(test_loader.dataset.class_map.values()),
        new_order=[4,0,1,2,3,5,6,7,8]
    )