import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score
from utils.metrics import ClasswiseMetric, NamedMetric
from torchmetrics import ConfusionMatrix

from utils.utils import save_state, load_state
from datasets import ClassificationDataset
from config import Config
from utils.training_utils import get_loss_fn, get_optimizer
from utils.experiment_tracker import ExperimentTracker
from torch_ema import ExponentialMovingAverage

class ClassifcationTrainer:
    def __init__(self, model: nn.Module, config:Config, tracker:ExperimentTracker, metrics = None):
        self.tracker = tracker
        self.model = model
        
        if config.use_ema:
            self.tracker.write_message(f'Using Exponential Moving Average (EMA): beta={config.ema_decay}')
            self.ema = ExponentialMovingAverage(
                model.parameters(), decay=config.ema_decay
            )
        self.config = config
        self.optim = get_optimizer(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            config=config 
        )
        self.loss_fn = get_loss_fn(config)
        self.device = config.device
        self.num_classes = config.num_classes

        self.conf = ConfusionMatrix(task='multiclass',num_classes=self.num_classes).to(self.device)
        self.metrics = metrics
        if metrics is None:
            self.metrics = [
                NamedMetric(MulticlassAccuracy(num_classes=config.num_classes, average='micro'), name='micro_acc'),
                NamedMetric(MulticlassAccuracy(num_classes=config.num_classes, average='macro'), name='macro_acc'),
                NamedMetric(MulticlassF1Score(num_classes=config.num_classes, average='macro'), name='f1_score'),
                NamedMetric(MulticlassAUROC(num_classes=config.num_classes), name='auc_roc'),
                NamedMetric(ClasswiseMetric(config.num_classes, metric='accuracy', device=config.device), name='classwise_acc')
            ]
        for metric in self.metrics:
            metric.to(config.device)
        
        self.scaler = GradScaler(config.device)


    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        num_epochs = self.config.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=num_epochs, eta_min=1e-6)
        train_losses = []
        val_losses = [np.inf]

        es_counter = 0
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            train_loss = self.train_epoch(train_loader)
            val_loss, metrics_dict = self.evaluate(val_loader, use_progress_bar=False)
            scheduler.step()
            print(f'Learning Rate: {scheduler.get_last_lr()[-1]:0.4e}')
            
            es_counter += 1
            
            self.tracker.track('train_loss', train_loss)
            self.tracker.track('val_loss', val_loss)
            for key in metrics_dict.keys():
                self.tracker.track(f'{key}', metrics_dict[key])
            self.tracker.track('lr', scheduler.get_last_lr()[-1])
            self.tracker.track('es_counter', es_counter)
            self.tracker.log()
            self.tracker.save_to_json()

            save_state(f'{self.config.save_path}/{self.model.__class__.__name__}_running.pt', self.model, self.optim)
            if val_loss < min(val_losses):
                print(f'val_loss improved from {np.min(val_losses):0.4f} to {val_loss:0.4f}')
                self.tracker.write_message(f'val_loss improved from {np.min(val_losses):0.4f} to {val_loss:0.4f}')
                save_state(f'{self.config.save_path}/{self.model.__class__.__name__}_best.pt', self.model, self.optim)
                es_counter = 0
            if es_counter >= self.config.early_stopping_patience:
                self.tracker.write_message('Early Stopping')
                break
            train_losses.append(train_loss)
            val_losses.append(val_loss)

    

    def train_epoch(self, loader: DataLoader):
        self.model.train()
        losses = []
        data_iter = loader
        if self.config.train_progress_bar:
            data_iter = tqdm(loader, leave=True)
            
        for images, labels in data_iter:
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optim.zero_grad()
            # with torch.autocast(self.device):
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            if torch.isnan(loss).any():
                self.tracker.write_message("NaN train loss detected! Training stopped.")
                exit()
            losses.append(loss.item())
            
            loss.backward()
            # self.scaler.scale(loss).backward()
            if self.config.grad_clip_norm != -1:
                # self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip_norm)
            # self.scaler.step(self.optim)
            # self.scaler.update()
            self.optim.step()
            
            if self.config.use_ema:
                self.ema.update()  # Update EMA after optimizer step
            
        print(f'Train Loss: {np.mean(losses):0.4f}')
        return np.mean(losses)
    

    def evaluate(self, loader: DataLoader, generate_confusion_matrix=False, use_progress_bar=True, _type='val', return_predictions=False):
        '''
        Evaludation Function
        
        loader: A Pytorch Dataloader
        use_progress_bar: Whether to show progressbar. Important for large datasets.
        generate_confusion_matrix: Whether to generate a confusion matrix.
        _type: either "val" or "test". This is for logging purposes.
        return_predictions: Whether to return labels, predictions, and softmax scores for external analysis.
        '''
        self.model.eval()
        
        if self.config.use_ema:
            self.ema.store()      # Store current params
            self.ema.copy_to()    # Apply EMA weights
        
        losses = []

        self.conf.reset()
        for metric in self.metrics:
            metric.reset() 

        # Storage for predictions if requested
        if return_predictions:
            all_labels = []
            all_predictions = []
            all_scores = []

        data_iter = loader
        if use_progress_bar:
            data_iter = tqdm(loader, leave=False)

        with torch.inference_mode():
            for images, labels in data_iter:
                # print(labels)

                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                losses.append(loss.item())

                for metric in self.metrics:
                    metric.update(outputs, labels)
                self.conf.update(outputs, labels)
                
                # Store predictions if requested
                if return_predictions:
                    all_labels.append(labels.cpu())
                    all_predictions.append(outputs.argmax(dim=1).cpu())
                    all_scores.append(torch.softmax(outputs, dim=1).cpu())
        
        print(f'Eval Loss: {np.mean(losses):0.4f}', end=' | ')
        
        if self.config.use_ema:
            self.ema.restore()
        
        
        metrics_dict = {}
        for metric in self.metrics:
            metrics_dict[f'{_type}_{metric.name}'] = metric.compute().item()
        
        if generate_confusion_matrix:
            metrics_dict[f'{_type}_confusion_matrix'] = self.conf.compute().cpu().numpy()
        
        # Add predictions to metrics_dict if requested
        if return_predictions:
            scores_array = torch.cat(all_scores).numpy()
            num_classes = scores_array.shape[1]
            
            predictions_dict = {
                'labels': torch.cat(all_labels).numpy(),
                'predictions': torch.cat(all_predictions).numpy()
            }
            
            # Add softmax scores for each class as separate columns
            for i in range(num_classes):
                predictions_dict[f'score_{i}'] = scores_array[:, i]
            
            return np.mean(losses), metrics_dict, predictions_dict
        
        return np.mean(losses), metrics_dict


