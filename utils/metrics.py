import torch
from torchmetrics import Metric

class NamedMetric:
    def __init__(self, metric: Metric, name: str):
        self.metric = metric
        self.name = name

    def compute(self):
        return self.metric.compute()
    
    def reset(self):
        return self.metric.reset()
    
    def update(self, outputs, targets):
        return self.metric.update(outputs, targets)
    
    def to(self, device: str):
        self.metric.to(device)



# class ClasswiseAccuracy:
#     def __init__(self, num_classes, device='cpu'):
#         self.num_classes = num_classes
#         self.device = device
#         self.reset()
    
#     def to(self, device: str):
#         self.device = device

#     def update(self, preds, targets):
#         """
#         Args:
#             preds: Logits tensor of shape (N, num_classes)
#             targets: Ground truth tensor of shape (N,) with class indices
#         """
#         preds = torch.argmax(preds, dim=1)
#         preds = preds.to(self.device)
#         targets = targets.to(self.device)

#         for i in range(self.num_classes):
#             mask = targets == i
#             self.total[i] += mask.sum()
#             self.correct[i] += (preds[mask] == targets[mask]).sum()

#     def compute(self):
#         accuracies = self.correct / (self.total + 1e-8)

#         acc_dict = {
#             f"Class_{i}": f'{acc.item():0.4f}'
#             for i, acc in enumerate(accuracies)
#         }

#         # Wrapper is used so that it does not give error when calling metric.item()
#         return Wrapper(acc_dict)

#     def reset(self):
#         self.correct = torch.zeros(self.num_classes, device=self.device)
#         self.total = torch.zeros(self.num_classes, device=self.device)


class Wrapper:
    def __init__(self, item):
        self.item_ = item
    def item(self):
        return self.item_


import torch


class ClasswiseMetric:
    def __init__(self, num_classes, metric='accuracy', device='cpu'):
        """
        Args:
            num_classes (int)
            metric (str): one of ['accuracy', 'precision', 'recall', 'f1']
            device (str)
        """
        assert metric in ['accuracy', 'precision', 'recall', 'f1']
        self.num_classes = num_classes
        self.metric = metric
        self.device = device
        self.reset()
    
    def to(self, device: str):
        self.device = device
        self.tp = self.tp.to(device)
        self.fp = self.fp.to(device)
        self.fn = self.fn.to(device)
        self.tn = self.tn.to(device)

    def update(self, preds, targets):
        """
        Args:
            preds: Logits tensor of shape (N, num_classes)
            targets: Ground truth tensor of shape (N,)
        """
        preds = torch.argmax(preds, dim=1).to(self.device)
        targets = targets.to(self.device)

        for i in range(self.num_classes):
            pred_i = preds == i
            target_i = targets == i

            self.tp[i] += (pred_i & target_i).sum()
            self.fp[i] += (pred_i & ~target_i).sum()
            self.fn[i] += (~pred_i & target_i).sum()
            self.tn[i] += (~pred_i & ~target_i).sum()

    def compute(self):
        eps = 1e-8
        tp = self.tp.float()
        fp = self.fp.float()
        fn = self.fn.float()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        if self.metric == 'accuracy':   # per-class accuracy
            scores = recall
        elif self.metric == 'precision':
            scores = precision
        elif self.metric == 'recall':
            scores = recall
        elif self.metric == 'f1':
            scores = f1

        metric_dict = {f"Class_{i}": f'{scores[i].item():0.4f}' for i in range(self.num_classes)}
        return Wrapper(metric_dict)

    def reset(self):
        self.tp = torch.zeros(self.num_classes, device=self.device)
        self.fp = torch.zeros(self.num_classes, device=self.device)
        self.fn = torch.zeros(self.num_classes, device=self.device)
        self.tn = torch.zeros(self.num_classes, device=self.device)
