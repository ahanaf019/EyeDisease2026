import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from config import Config
from utils.utils import load_state, set_reproducable
from datasets import get_labeled_dataset
from trainers import ClassifcationTrainer
from utils.arg_parser import parse_testing_arguments
from visualize.confusion_matrix import plot_normalized_confusion_matrix
from config import Config
from models import get_model
from utils.experiment_tracker import ExperimentTracker
from utils.metrics import NamedMetric, ClasswiseMetric
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    args = parse_testing_arguments()
    config = Config.load_from_json(args.snapshot)
    config.train_db = 'None'
    config.use_ema = False

    if config.seed != 0:
        set_reproducable(config.seed)
    
    tracker = ExperimentTracker(config.save_path, filename='test_results.txt', tensorboard=False)
    tracker.write_message(f'Testing snapshot from: {args.snapshot}')
    tracker.write_message(f'Test dataset: {args.test_db}')
    _, _, test_loader = get_labeled_dataset(config, tracker)
    model = get_model(config.model, config.embed_dim, config.num_classes).to(config.device)
    trainer = ClassifcationTrainer(
        model=model, 
        config=config, 
        tracker=tracker,
        metrics = [
                NamedMetric(MulticlassAccuracy(num_classes=config.num_classes, average='micro'), name='micro_acc'),
                NamedMetric(MulticlassAccuracy(num_classes=config.num_classes, average='macro'), name='macro_acc'),
                NamedMetric(MulticlassF1Score(num_classes=config.num_classes, average='macro'), name='f1_score'),
                NamedMetric(MulticlassAUROC(num_classes=config.num_classes), name='auc_roc'),
                NamedMetric(ClasswiseMetric(config.num_classes, metric='accuracy', device=config.device), name='classwise_acc'),
                NamedMetric(ClasswiseMetric(config.num_classes, metric='precision', device=config.device), name='classwise_pr'),
                NamedMetric(ClasswiseMetric(config.num_classes, metric='f1', device=config.device), name='classwise_f1')
            ]
    )
    
    load_state(path=f'{config.save_path}/{model.__class__.__name__}_best.pt', model=model, optim=None)
    tracker.write_message('Testing...')
    test_loss, metrics_dict, predictions = trainer.evaluate(test_loader, generate_confusion_matrix=True, use_progress_bar=True, _type='test', return_predictions=True)
    
    tracker.write_message(f'test_loss :\t{test_loss}')
    for key in metrics_dict.keys():
        tracker.write_message(f'{key} :\t{metrics_dict[key]}')
        
    plot_normalized_confusion_matrix(
        metrics_dict['test_confusion_matrix'], 
        save_path=f'{config.save_path}/{model.__class__.__name__}_cm.svg',
        class_labels=list(test_loader.dataset.class_map.values()),
        new_order=[4,0,1,2,3,5,6,7,8]
        )
    
    predictions['image'] = test_loader.dataset.images
    predictions['db_label'] = test_loader.dataset.labels
    pd.DataFrame(predictions).to_csv(f'{config.save_path}/predictions.csv', index=False)
