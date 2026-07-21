import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from config import Config
from utils.utils import load_state, set_reproducable, plot_dataloader_grid
from datasets import get_labeled_dataset
from trainers import ClassifcationTrainer
from utils.arg_parser import parse_training_arguments
from config import Config
from models import get_model
from utils.experiment_tracker import ExperimentTracker
from profiler import profile_training

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    args = parse_training_arguments()
    config = Config(**vars(args), create_experiment=True)

    if config.seed != 0:
        set_reproducable(config.seed)
    
    experiment_tracker = ExperimentTracker(config.save_path)
    train_loader, val_loader, test_loader = get_labeled_dataset(config, experiment_tracker)
    
    plot_dataloader_grid(train_loader, 4, f'{config.save_path}/training_samples.png')
    plot_dataloader_grid(val_loader, 4, f'{config.save_path}/validation_samples.png')


    model = get_model(config.model, config.embed_dim, config.num_classes).to(config.device)
    trainer = ClassifcationTrainer(model=model, config=config, tracker=experiment_tracker)
    trainer.fit(train_loader, val_loader)
    
    # Measure model complexity, inference + training times, etc
    profile_training(trainer, train_loader, val_loader, input_res=(3, 224, 224)) 
    