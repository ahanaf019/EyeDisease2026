import argparse

def parse_training_arguments():
    parser = argparse.ArgumentParser(description="Eye Disease Classification by ahanaf019")

    # Add arguments
    parser.add_argument('--working_dir', type=str, required=True, help='The directory to save weights, results, figures, configs etc')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--loss_fn', type=str, default='mse', help='Learning rate')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of output classes')
    parser.add_argument('--embed_dim', type=int, required=True, help='Embedding space dimensions in the classification head')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--early_stopping_patience', type=int, default=100, help='Patience for Early Stopping')
    parser.add_argument('--image_size', type=none_or_int, default=224, help='Image size')
    parser.add_argument('--db_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--train_db', type=str, required=True, help='Dataset to train the model')
    

    parser.add_argument('--sampling_mode', type=str, required=False, default='default', help='Whether to oversample or not. Only applied to the training dataset. Accepted values: \"default\", \"all_to_N\", \"under_to_N\"')
    parser.add_argument('--sample_size', type=int, required=False, default=none_or_int, help='Required when sampling_mode is not \"default\"')

    parser.add_argument('--test_db', type=str, required=True, help='Dataset for evaluation')
    parser.add_argument('--seed', type=int, default=0, help='Set random seed (0=random, others=set seed)')
    parser.add_argument('--db_worker_count', type=int, default=1, help='Number of worker processes for loading the datasets')
    parser.add_argument('--grad_clip_norm', type=float, default=-1, help='Gradient Clipping: -1: no clipping, others: clip grad')
    parser.add_argument('--train_progress_bar', type=bool, default=True, help='Show progress bar during training')
    # parser.add_argument('--num_warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    
    
    parser.add_argument('--use_ema', type=bool, default=False, help='Use Exponential Moving Average')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='Decay value of EMA')

    # Parse them
    args = parser.parse_args()
    return args

def parse_testing_arguments():
    parser = argparse.ArgumentParser(description="Eye Disease Classification by ahanaf019")

    # Add arguments
    parser.add_argument('--snapshot', type=str, required=True, help='The directory where weights, results, configs etc are saved.')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--db_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--test_db', type=str, required=True, help='Dataset for evaluation')
    parser.add_argument('--seed', type=int, default=0, help='Set random seed (0=random, others=set seed)')
    parser.add_argument('--db_worker_count', type=int, default=1, help='Number of worker processes for loading the datasets')

    # Parse them
    args = parser.parse_args()
    return args


def parse_ensenble_testing_arguments():
    parser = argparse.ArgumentParser(description="Eye Disease Classification by ahanaf019")

    # Add arguments
    parser.add_argument('--snapshot_dir', type=str, required=True, help='The directory where all the snapshots are saved.')
    parser.add_argument('--snapshots', type=str, required=True, help='Comma seperated number of snapshots.')
    parser.add_argument('--ensemble_mode', type=str, required=True, help='One of mean, sum, weighted_sum, max, mean_plus_max.')
    parser.add_argument('--ensemble_weights', type=str, required=False, default='1', help='Comma seperated weights. Applicable when ensemble_mode=weighted_sum.')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--db_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--test_db', type=str, required=True, help='Dataset for evaluation')
    parser.add_argument('--seed', type=int, default=0, help='Set random seed (0=random, others=set seed)')
    parser.add_argument('--db_worker_count', type=int, default=1, help='Number of worker processes for loading the datasets')

    # Parse them
    args = parser.parse_args()
    return args


def none_or_int(value):
    if value == 'None':
        return None
    return int(value)