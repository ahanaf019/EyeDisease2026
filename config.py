TRAIN_DB_PATH = '/home/ahanaf/Datasets/MergedDR/'
TEST_PATH = '/home/ahanaf/Datasets/MergedDR/DDR/test/'
NUM_CLASSES=5
IMAGE_LIMIT_PER_CLASS = 1550
EMBED_DIM = 2560
IMAGE_SIZE = 448        # FIXED
BATCH_SIZE = 16
NUM_EPOCHS = 100
INIT_LEARNING_RATE = 5e-4



# loss_cls = cross_entropy(logits, labels)
# loss_metric = triplet_loss(embeddings, labels)
# loss = loss_cls + 0.5 * loss_metric

import os
import json
class Config:
    def __init__(self, *args, create_experiment: bool = True, **kwargs):
        self.working_dir: str = kwargs['working_dir']
        self.model: str = kwargs['model']
        self.num_classes: str = kwargs['num_classes']
        self.embed_dim: str = kwargs['embed_dim']
        self.epochs: int = kwargs['epochs']
        self.optimizer: str = kwargs['optimizer']
        self.weight_decay: int = kwargs['weight_decay']
        self.lr: float = kwargs['lr']
        self.grad_clip_norm: int = kwargs['grad_clip_norm']
        self.loss_fn: str = kwargs['loss_fn']
        self.batch_size: int = kwargs['batch_size']
        self.early_stopping_patience: int = kwargs['early_stopping_patience']
        self.image_size: int = kwargs['image_size']
        self.device: str = kwargs['device']
        self.db_dir: str = kwargs['db_dir']
        self.train_db: str = kwargs['train_db']
        self.test_db: str = kwargs['test_db']
        self.seed: int = kwargs['seed']
        self.db_worker_count: int = kwargs['db_worker_count']
        self.train_progress_bar: bool = kwargs['train_progress_bar']
        self.sampling_mode: str = kwargs['sampling_mode']
        self.sample_size: int = kwargs['sample_size']
        # self.num_warmup_epochs: int = kwargs['num_warmup_epochs']
        
        
        self.use_ema: bool = kwargs['use_ema']
        self.ema_decay: float = kwargs['ema_decay']

        if create_experiment:
            self.create_experiments_path()
            self.save_config_as_json()


    def __str__(self):
        out = f'{self.__class__.__name__} object:\n'
        for key in self.__dict__.keys():
            out += f'\t{key}: {self.__dict__[key]}\n'
        return out


    def create_experiments_path(self):
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir, exist_ok=True)
        
        if len(os.listdir(self.working_dir)) == 0:
            self.save_path = f'{self.working_dir}/experiment-{0:04d}'
            os.makedirs(self.save_path)
        
        else:
            last_experiment = sorted(os.listdir(self.working_dir))[-1]
            count = int(last_experiment.split('-')[-1]) + 1
            self.save_path = f'{self.working_dir}/experiment-{count:04d}'
            os.makedirs(self.save_path)

    def save_config_as_json(self):
        with open(f'{self.save_path}/config.json', 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    @classmethod
    def load_from_json(cls, snapshot_path: str):
        """
        Load a Config object from a saved config.json file.
        """
        with open(f'{snapshot_path}/config.json', 'r') as f:
            data = json.load(f)
        # print(data)
        # Remove fields that are created at runtime
        # data.pop('save_path', None)

        instance = cls(**data, create_experiment=False)
        cls.save_path = data['save_path']
        return instance