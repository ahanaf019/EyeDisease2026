import json
import logging
from torch.utils.tensorboard import SummaryWriter

class ExperimentTracker:
    def __init__(self, save_path: str, filename='experiment_tracker.log', tensorboard=True):
        self.logger = logging.getLogger('ExperimentTracker')
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(save_path + '/' + filename)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        # Add the file handler to the logger
        self.logger.addHandler(file_handler)

        self.save_path = save_path
        
        if tensorboard:
            self.tensorboard_writer = SummaryWriter(log_dir=save_path)
        self.dictionary = {}

    def track(self, item_name: str, value):
        if item_name not in self.dictionary.keys():
            self.dictionary[item_name] = []
        self.dictionary[item_name].append(value)
        
        if 'train' in item_name:
            tb_itemname = f'train/{item_name.replace("train", "")}'
        elif 'val' in item_name:
            tb_itemname = f'val/{item_name.replace("val", "")}'
        else:
            tb_itemname = item_name
        
        # Skip plotting the confusion matrix and classwise acc
        if 'confusion_matrix' not in tb_itemname and 'classwise_acc' not in tb_itemname:
            self.tensorboard_writer.add_scalar(tb_itemname, value, len(self.dictionary[item_name]))

    def save_to_json(self):
        with open(self.save_path  + '/' + 'logs.json', 'w') as f:
            json.dump(self.dictionary, f, indent=4)
    
    def log(self):
        message = []
        for key in self.dictionary.keys():
            if self.dictionary[key] == 'lr':
                message.append(f'{key}: {self.dictionary[key][-1]:0.4e}')
            elif isinstance(self.dictionary[key][-1], float):
                message.append(f'{key}: {self.dictionary[key][-1]:0.4f}')
            else:
                message.append(f'{key}: {self.dictionary[key][-1]}')
        self.logger.info(' | '.join(message))
    
    def write_message(self, message):
        self.logger.info(message)