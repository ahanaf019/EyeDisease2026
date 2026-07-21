import os
from datasets.classification_dataset import ClassificationDataset
import albumentations as A
from torchvision import transforms
from config import Config
import pandas as pd
import json
from torch.utils.data import DataLoader
from utils.experiment_tracker import ExperimentTracker
import cv2

IMAGE_NORM_MEANS = [0.485, 0.456, 0.406]
IMAGE_NORM_STD = [0.229, 0.224, 0.225]


_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_NORM_MEANS, std=IMAGE_NORM_STD)
])


def get_labeled_dataset(config: Config, tracker: ExperimentTracker):
    if config.train_db.lower() != 'none':
        if not os.path.exists(f'{config.db_dir}/{config.train_db}'):
            raise ValueError("Invalid dataset dir")
        if not os.path.exists(f'{config.db_dir}/{config.train_db}/db_split.csv'):
            raise ValueError("db_split.csv not found")
        
        df_train = pd.read_csv(f'{config.db_dir}/{config.train_db}/db_split.csv')
    
        train_df = df_train[df_train['split'] == 'train']
        val_df = df_train[df_train['split'] == 'val']
        
        tracker.write_message(f'Using the {config.train_db} dataset for training')
        tracker.write_message(f'Training size={train_df.shape[0]}')
        tracker.write_message(f'Validation size={val_df.shape[0]}')
    
        train_images, train_labels = list(train_df['image_path']), list(train_df['labels_encoded'])
        val_images, val_labels = list(val_df['image_path']), list(val_df['labels_encoded'])
    
        with open(f"{config.db_dir}/{config.train_db}/class_map.json", "r") as f:
            class_map = json.load(f)
        tracker.write_message(f'Sampling Mode: {config.sampling_mode}')
        tracker.write_message(f'Sample Size Per Class: {config.sample_size}')
        train_db = ClassificationDataset(
            images=train_images,
            labels=train_labels,
            image_size=config.image_size,
            db_root=f'{config.db_dir}/{config.train_db}/',
            class_map=class_map,
            augments=get_training_augments(config),
            transforms=_transforms,
            mode=config.sampling_mode,
            N=config.sample_size,
        )
        val_db = ClassificationDataset(
            images=val_images,
            labels=val_labels,
            image_size=config.image_size,
            db_root=f'{config.db_dir}/{config.train_db}/',
            class_map=class_map,
            augments=get_test_augments(config),
            transforms=_transforms,
            mode='default',
            N=None,
        )
        tracker.write_message(f'Total Training Samples (After dataloader construction): {len(train_db)}')
        tracker.write_message(f'Total Validation Samples (After dataloader construction): {len(val_db)}')

        train_loader = DataLoader(train_db, batch_size=config.batch_size, shuffle=True, num_workers=config.db_worker_count, prefetch_factor=2)
        val_loader = DataLoader(val_db, batch_size=16, shuffle=False, num_workers=2)
    else:
        train_loader, val_loader = None, None
        tracker.write_message('WARNING: training dataset not specified')    
    
    if config.test_db.lower() != 'none':
        if not os.path.exists(f'{config.db_dir}/{config.test_db}'):
            raise ValueError("Invalid dataset dir")
        if not os.path.exists(f'{config.db_dir}/{config.test_db}/db_split.csv'):
            raise ValueError("db_split.csv not found")
    
        df_test = pd.read_csv(f'{config.db_dir}/{config.test_db}/db_split.csv')
        
        test_df = df_test[df_test['split'] == 'test']
        
        with open(f"{config.db_dir}/{config.test_db}/class_map.json", "r") as f:
            class_map = json.load(f)
        
        tracker.write_message(f'Test size={test_df.shape[0]}')
        
        test_images, test_labels = list(test_df['image_path']), list(test_df['labels_encoded'])
        
        test_db = ClassificationDataset(
            images=test_images,
            labels=test_labels,
            image_size=config.image_size,
            db_root=f'{config.db_dir}/{config.test_db}/',
            class_map=class_map,
            augments=get_test_augments(config),
            transforms=_transforms,
            mode='default',
            N=None,
        )
        
        test_loader = DataLoader(test_db, batch_size=16, shuffle=False, num_workers=2)
    else:
        test_loader = None
        tracker.write_message('WARNING: test dataset not specified') 
    return train_loader, val_loader, test_loader


def get_unlabeled_dataset(config:Config):
    pass



def get_training_augments(config: Config):
    size = get_crop_size(config.image_size)

    return A.Compose([
        # --- Spatial (very conservative) ---
        A.Resize(size, size, interpolation=cv2.INTER_CUBIC),
        A.RandomResizedCrop(size=(config.image_size, config.image_size), scale=(0.8, 1)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.5),

        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(-0.05, 0.05),
            rotate=(-15, 15),
            shear=(-10, 10),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),

        # --- Illumination & Contrast (CRITICAL) ---
        A.CLAHE(
            clip_limit=(2.0, 4.0),
            tile_grid_size=(8, 8),
            p=0.7,
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.6,
        ),

        # For lesion emphasis
        A.OneOf([
            A.Sharpen(alpha=(0.1, 0.3)),
            A.Emboss(alpha=(0.1, 0.3)),
        ], p=0.2),

        # --- Camera / Sensor artifacts ---
        A.GaussNoise(p=0.3),

        A.ImageCompression(
            quality_range=(85, 100),
            p=0.3,
        ),

        # --- Color consistency ---
        A.ColorJitter(
            brightness=0.05,
            contrast=0.05,
            saturation=0.05,
            hue=0.02,
            p=0.3,
        ),
    ])

def get_test_augments(config: Config):
    size = get_crop_size(config.image_size)

    return A.Compose([
        A.Resize(size, size, interpolation=cv2.INTER_CUBIC),
        A.CenterCrop(config.image_size, config.image_size),
        A.CLAHE(clip_limit=3.0, p=1.0),
    ], seed=config.seed)

def get_crop_size(image_size):
    if image_size == 224:
        size = 256
    elif image_size == 448:
        size = 512
    elif image_size == 672:
        size = 768
    return size