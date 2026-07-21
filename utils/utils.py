from glob import glob
import cv2
import torch
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
import numpy as np

def set_reproducable(seed=42):
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def change_learning_rate(optimizer: torch.optim.Optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def set_freeze_root_children(model: torch.nn.Module, n: int, freeze: bool = True):
    """
    Freezes or unfreezes the first `n` root-level children modules of the model.

    Args:
        model (torch.nn.Module): The model to modify.
        n (int): Number of top-level modules to affect.
        freeze (bool): If True, freeze the modules; if False, unfreeze them.
    """
    children = list(model.children())
    
    for i, child in enumerate(children):
        if i < n:
            for param in child.parameters():
                param.requires_grad = not freeze



def read_image(image_path: str, size:int=256):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if size is not None:
        image = cv2.resize(image, (size, size))
    return image


def save_state(save_path: str, model: nn.Module, optim: torch.optim.Optimizer, info: dict | None = None):
    state_dict = {
        'model_state': model.state_dict(),
        'optim_state': optim.state_dict(),
        'info': info
    }
    try:
        torch.save(state_dict, save_path)
        print(f'State saved.')
        # print('Info:', info)
    except Exception as e:
        print(f'ERROR: {e}')

def load_state(path: str, model: nn.Module=None, optim: torch.optim.Optimizer=None)-> tuple[nn.Module, torch.optim.Optimizer]:
    obj = torch.load(path)
    if model is not None:
        model.load_state_dict(obj['model_state'])
        print('Model State Loaded')
    if optim is not None:
        optim.load_state_dict(obj['optim_state'])
        print('Optimizer State Loaded')
    print(f'Loaded state.')
    return model, optim


def denormalize(image, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return image * std + mean



def get_images_and_labels(db_path, limit_per_class=20, val_split=0.2, shuffle_seed=224, random_state=224, print_info=False):
    classes = sorted(glob(f'{db_path}/*'))
    images = sorted(glob(f'{classes}/*'))

    train_images_list = []
    train_labels_list = []
    val_images_list = []
    val_labels_list = []
    unsup_images_list = []
    unsup_labels_list = []

    for _class in classes:
        random.seed(random_state)
        images = sorted(glob(f'{_class}/*'))
        random.shuffle(images)
        labels = [int(image.split('/')[-2]) for image in images]
        val_count = int((val_split) * len(labels[:limit_per_class]))
        
        val_images_list += images[:val_count]
        val_labels_list += labels[:val_count]

        random.seed(shuffle_seed)
        images = images[val_count:]
        labels = labels[val_count:]
        random.shuffle(images)

        train_images_list += images[:limit_per_class]
        train_labels_list += labels[:limit_per_class]
        unsup_images_list += images
        unsup_labels_list += labels

        if print_info:
            print(len(images), len(train_images_list), val_count, len(images[:limit_per_class]))
    
    
    return train_images_list, train_labels_list, val_images_list, val_labels_list, unsup_images_list, unsup_labels_list


import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import islice

def plot_dataloader_grid(
    dataloader,
    n: int,
    file_path: str,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """
    Plot images and labels from a DataLoader in an NxN grid and save to disk.

    Args:
        dataloader: PyTorch DataLoader
        n (int): grid size (NxN)
        file_path (str): path to save the figure
        mean (tuple): normalization mean
        std (tuple): normalization std
    """
    # Get one batch
    images, labels = next(iter(dataloader))

    batch_size = images.size(0)
    max_images = min(batch_size, n * n)

    # Prepare figure
    fig, axes = plt.subplots(n, n, figsize=(n * 3, n * 3))
    axes = axes.flatten()

    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    for i in range(n * n):
        ax = axes[i]
        ax.axis("off")

        if i >= max_images:
            continue

        img = images[i].cpu()
        label = labels[i].item()

        # Denormalize
        img = img * std + mean
        img = img.clamp(0, 1)

        # Convert to HWC for matplotlib
        img = img.permute(1, 2, 0).numpy()

        ax.imshow(img)
        ax.set_title(f"Label: {label}", fontsize=10)

    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight")
    plt.close(fig)
