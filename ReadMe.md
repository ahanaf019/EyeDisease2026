# Comparative analysis of CNN, Transformer, and Mamba-based models for automated eye disease classification from fundus images

**Authors:** Md. Ahanaf Arif Khan, Sangeeta Biswas

This paper compares six models across the CNN, Transformer, and Mamba families for automated multi-class eye disease classification from color fundus images. All experiments were performed using the Eye Disease Image Dataset

## Setup

Two virtual environments are used, otherwise identical in dependencies (`requirements.txt`):

1. **vim-env** — has the custom CUDA kernels for Vim installed, following the setup instructions in the [Vim repo](https://github.com/hustvl/Vim). Used only for training and evaluating the Vim model.
2. **base-env** — used for training and evaluating all other models (ConvNeXt-B, DeiT, MambaVision, RegNet-Y, Swin v1/v2).

```bash
# base environment
python -m venv base-env
source base-env/bin/activate
pip install -r requirements.txt

# vim environment
python -m venv vim-env
source vim-env/bin/activate
pip install -r requirements.txt
# then follow https://github.com/hustvl/Vim to build the Vim CUDA kernels
```

## Repository Structure

```
.
├── bootstrap_ci.py            # bootstrap CIs for accuracy/F1/AUC-ROC from a predictions CSV
├── config.py                  # training/evaluation configuration
├── datasets/                  # dataset loading and preprocessing for EDID
├── data_split/
├── models/                    # ConvNeXt-B, DeiT, MambaVision, RegNet-Y, Swin v1/v2, ViM
├── profiler.py                # GPU memory, epoch time, inference FPS, MACs/GFLOPs
├── requirements.txt           # Python dependencies
├── rope.py                    # rotary position embedding
├── significance_testing.ipynb # statistical significance testing between models
├── test_ensemble.py           # evaluates an ensemble of saved snapshots
├── test_model.py              # evaluates a single trained snapshot
├── trainers/                  # ClassificationTrainer, train/eval loops
├── train_model.py             # main training entry point
├── utils/                     # arg parsing, experiment tracking, metrics, helpers
└── visualize/                 # confusion matrix plotting
```

## Usage

### Train

Trains a model from scratch on EDID and saves checkpoints, logs, and config under `--working_dir` as a new experiment folder.

```bash
python train_model.py \
    --model ConvNeXtB \
    --num_classes 9 \
    --embed_dim 1024 \
    --epochs 50 \
    --optimizer adamw \
    --lr 0.0001 \
    --weight_decay 0.00 \
    --db_dir /path/to/datasets \
    --train_db Eye_Disease_Image_Dataset \
    --test_db Eye_Disease_Image_Dataset \
    --working_dir /path/to/experiments \
    --image_size 224 \
    --batch_size 32 \
    --db_worker_count 16 \
    --seed 42 \
    --grad_clip_norm 1.0 \
    --loss_fn crossentropy \
    --use_ema True \
    --ema_decay 0.999 \
    --early_stopping_patience 15 \
    --sampling_mode under_to_N \
    --sample_size 500
```

### Test

Loads a single trained snapshot and evaluates it on the test set, writing predictions and metrics to the snapshot folder.

```bash
python test_model.py \
    --snapshot /path/to/experiments/experiment-0004 \
    --db_dir /path/to/datasets \
    --test_db Eye_Disease_Image_Dataset
```

### Ensemble

Combines predictions from multiple trained snapshots (by experiment number) using the given ensemble mode and evaluates the combined result.

```bash
python test_ensemble.py \
    --snapshot_dir /path/to/experiments \
    --snapshots 8,15,17 \
    --ensemble_mode max \
    --db_dir /path/to/datasets \
    --test_db Eye_Disease_Image_Dataset
```

### Bootstrap confidence intervals

Takes the predictions CSV produced by `test_model.py` and reports bootstrap confidence intervals for accuracy, F1, and AUC-ROC. Saves a summary CSV and a full text log next to the input CSV.

```bash
python bootstrap_ci.py \
    --csv /path/to/experiments/experiment-0023/predictions.csv \
    --n_boot 1000 \
    --ci 95 \
    --seed 0 \
```

## Citation

The paper is currently under review. This section will be updated with the full citation once it is published.

## Contact

**Sangeeta Biswas**

Department of Computer Science and Engineering, Faculty of Engineering, University of Rajshahi, Rajshahi-6205, Bangladesh.
Contact: [sangeeta.cse@ru.ac.bd](mailto:sangeeta.cse@ru.ac.bd)
