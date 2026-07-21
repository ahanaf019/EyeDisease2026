"""
Profiles the ClassificationTrainer: model complexity, peak GPU memory,
training time per epoch, and inference FPS.

Usage:
    from profile_trainer import profile_training
    profile_training(trainer, train_loader, val_loader)
"""

import time
import numpy as np
import torch
from ptflops import get_model_complexity_info


def _is_cuda(device):
    return torch.cuda.is_available() and "cuda" in str(device)


def _reset_cuda_peak(device):
    if _is_cuda(device):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def _read_cuda_peak_mb(device):
    if not _is_cuda(device):
        return 0.0
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def _measure_model_complexity(model, input_res):
    macs, params = get_model_complexity_info(
        model, input_res, as_strings=False,
        print_per_layer_stat=False, verbose=False,
    )
    return macs, macs / 1e9, params


def _time_train_epoch(trainer, train_loader, device):
    if _is_cuda(device):
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    trainer.train_epoch(train_loader)
    if _is_cuda(device):
        torch.cuda.synchronize(device)
    return time.perf_counter() - start


def _measure_inference_fps(model, val_loader, device, warmup_batches=3, repeats=3):
    model.eval()
    fps_values = []
    latency_values = []

    with torch.inference_mode():
        for _ in range(repeats):
            total_images = 0
            total_time = 0.0

            for i, (images, _) in enumerate(val_loader):
                images = images.to(device)

                if _is_cuda(device):
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                model(images)
                if _is_cuda(device):
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start

                if i >= warmup_batches:
                    total_images += images.size(0)
                    total_time += elapsed

            fps_values.append(total_images / total_time if total_time > 0 else 0.0)
            latency_values.append(
                total_time / max(len(val_loader) - warmup_batches, 1) * 1000
            )

    return np.mean(fps_values), np.std(fps_values), np.mean(latency_values)


def profile_training(trainer, train_loader, val_loader, input_res=(3, 224, 224)):
    device = trainer.config.device
    model = trainer.model
    num_batches = len(train_loader)
    num_samples = len(train_loader.dataset)

    print("Calculating model complexity...")
    macs, gflops, params = _measure_model_complexity(model, input_res)

    print("Warmup +  peak GPU memory...")
    _reset_cuda_peak(device)
    trainer.train_epoch(train_loader)
    trainer.evaluate(val_loader, use_progress_bar=False)
    peak_memory_mb = _read_cuda_peak_mb(device)

    print("Calculating avg training time...")
    epoch_times = []
    for _ in range(3):
        elapsed = _time_train_epoch(trainer, train_loader, device)
        epoch_times.append(elapsed)
        trainer.evaluate(val_loader, use_progress_bar=False)
    epoch_time_mean = np.mean(epoch_times)
    epoch_time_std = np.std(epoch_times)

    print("Calculating inference FPS...")
    fps_mean, fps_std, batch_latency_ms = _measure_inference_fps(
        model, val_loader, device, warmup_batches=3, repeats=3
    )

    minutes, seconds = divmod(epoch_time_mean, 60)

    print("\n--- profiling results ---")
    print(f"model complexity (input {input_res[0]}x{input_res[1]}x{input_res[2]}): "
          f"{macs:,.0f} MACs, {gflops:.2f} GFLOPs, {params:,.0f} params")

    if _is_cuda(device):
        print(f"peak GPU memory during training: {peak_memory_mb:,.1f} MB")
    else:
        print("peak GPU memory: n/a, no CUDA")

    print(f"training time per epoch: {epoch_time_mean:.2f}s ± {epoch_time_std:.2f}s "
          f"({int(minutes)}m {seconds:.1f}s), {num_samples / epoch_time_mean:,.1f} samples/sec, "
          f"{epoch_time_mean / num_batches * 1000:.1f} ms/batch")

    print(f"inference: {fps_mean:,.1f} ± {fps_std:.1f} FPS, {batch_latency_ms:.1f} ms/batch")