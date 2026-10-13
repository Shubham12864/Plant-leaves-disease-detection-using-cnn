#!/usr/bin/env python3
"""Fine-tune Torchvision EfficientNet-B0 on PlantVillage with two-GPU DDP."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
from torchvision import models
from tqdm.auto import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224
LOCAL_BATCH_SIZE = 16
MAX_EPOCHS = 12
PATIENCE = 3
SEED = 42

class CachedImages(Dataset):
    """Read only one uint8 image at a time from a disk-backed NPY cache."""
    def __init__(self, image_file, labels_file):
        self.images = np.load(image_file, mmap_mode="r", allow_pickle=False)
        self.labels = np.load(labels_file, mmap_mode="r", allow_pickle=False)
        if self.images.ndim != 4 or self.images.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise ValueError(f"Bad cached image shape: {self.images.shape}")
        if self.images.dtype != np.uint8 or len(self.images) != len(self.labels):
            raise ValueError("Image cache must be uint8 and match its label count")
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        image = torch.from_numpy(np.array(self.images[index], copy=True))
        label = int(self.labels[index])
        return image, label

def gpu_augment_and_normalize(image, device, training):
    """Convert NHWC uint8 to normalized NCHW float and augment on this rank's GPU."""
    x = image.to(device=device, dtype=torch.float32, non_blocking=True)
    x = x.permute(0, 3, 1, 2).div_(255.0).contiguous(memory_format=torch.channels_last)
    if training:
        batch = x.shape[0]
        flip = torch.rand(batch, device=device) < 0.5
        x = torch.where(flip.view(batch, 1, 1, 1), x.flip(3), x)
        angle = (torch.rand(batch, device=device) * 2.0 - 1.0) * (12.0 * math.pi / 180.0)
        cosine, sine = angle.cos(), angle.sin()
        theta = torch.zeros(batch, 2, 3, device=device)
        theta[:, 0, 0], theta[:, 0, 1] = cosine, -sine
        theta[:, 1, 0], theta[:, 1, 1] = sine, cosine
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        brightness = torch.rand(batch, 1, 1, 1, device=device) * 0.24 + 0.88
        x = (x * brightness).clamp_(0.0, 1.0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)

def score(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)

def evaluate(model, dataset, classes, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    y_true, y_pred = [], []
    started = time.perf_counter()
    progress = tqdm(loader, desc="EfficientNet-B0 validation", unit="batch", leave=True, mininterval=1.0)
    with torch.inference_mode():
        for images, labels in progress:
            x = gpu_augment_and_normalize(images, device, training=False)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
            seen = len(y_true)
            progress.set_postfix(val_accuracy=f"{accuracy_score(y_true, y_pred):.4f}",
                                 img_s=f"{seen / max(time.perf_counter() - started, 1e-6):.1f}", refresh=False)
    model.train()
    return score(y_true, y_pred)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--val-labels", required=True)
    parser.add_argument("--classes-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Select GPU T4 x2 in Kaggle settings.")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 2:
        raise RuntimeError(f"Expected 2 DDP processes for the two T4s; torchrun started {world}.")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(1)
    random.seed(SEED + rank); np.random.seed(SEED + rank)
    torch.manual_seed(SEED + rank); torch.cuda.manual_seed_all(SEED + rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True

    classes = json.loads(Path(args.classes_json).read_text(encoding="utf-8"))
    train_ds = CachedImages(args.train_images, args.train_labels)
    val_ds = CachedImages(args.val_images, args.val_labels)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Train and validation caches must both contain images")
    if int(train_ds.labels.max()) >= len(classes) or int(val_ds.labels.max()) >= len(classes):
        raise ValueError("Cached label index does not match the class list")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = output_dir / "efficientnet_b0_best.pt", output_dir / "efficientnet_b0_last.pt"
    resume_config = {
        "architecture": "efficientnet_b0", "classes": classes, "seed": SEED,
        "image_size": IMAGE_SIZE, "local_batch_size": LOCAL_BATCH_SIZE,
        "global_batch_size": LOCAL_BATCH_SIZE * world, "optimizer": "AdamW",
        "learning_rate": 1e-4, "weight_decay": 1e-4, "pretrained": True,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
    }
    resume = None
    if last_path.is_file():
        resume = torch.load(last_path, map_location="cpu", weights_only=False)
        if resume.get("resume_config") != resume_config:
            raise ValueError("Existing EfficientNet checkpoint settings do not match this run. Preserve it and use a fresh output folder.")
        print(f"Rank {rank}: resume checkpoint found at completed epoch {resume['epoch']}.", flush=True)

    weights = models.EfficientNet_B0_Weights.DEFAULT if (rank == 0 and resume is None) else None
    base = models.efficientnet_b0(weights=weights)
    base.classifier[-1] = nn.Linear(base.classifier[-1].in_features, len(classes))
    base = base.to(device=device, memory_format=torch.channels_last)
    if resume is not None:
        base.load_state_dict(resume["model_state_dict"])
    model = DDP(base, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=True)
    optimizer = torch.optim.AdamW(model.module.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scaler.load_state_dict(resume["scaler_state_dict"])
        start_epoch = int(resume["epoch"]) + 1
        history = resume.get("history", []) if rank == 0 else []
        best_f1 = float(resume.get("best_val_macro_f1", -1.0))
        best_epoch = int(resume.get("best_epoch", 0))
        stale = int(resume.get("stale", 0))
        prior_seconds = float(resume.get("elapsed_seconds", 0.0))
        if stale >= PATIENCE:
            start_epoch = MAX_EPOCHS + 1
            if rank == 0:
                print(f"Early stopping was already reached at epoch {resume['epoch']}; using the saved best checkpoint.", flush=True)
    else:
        start_epoch, history, best_f1, best_epoch, stale, prior_seconds = 1, [], -1.0, 0, 0, 0.0

    if rank == 0:
        print(f"EfficientNet-B0 DDP | GPUs={torch.cuda.get_device_name(0)}, {torch.cuda.get_device_name(1)} | "
              f"global batch={LOCAL_BATCH_SIZE * world} ({LOCAL_BATCH_SIZE}/GPU) | "
              f"train={len(train_ds):,} | val={len(val_ds):,} | classes={len(classes)}", flush=True)
    started = time.time()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        epoch_seed = SEED + epoch
        order = np.random.default_rng(epoch_seed).permutation(len(train_ds))
        rank_indices = order[rank::world].tolist()
        train_loader = DataLoader(Subset(train_ds, rank_indices), batch_size=LOCAL_BATCH_SIZE,
                                  shuffle=False, drop_last=True, num_workers=0, pin_memory=True)
        model.train()
        epoch_loss = torch.zeros((), device=device)
        images_seen = 0
        epoch_started = time.perf_counter()
        progress = tqdm(total=len(train_loader), desc=f"EfficientNet-B0 epoch {epoch}/{MAX_EPOCHS} train",
                        unit="batch", leave=True, mininterval=1.0, disable=(rank != 0))
        for step, (images, labels) in enumerate(train_loader, 1):
            x = gpu_augment_and_normalize(images, device, training=True)
            y = labels.to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.detach()
            images_seen += len(y)
            if rank == 0:
                progress.update(1)
                if step % 20 == 0 or step == len(train_loader):
                    elapsed = max(time.perf_counter() - epoch_started, 1e-6)
                    progress.set_postfix(loss=f"{loss.detach().item():.4f}",
                                         avg_loss=f"{(epoch_loss.detach().item() / step):.4f}",
                                         img_s=f"{images_seen * world / elapsed:.1f}", refresh=True)
        progress.close()
        # Report the mean loss over both process shards, not only rank 0's half.
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        dist.barrier()

        report = [None]
        if rank == 0:
            train_loss = float(epoch_loss.detach().item() / max(len(train_loader) * world, 1))
            val_metrics = evaluate(model.module, val_ds, classes, device)
            row = {"model": "efficientnet_b0", "epoch": epoch, "train_loss": train_loss,
                   **{f"val_{key}": value for key, value in val_metrics.items()},
                   "elapsed_minutes": round((prior_seconds + time.time() - started) / 60.0, 2)}
            history.append(row)
            if val_metrics["macro_f1"] > best_f1:
                best_f1, best_epoch, stale = val_metrics["macro_f1"], epoch, 0
                atomic_save({"architecture": "efficientnet_b0", "model_state_dict": model.module.state_dict(),
                            "classes": classes, "image_size": IMAGE_SIZE, "pretrained": True,
                            "normalization_mean": list(IMAGENET_MEAN), "normalization_std": list(IMAGENET_STD),
                            "epoch": epoch, "val_macro_f1": best_f1, "seed": SEED}, best_path)
            else:
                stale += 1
            elapsed_seconds = prior_seconds + time.time() - started
            atomic_save({"architecture": "efficientnet_b0", "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(),
                        "classes": classes, "epoch": epoch, "best_val_macro_f1": best_f1,
                        "best_epoch": best_epoch, "stale": stale, "history": history,
                        "elapsed_seconds": elapsed_seconds, "resume_config": resume_config}, last_path)
            pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
            print(f"Epoch {epoch}/{MAX_EPOCHS} complete | train_loss={train_loss:.4f} "
                  f"| val_accuracy={val_metrics['accuracy']:.4f} | val_macro_f1={val_metrics['macro_f1']:.4f} "
                  f"| val_macro_precision={val_metrics['macro_precision']:.4f} "
                  f"| val_macro_recall={val_metrics['macro_recall']:.4f} "
                  f"| best_epoch={best_epoch} | elapsed={elapsed_seconds / 60.0:.1f} min", flush=True)
            report[0] = {"best_f1": best_f1, "best_epoch": best_epoch, "stale": stale,
                         "stop": stale >= PATIENCE, "elapsed_seconds": elapsed_seconds}
        dist.broadcast_object_list(report, src=0, device=device)
        best_f1, best_epoch, stale = report[0]["best_f1"], report[0]["best_epoch"], report[0]["stale"]
        prior_seconds = report[0]["elapsed_seconds"]
        if report[0]["stop"]:
            if rank == 0:
                print(f"Early stopping: validation macro-F1 did not improve for {PATIENCE} epochs.", flush=True)
            break

    dist.barrier()
    if rank == 0:
        if not best_path.is_file():
            raise FileNotFoundError("No best checkpoint was saved; training failed before validation.")
        best = torch.load(best_path, map_location=device, weights_only=False)
        model.module.load_state_dict(best["model_state_dict"])
        val_metrics = evaluate(model.module, val_ds, classes, device)
        summary = {
            "model": "efficientnet_b0", "pretrained": True, "validation_images": len(val_ds),
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "best_epoch": best_epoch, "parameters": sum(p.numel() for p in model.module.parameters()),
            "checkpoint_mb": round(best_path.stat().st_size / (1024 ** 2), 2),
            "train_minutes": round(prior_seconds / 60.0, 2), "training_mode": "imagenet_finetune_ddp",
            "used_gpus": world, "local_batch_size": LOCAL_BATCH_SIZE,
            "batch_size": LOCAL_BATCH_SIZE * world, "checkpoint": str(best_path),
        }
        (output_dir / "validation_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Training complete:", json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
