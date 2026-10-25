#!/usr/bin/env python3
"""Fine-tune a supported CNN on PlantVillage with two-GPU DDP.

This is the v2 training engine. It replaces the pilot
``train_efficientnet_b0_ddp.py`` and is architecture-agnostic, so the
EfficientNet-B0 run and the ResNet18 run share one auditable code path.

Why v2 exists (measured problems in the pilot run)
--------------------------------------------------
* The pilot used ``num_workers=0`` on a 5.5 GB memory-mapped cache. Every batch
  page-faulted in the main process, so the two T4s idled while data loaded.
  Per-epoch time grew 1.7 -> 15.65 min as the page cache evicted. v2 uses
  worker processes with prefetch, which is the standard fix and should land
  around 1 min/epoch.
* The pilot set ``cudnn.benchmark = False`` and used a global batch of 32.
  v2 enables benchmark (input shape is fixed) and defaults to 64 per GPU.
* The loss was plain, unweighted cross-entropy on a 38.4x imbalanced training
  set (Orange HLB 3,842 vs Potato healthy 100). v2 supports class-balanced
  cross-entropy so macro-F1, the model-selection metric, is not starved by
  rare classes.
* The pilot saved only headline metrics. v2 also writes per-class
  precision/recall/F1/support, the confusion matrix and calibration
  diagnostics, which the report and the guide review need anyway.

What v2 adds for accuracy
-------------------------
Class-balanced cross-entropy, label smoothing, cosine schedule with linear
warmup, gradient clipping, and an EMA copy of the weights that is evaluated and
kept when it beats the raw weights.

What v2 adds for safety and reproducibility
-------------------------------------------
``weights_only=True`` checkpoint loading with an explicit fallback, and a
dataset fingerprint stored in the checkpoint so a resumed run cannot silently
continue against a different split.

Usage
-----
    torchrun --standalone --nproc_per_node=2 scripts/train_ddp.py \
        --arch efficientnet_b0 \
        --train-images /path/train_x.npy --train-labels /path/train_y.npy \
        --val-images /path/val_x.npy --val-labels /path/val_y.npy \
        --classes-json /path/classes.json --output-dir /path/out

Add ``--dry-run`` to execute three optimizer steps and exit, which is the
cheap way to prove a Kaggle session is wired up before committing GPU hours.
"""
from __future__ import annotations

import os

# Keep BLAS single-threaded per rank: DDP already saturates the cores and
# oversubscription is a common cause of stalled steps.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import contextlib
import copy
import hashlib
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
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

try:  # torchvision is a hard dependency in practice; import lazily for CLI help
    import torchvision
    from torchvision import models
except Exception as exc:  # pragma: no cover - only hit in broken installs
    torchvision = None
    models = None
    _TV_IMPORT_ERROR = exc

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SMALLCNN_MEAN = (0.5, 0.5, 0.5)
SMALLCNN_STD = (0.5, 0.5, 0.5)

ARCHES = ("efficientnet_b0", "resnet18")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class CachedImages(Dataset):
    """Read one uint8 image at a time from a disk-backed NPY memory map."""

    def __init__(self, image_file, labels_file, image_size: int):
        self.images = np.load(image_file, mmap_mode="r", allow_pickle=False)
        self.labels = np.load(labels_file, mmap_mode="r", allow_pickle=False)
        expected = (image_size, image_size, 3)
        if self.images.ndim != 4 or tuple(self.images.shape[1:]) != expected:
            raise ValueError(f"Bad cached image shape: {self.images.shape}, expected (*, {expected})")
        if self.images.dtype != np.uint8:
            raise ValueError(f"Image cache must be uint8, found {self.images.dtype}")
        if len(self.images) != len(self.labels):
            raise ValueError("Image cache and label cache have different lengths")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        # copy() detaches the row from the memory map so the loader can batch it
        image = torch.from_numpy(self.images[index].copy())
        return image, int(self.labels[index])


def build_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    if models is None:  # pragma: no cover
        raise RuntimeError(f"torchvision import failed: {_TV_IMPORT_ERROR}")
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch {arch!r}; choose from {ARCHES}")
    return model


def dataset_fingerprint(train_labels, val_labels, classes, image_size: int) -> str:
    """Stable identifier for the exact split a checkpoint was trained on."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(train_labels).tobytes())
    digest.update(np.ascontiguousarray(val_labels).tobytes())
    digest.update(json.dumps(classes, separators=(",", ":")).encode("utf-8"))
    digest.update(str(image_size).encode("utf-8"))
    return digest.hexdigest()


def block_shuffled_indices(n: int, rng: np.random.Generator, block: int = 2048) -> np.ndarray:
    """Permute block order, then shuffle inside each block.

    Reading in blocks keeps the page cache local instead of jumping across the
    whole 5 GB map every batch, which was the pilot's failure mode.
    """
    order = np.arange(n)
    starts = np.arange(0, n, block)
    rng.shuffle(starts)
    out = []
    for start in starts:
        chunk = order[start:start + block]
        rng.shuffle(chunk)
        out.append(chunk)
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# Augmentation and normalisation (all on GPU)
# --------------------------------------------------------------------------- #
def amp_context(use_cuda: bool):
    """FP16 autocast on the accelerator, no-op on CPU (CPU autocast is slower)."""
    if use_cuda:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def gpu_augment_and_normalize(image_u8, device, training: bool, mean, std, rotation_deg: float = 12.0):
    """NHWC uint8 -> normalised NCHW float, with augmentation applied on the GPU.

    Augmentation runs on the accelerator so no CPU worker is the bottleneck and
    the worker processes only have to memcpy bytes out of the memory map.
    """
    x = image_u8.to(device=device, dtype=torch.float32, non_blocking=True)
    x = x.permute(0, 3, 1, 2).div_(255.0).contiguous(memory_format=torch.channels_last)
    if training:
        batch = x.shape[0]
        # horizontal flip
        flip = torch.rand(batch, device=device) < 0.5
        x = torch.where(flip.view(batch, 1, 1, 1), x.flip(3), x)
        # small rotation
        if rotation_deg > 0:
            angle = (torch.rand(batch, device=device) * 2.0 - 1.0) * math.radians(rotation_deg)
            cosine, sine = angle.cos(), angle.sin()
            theta = torch.zeros(batch, 2, 3, device=device)
            theta[:, 0, 0], theta[:, 0, 1] = cosine, -sine
            theta[:, 1, 0], theta[:, 1, 1] = sine, cosine
            grid = F.affine_grid(theta, x.shape, align_corners=False)
            x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        # brightness and per-channel contrast jitter
        brightness = torch.rand(batch, 1, 1, 1, device=device) * 0.24 + 0.88
        x = (x * brightness).clamp_(0.0, 1.0)
        gain = torch.rand(batch, 3, 1, 1, device=device) * 0.16 + 0.92
        x = (x * gain).clamp_(0.0, 1.0)
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return ((x - mean_t) / std_t).contiguous(memory_format=torch.channels_last)


# --------------------------------------------------------------------------- #
# Metrics and calibration
# --------------------------------------------------------------------------- #
def expected_calibration_error(probabilities: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Binned |accuracy - confidence|, weighted by bin population."""
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def brier_score(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and one-hot labels."""
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def score_predictions(y_true, y_pred, probabilities=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[0]),
        "macro_recall": float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[1]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if probabilities is not None:
        scores = probabilities.astype(np.float64)
        labels = np.arange(scores.shape[1])
        metrics["top3_accuracy"] = float(top_k_accuracy_score(y_true, scores, k=3, labels=labels))
        metrics["ece"] = expected_calibration_error(scores, y_true)
        metrics["brier"] = brier_score(scores, y_true)
        metrics["mean_confidence"] = float(scores.max(axis=1).mean())
        metrics["frac_below_0_5"] = float((scores.max(axis=1) < 0.5).mean())
    return metrics


def classification_table(y_true, y_pred, classes) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(classes)), zero_division=0
    )
    return pd.DataFrame(
        {
            "class": classes,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #
class ModelEma:
    """Exponential moving average of model weights, kept on the evaluating rank."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        # Ramp the effective decay early so the EMA is not anchored to the
        # random initialisation for the first few hundred steps.
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for ema_parameter, parameter in zip(self.module.parameters(), model.parameters()):
            ema_parameter.mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
        for ema_buffer, buffer in zip(self.module.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, dataset, device, mean, std, batch_size, workers, use_cuda=True) -> tuple:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=use_cuda,
        persistent_workers=workers > 0,
    )
    y_true, y_pred, probability_rows = [], [], []
    started = time.perf_counter()
    progress = tqdm(loader, desc="validation", unit="batch", leave=True, mininterval=1.0, disable=None)
    for images, labels in progress:
        x = gpu_augment_and_normalize(images, device, training=False, mean=mean, std=std)
        with amp_context(use_cuda):
            logits = model(x)
        probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(probabilities.argmax(axis=1).tolist())
        probability_rows.append(probabilities)
        seen = len(y_true)
        progress.set_postfix(
            val_accuracy=f"{np.mean(np.asarray(y_true) == np.asarray(y_pred)):.4f}",
            img_s=f"{seen / max(time.perf_counter() - started, 1e-6):.1f}",
            refresh=False,
        )
    probabilities = np.concatenate(probability_rows, axis=0)
    return score_predictions(y_true, y_pred, probabilities), np.asarray(y_true), np.asarray(y_pred), probabilities


# --------------------------------------------------------------------------- #
# Checkpoint IO
# --------------------------------------------------------------------------- #
def atomic_save(payload, path: Path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def safe_load(path: Path, map_location):
    """Load a checkpoint, preferring the non-executing ``weights_only`` path."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:  # older torch or an exotic payload
        print(f"  weights_only=True failed for {path.name} ({exc}); retrying in compatibility mode", flush=True)
        return torch.load(path, map_location=map_location, weights_only=False)


def clean_state_dict(state):
    if state and all(str(key).startswith("module.") for key in state):
        return {key[7:]: value for key, value in state.items()}
    return state


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--val-labels", required=True)
    parser.add_argument("--classes-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arch", default="efficientnet_b0", choices=ARCHES)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--local-batch-size", type=int, default=64, help="per-GPU batch; global = this x world size")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-factor", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-fraction", type=float, default=0.10)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--class-weight-alpha", type=float, default=0.5,
                        help="0 disables class-balanced loss; 1 is full inverse frequency")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="0 disables EMA")
    parser.add_argument("--rotation-deg", type=float, default=12.0)
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers per rank")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--compile", action="store_true", help="torch.compile the model (slower first step)")
    parser.add_argument("--shuffle-mode", default="blocked", choices=("random", "blocked"))
    parser.add_argument("--dry-run", action="store_true", help="run 3 optimizer steps then exit")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"),
                        help="auto uses CUDA when present; cpu exists for local smoke tests")
    # Escape hatch for environments where torchrun's TCP store is unavailable
    # (some Windows CPU wheels are built without libuv). A file-based store keeps
    # the same code path reachable without the launcher.
    parser.add_argument("--init-method", default="env://",
                        help="URL for process-group init; non-env values require --rank/--world-size")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--no-dist", action="store_true",
                        help="single process, no process group; for local code checks only")
    args = parser.parse_args()

    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. In Kaggle select Accelerator -> GPU T4 x2.")
    if not use_cuda and args.device == "auto":
        raise RuntimeError(
            "No CUDA device. Pass --device cpu to run the engine on CPU (slow, intended for a "
            "small local check) or select GPU T4 x2 in Kaggle."
        )

    backend = "nccl" if use_cuda else "gloo"
    distributed = not args.no_dist
    if not distributed:
        rank, world, local_rank = 0, 1, 0
    elif args.init_method == "env://":
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        if args.rank is None or args.world_size is None:
            raise ValueError("--init-method other than env:// requires --rank and --world-size")
        dist.init_process_group(
            backend=backend, init_method=args.init_method, rank=args.rank, world_size=args.world_size
        )
        rank, world, local_rank = args.rank, args.world_size, args.rank

    def synchronise():
        """Barrier when a process group exists, otherwise a no-op."""
        if distributed:
            dist.barrier()

    def unwrap(module: nn.Module) -> nn.Module:
        return module.module if isinstance(module, DDP) else module
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    torch.set_num_threads(1)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed + rank)
        # Fixed 224x224 input means cuDNN can pick and keep the fastest algorithm.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    classes = json.loads(Path(args.classes_json).read_text(encoding="utf-8"))
    train_ds = CachedImages(args.train_images, args.train_labels, args.image_size)
    val_ds = CachedImages(args.val_images, args.val_labels, args.image_size)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Both train and validation caches must contain images")
    if int(np.asarray(train_ds.labels).max()) >= len(classes) or int(np.asarray(val_ds.labels).max()) >= len(classes):
        raise ValueError("A cached label index is out of range for the class list")

    fingerprint = dataset_fingerprint(
        np.asarray(train_ds.labels), np.asarray(val_ds.labels), classes, args.image_size
    )

    mean, std = (SMALLCNN_MEAN, SMALLCNN_STD) if args.arch == "smallcnn" else (IMAGENET_MEAN, IMAGENET_STD)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / f"{args.arch}_best.pt"
    last_path = output_dir / f"{args.arch}_last.pt"

    resume_config = {
        "architecture": args.arch,
        "classes": classes,
        "seed": args.seed,
        "image_size": args.image_size,
        "local_batch_size": args.local_batch_size,
        "global_batch_size": args.local_batch_size * world,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "class_weight_alpha": args.class_weight_alpha,
        "ema_decay": args.ema_decay,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "dataset_fingerprint": fingerprint,
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
    }

    resume = None
    if last_path.is_file() and not args.dry_run:
        resume = safe_load(last_path, "cpu")
        stored = resume.get("resume_config", {})
        if stored.get("dataset_fingerprint") not in (None, fingerprint):
            raise RuntimeError(
                "The resume checkpoint was trained on a different split "
                f"({stored.get('dataset_fingerprint')} != {fingerprint}). "
                "Move the old run aside or use a fresh --output-dir."
            )
        if {k: v for k, v in stored.items() if k != "dataset_fingerprint"} != {
            k: v for k, v in resume_config.items() if k != "dataset_fingerprint"
        }:
            raise RuntimeError("Resume checkpoint settings differ from this run. Use a fresh --output-dir.")
        if rank == 0:
            print(f"Resuming from completed epoch {resume['epoch']}.", flush=True)

    # Class-balanced loss, softened by --class-weight-alpha. alpha=0 is plain CE.
    train_labels = np.asarray(train_ds.labels)
    class_counts = np.bincount(train_labels, minlength=len(classes)).astype(np.float64)
    if args.class_weight_alpha > 0:
        present = class_counts > 0
        weights = np.ones(len(classes), dtype=np.float64)
        weights[present] = (class_counts.sum() / (len(classes) * class_counts[present])) ** args.class_weight_alpha
        weights = weights / weights[present].mean()
    else:
        weights = np.ones(len(classes), dtype=np.float64)
    loss_weight = torch.tensor(weights, dtype=torch.float32, device=device)

    pretrained = rank == 0 and resume is None
    base = build_model(args.arch, len(classes), pretrained=pretrained)
    base = base.to(device=device, memory_format=torch.channels_last)
    if resume is not None:
        base.load_state_dict(clean_state_dict(resume["model_state_dict"]))
    # Broadcast rank 0's weights *and* buffers so every rank starts identical.
    # Buffers matter for transfer learning: batch-norm running statistics on the
    # non-zero ranks would otherwise stay at their default (mean 0, var 1).
    if distributed:
        for parameter in base.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in base.buffers():
            dist.broadcast(buffer.data, src=0)

    # Build the EMA from the plain module, before torch.compile and DDP wrap it.
    ema = ModelEma(base, decay=args.ema_decay) if (args.ema_decay > 0 and rank == 0) else None

    if args.compile:
        try:
            base = torch.compile(base)
            if rank == 0:
                print("torch.compile enabled.", flush=True)
        except Exception as exc:
            print(f"torch.compile unavailable, continuing eagerly: {exc}", flush=True)

    model = (
        DDP(base, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=True, find_unused_parameters=False)
        if distributed else base
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # A disabled scaler still needs a valid device type, so name CPU explicitly
    # when there is no accelerator rather than constructing a CUDA object.
    try:
        scaler = torch.amp.GradScaler("cuda" if use_cuda else "cpu", enabled=use_cuda)
    except (TypeError, RuntimeError):  # torch < 2.3 positional signature
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    steps_per_epoch = max(len(train_ds) // (args.local_batch_size * world), 1)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(int(total_steps * args.warmup_fraction), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return args.min_lr_factor + (1.0 - args.min_lr_factor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch, history, best_f1, best_epoch, stale, prior_seconds = 1, [], -1.0, 0, 0, 0.0
    global_step = 0
    if resume is not None:
        loadable = {k: v for k, v in resume.items() if k not in {"model_state_dict", "resume_config"}}
        optimizer.load_state_dict(loadable["optimizer_state_dict"])
        scaler.load_state_dict(loadable["scaler_state_dict"])
        if "scheduler_state_dict" in loadable:
            scheduler.load_state_dict(loadable["scheduler_state_dict"])
        if ema is not None and loadable.get("ema_state_dict"):
            ema.module.load_state_dict(loadable["ema_state_dict"])
            ema.updates = int(loadable.get("ema_updates", 0))
        start_epoch = int(resume["epoch"]) + 1
        history = resume.get("history", []) if rank == 0 else []
        best_f1 = float(resume.get("best_val_macro_f1", -1.0))
        best_epoch = int(resume.get("best_epoch", 0))
        stale = int(resume.get("stale", 0))
        prior_seconds = float(resume.get("elapsed_seconds", 0.0))
        global_step = int(resume.get("global_step", 0))
        if stale >= args.patience:
            start_epoch = args.epochs + 1
            if rank == 0:
                print("Early stopping was already reached; using the saved best checkpoint.", flush=True)

    if rank == 0:
        gpu_label = (
            ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
            if use_cuda else "cpu"
        )
        print(
            f"{args.arch} | devices={gpu_label} | "
            f"global batch={args.local_batch_size * world} ({args.local_batch_size}/GPU) | "
            f"workers/rank={args.workers} | lr={args.lr} | train={len(train_ds):,} | "
            f"val={len(val_ds):,} | classes={len(classes)} | steps/epoch={steps_per_epoch}",
            flush=True,
        )
        print(f"Dataset fingerprint: {fingerprint[:16]}...", flush=True)

    started = time.time()
    criterion = nn.CrossEntropyLoss(weight=loss_weight, label_smoothing=args.label_smoothing)

    for epoch in range(start_epoch, args.epochs + 1):
        rng = np.random.default_rng(args.seed + epoch)
        if args.shuffle_mode == "blocked":
            order = block_shuffled_indices(len(train_ds), rng)
        else:
            order = rng.permutation(len(train_ds))
        rank_indices = order[rank::world].tolist()

        train_loader = DataLoader(
            Subset(train_ds, rank_indices),
            batch_size=args.local_batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=args.workers,
            pin_memory=use_cuda,
            persistent_workers=args.workers > 0,
            prefetch_factor=4 if args.workers > 0 else None,
        )
        model.train()
        epoch_loss = torch.zeros((), device=device)
        images_seen = 0
        epoch_started = time.perf_counter()
        progress = tqdm(
            total=len(train_loader),
            desc=f"{args.arch} epoch {epoch}/{args.epochs} train",
            unit="batch",
            leave=True,
            mininterval=1.0,
            disable=(rank != 0),
        )
        for step, (images, labels) in enumerate(train_loader, 1):
            x = gpu_augment_and_normalize(
                images, device, training=True, mean=mean, std=std, rotation_deg=args.rotation_deg
            )
            y = labels.to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(use_cuda):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            if ema is not None:
                ema.update(unwrap(model)._orig_mod if hasattr(unwrap(model), "_orig_mod") else unwrap(model))
            epoch_loss += loss.detach()
            images_seen += len(y)
            if rank == 0:
                progress.update(1)
                if step % 20 == 0 or step == len(train_loader):
                    elapsed = max(time.perf_counter() - epoch_started, 1e-6)
                    progress.set_postfix(
                        loss=f"{loss.detach().item():.4f}",
                        avg_loss=f"{(epoch_loss.detach().item() / step):.4f}",
                        img_s=f"{images_seen * world / elapsed:.1f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        refresh=True,
                    )
            if args.dry_run and step >= 3:
                break
        progress.close()
        if distributed:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        synchronise()

        if args.dry_run:
            if rank == 0:
                print("Dry run complete: 3 steps executed, no checkpoint written.", flush=True)
            synchronise()
            if distributed:
                dist.destroy_process_group()
            return

        report = [None]
        if rank == 0:
            train_loss = float(epoch_loss.detach().item() / max(len(train_loader) * world, 1))

            raw_metrics, y_true, y_pred, _ = evaluate(
                unwrap(model), val_ds, device, mean, std, args.eval_batch_size, args.workers, use_cuda
            )
            chosen_metrics, chosen_source = raw_metrics, "raw"
            if ema is not None:
                ema_metrics, ema_true, ema_pred, _ = evaluate(
                    ema.module, val_ds, device, mean, std, args.eval_batch_size, args.workers, use_cuda
                )
                if ema_metrics["macro_f1"] > raw_metrics["macro_f1"]:
                    chosen_metrics, chosen_source = ema_metrics, "ema"
                    y_true, y_pred = ema_true, ema_pred

            row = {
                "model": args.arch,
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{key}": value for key, value in raw_metrics.items()},
                f"val_macro_f1_{chosen_source}": chosen_metrics["macro_f1"],
                "weights_used": chosen_source,
                "elapsed_minutes": round((prior_seconds + time.time() - started) / 60.0, 2),
            }
            history.append(row)

            if chosen_metrics["macro_f1"] > best_f1:
                best_f1, best_epoch, stale = chosen_metrics["macro_f1"], epoch, 0
                state = ema.module.state_dict() if chosen_source == "ema" else unwrap(model).state_dict()
                atomic_save(
                    {
                        "architecture": args.arch,
                        "model_state_dict": clean_state_dict(state),
                        "classes": classes,
                        "image_size": args.image_size,
                        "pretrained": True,
                        "normalization_mean": list(mean),
                        "normalization_std": list(std),
                        "epoch": epoch,
                        "val_macro_f1": best_f1,
                        "weights_are_ema": chosen_source == "ema",
                        "seed": args.seed,
                        "dataset_fingerprint": fingerprint,
                        "n_train": len(train_ds),
                        "n_val": len(val_ds),
                    },
                    best_path,
                )
                pd.DataFrame(classification_table(y_true, y_pred, classes)).to_csv(
                    output_dir / "validation_classification_report.csv", index=False
                )
                pd.DataFrame(
                    confusion_matrix(y_true, y_pred, labels=np.arange(len(classes))),
                    index=classes,
                    columns=classes,
                ).to_csv(output_dir / "validation_confusion_matrix.csv")
                (output_dir / "validation_metrics.json").write_text(
                    json.dumps({**chosen_metrics, "weights_used": chosen_source, "epoch": epoch}, indent=2),
                    encoding="utf-8",
                )
            else:
                stale += 1

            elapsed_seconds = prior_seconds + time.time() - started
            atomic_save(
                {
                    "architecture": args.arch,
                    "model_state_dict": clean_state_dict(unwrap(model).state_dict()),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "ema_state_dict": clean_state_dict(ema.module.state_dict()) if ema is not None else None,
                    "ema_updates": ema.updates if ema is not None else 0,
                    "classes": classes,
                    "epoch": epoch,
                    "best_val_macro_f1": best_f1,
                    "best_epoch": best_epoch,
                    "stale": stale,
                    "history": history,
                    "global_step": global_step,
                    "elapsed_seconds": elapsed_seconds,
                    "dataset_fingerprint": fingerprint,
                    "resume_config": resume_config,
                },
                last_path,
            )
            pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
            print(
                f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} | "
                f"val_acc={raw_metrics['accuracy']:.4f} | val_macro_f1={raw_metrics['macro_f1']:.4f} | "
                f"val_f1_used={chosen_metrics['macro_f1']:.4f} ({chosen_source}) | "
                f"ece={raw_metrics.get('ece', float('nan')):.4f} | best_epoch={best_epoch} | "
                f"elapsed={elapsed_seconds / 60.0:.2f} min",
                flush=True,
            )
            report[0] = {
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "stale": stale,
                "stop": stale >= args.patience,
                "elapsed_seconds": elapsed_seconds,
            }
        if distributed:
            dist.broadcast_object_list(report, src=0, device=device)
        best_f1, best_epoch, stale = report[0]["best_f1"], report[0]["best_epoch"], report[0]["stale"]
        prior_seconds = report[0]["elapsed_seconds"]
        if report[0]["stop"]:
            if rank == 0:
                print(f"Early stopping: validation macro-F1 did not improve for {args.patience} epochs.", flush=True)
            break

    synchronise()
    if rank == 0:
        if not best_path.is_file():
            raise FileNotFoundError("No best checkpoint was written; training never reached a validation pass.")
        best = safe_load(best_path, device)
        final = build_model(args.arch, len(classes), pretrained=False).to(device)
        final.load_state_dict(clean_state_dict(best["model_state_dict"]))
        final.eval()
        final_metrics, y_true, y_pred, _ = evaluate(final, val_ds, device, mean, std, args.eval_batch_size, args.workers, use_cuda)
        summary = {
            "model": args.arch,
            "pretrained": True,
            "validation_images": len(val_ds),
            "train_images": len(train_ds),
            **{f"val_{key}": value for key, value in final_metrics.items()},
            "best_epoch": best_epoch,
            "weights_are_ema": bool(best.get("weights_are_ema")),
            "parameters": sum(p.numel() for p in final.parameters()),
            "checkpoint_mb": round(best_path.stat().st_size / (1024 ** 2), 2),
            "train_minutes": round(prior_seconds / 60.0, 2),
            "training_mode": "imagenet_finetune_ddp_v2",
            "used_gpus": world,
            "local_batch_size": args.local_batch_size,
            "batch_size": args.local_batch_size * world,
            "workers_per_rank": args.workers,
            "learning_rate": args.lr,
            "label_smoothing": args.label_smoothing,
            "class_weight_alpha": args.class_weight_alpha,
            "ema_decay": args.ema_decay,
            "checkpoint": str(best_path),
            "dataset_fingerprint": fingerprint,
        }
        (output_dir / "validation_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Training complete:", json.dumps(summary, indent=2), flush=True)
    synchronise()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
