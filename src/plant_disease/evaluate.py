import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpoint_io import load_checkpoint
from .data import IMAGENET_MEAN, IMAGENET_STD, SMALLCNN_MEAN, SMALLCNN_STD, PlantVillageImages, load_split
from .metrics import classification_metrics
from .models import build_model


def _clean_state_dict(state):
    if state and all(str(key).startswith("module.") for key in state):
        return {key[7:]: value for key, value in state.items()}
    return state


def evaluate_checkpoint(model_name: str, checkpoint_path: Path, dataset_root: Path, split="validation", batch_size=64):
    if split != "validation":
        raise ValueError("Architecture comparison only evaluates validation. Use Notebook 2 for locked test evaluation.")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path)
    frame, manifest_classes = load_split(Path(dataset_root), split)
    classes = checkpoint.get("classes") if isinstance(checkpoint, dict) else None
    if not classes:
        raise ValueError("Checkpoint must include its ordered classes metadata.")
    if classes != manifest_classes:
        raise ValueError("Checkpoint class order differs from the PlantVillage manifest.")
    saved_architecture = checkpoint.get("architecture")
    if saved_architecture and saved_architecture != model_name:
        raise ValueError(f"Checkpoint architecture is {saved_architecture!r}, not {model_name!r}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, len(classes), pretrained=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state is None:
        raise ValueError("Checkpoint does not contain model weights.")
    model.load_state_dict(_clean_state_dict(state))
    model.to(device).eval()
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(min(2, torch.cuda.device_count()))))

    dataset = PlantVillageImages(Path(dataset_root), frame, classes, model_name)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    y_true, y_pred = [], []
    started = time.perf_counter()
    progress = tqdm(loader, desc=f"{model_name} validation", unit="batch")
    with torch.inference_mode():
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
            progress.set_postfix(val_acc=f"{np.mean(np.asarray(y_true) == np.asarray(y_pred)):.4f}", refresh=False)

    metrics = classification_metrics(y_true, y_pred)
    return {
        "model": model_name,
        "split": split,
        "images": len(frame),
        **{f"val_{key}": value for key, value in metrics.items()},
        "inference_minutes": round((time.perf_counter() - started) / 60, 3),
        "checkpoint": str(checkpoint_path),
    }


def main(model_name: str, argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=f"Calculate {model_name} validation accuracy and macro metrics.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root containing data/manifests and data/raw")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output file")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    summary = evaluate_checkpoint(model_name, args.checkpoint, args.dataset_root, batch_size=args.batch_size)
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
