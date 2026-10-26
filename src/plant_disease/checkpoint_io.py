"""Single place that loads checkpoints, so the unsafe pickle path is not scattered."""
from pathlib import Path

import torch


def load_checkpoint(path, map_location="cpu"):
    """Load a checkpoint, preferring the non-executing ``weights_only`` path."""
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        print(f"  weights_only=True failed for {path.name} ({exc}); retrying in compatibility mode", flush=True)
        return torch.load(path, map_location=map_location, weights_only=False)
