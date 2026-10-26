"""Checkpoint contract: every saved model must load and agree with the class order."""
from pathlib import Path

import pytest
import torch

from plant_disease.checkpoint_io import load_checkpoint
from plant_disease.models import build_model

# 38-class heads, not ImageNet's 1000. ResNet18 replaces a 1000-way fc
# (513,000 params) with a 38-way one (19,494), which is why this is 11,196,006
# and not the 11,689,512 quoted for the ImageNet model.
EXPECTED = {
    "efficientnet_b0": 4056226,
    "resnet18": 11196006,
    "smallcnn": 398662,
    "convnext_tiny": 27849350,
}

CHECKPOINTS = {
    "efficientnet_b0": "checkpoints/efficientnet_b0/efficientnet_b0_best.pt",
    "resnet18": "checkpoints/resnet18/resnet18_best.pt",
    "smallcnn": "checkpoints/smallcnn/smallcnn_best.pt",
    "convnext_tiny": "checkpoints/convnext_tiny/convnext_tiny_best.pt",
}


def _efficientnet_path(repo: Path) -> Path:
    path = repo / CHECKPOINTS["efficientnet_b0"]
    if not path.is_file():
        pytest.skip(f"{CHECKPOINTS['efficientnet_b0']} not present (weights are not kept in git)")
    return path


@pytest.mark.parametrize("arch,relative", sorted(CHECKPOINTS.items()))
def test_checkpoint_loads_strictly_and_matches_parameter_count(repo: Path, arch: str, relative: str):
    path = repo / relative
    if not path.is_file():
        pytest.skip(f"{relative} not present")
    blob = load_checkpoint(path)
    assert "model_state_dict" in blob, sorted(blob)
    assert len(blob["classes"]) == 38

    model = build_model(arch, 38, pretrained=False)
    state = {k.replace("module.", "", 1): v for k, v in blob["model_state_dict"].items()}
    model.load_state_dict(state, strict=True)
    assert sum(p.numel() for p in model.parameters()) == EXPECTED[arch]


def test_class_order_matches_the_manifest(repo: Path, manifests: Path):
    import pandas as pd

    blob = load_checkpoint(_efficientnet_path(repo))
    dev = pd.read_csv(manifests / "plantvillage_development.csv")
    expected = sorted(set(dev.label))
    assert list(blob["classes"]) == expected, "checkpoint class order drifted from the manifest"


def test_checkpoint_records_its_normalisation(repo: Path):
    blob = load_checkpoint(_efficientnet_path(repo))
    assert blob["image_size"] == 224
    assert blob["normalization_mean"] == [0.485, 0.456, 0.406]
    assert blob["normalization_std"] == [0.229, 0.224, 0.225]
