"""Model definitions used by the PlantVillage project."""

from torch import nn

from .convnext import build_convnext_tiny
from .efficientnet_b0 import build_efficientnet_b0
from .resnet18 import build_resnet18
from .smallcnn import SmallCNN

MODEL_NAMES = ("smallcnn", "resnet18", "efficientnet_b0", "convnext_tiny")


def build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """Build one supported architecture with a project-sized output layer."""
    if name == "smallcnn":
        return SmallCNN(num_classes)
    if name == "resnet18":
        return build_resnet18(num_classes, pretrained=pretrained)
    if name == "efficientnet_b0":
        return build_efficientnet_b0(num_classes, pretrained=pretrained)
    if name == "convnext_tiny":
        return build_convnext_tiny(num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model {name!r}. Choose from: {', '.join(MODEL_NAMES)}")
