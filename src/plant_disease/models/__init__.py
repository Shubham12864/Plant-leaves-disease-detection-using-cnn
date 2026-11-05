"""Model definitions used by the PlantVillage project."""

from torch import nn

from .convnext import build_convnext_tiny
from .convnext_v2 import build_convnext_v2_tiny, ConvNeXtV2, DualHeadCNN
from .unified_gated_cnn import UnifiedGatedPlantCNN, build_unified_plant_cnn
from .efficientnet_b0 import build_efficientnet_b0
from .resnet18 import build_resnet18
from .smallcnn import SmallCNN

# Core baseline architectures compared in the initial lab-to-field sweep
MODEL_NAMES = ("smallcnn", "resnet18", "efficientnet_b0", "convnext_tiny")

# Extended research architectures
EXTENDED_MODEL_NAMES = ("convnext_v2_tiny", "unified_gated_cnn")
ALL_MODEL_NAMES = MODEL_NAMES + EXTENDED_MODEL_NAMES


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
    if name == "convnext_v2_tiny":
        return build_convnext_v2_tiny(num_classes, pretrained=pretrained)
    if name == "unified_gated_cnn":
        return build_unified_plant_cnn(num_crops=14, num_diseases=num_classes, single_output=True)
    raise ValueError(f"Unknown model {name!r}. Choose from: {', '.join(ALL_MODEL_NAMES)}")
