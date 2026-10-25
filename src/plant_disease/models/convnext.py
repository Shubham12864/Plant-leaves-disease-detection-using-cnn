"""ConvNeXt-Tiny, the fourth architecture in the comparison.

Included for one reason: the other three are a scratch CNN, a 2015-era residual
net and an efficient NAS-style net. ConvNeXt is a different design family
(transformer-inspired macro-blocks, 7x7 depthwise convolutions, LayerNorm), so a
finding that holds across all four is a property of the method rather than of one
architecture. It is not expected to win on PlantVillage, which is already
saturated; see docs/RESEARCH_CLAIM.md.

Torchvision's ConvNeXt classifier head is a Sequential of
LayerNorm -> Flatten -> Linear, so the output layer is ``classifier[2]``.
"""
from torch import nn
from torchvision import models


def build_convnext_tiny(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model
