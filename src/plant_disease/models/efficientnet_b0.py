from torch import nn
from torchvision import models


def build_efficientnet_b0(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model
