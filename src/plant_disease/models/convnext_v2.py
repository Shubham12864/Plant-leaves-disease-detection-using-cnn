"""ConvNeXt-V2 architecture and Dual-Head CNN for agricultural pathology.

Features:
1. Pure 2D Convolutional architecture strictly complying with "using CNN".
2. 7x7 depthwise convolutions for large spatial context (leaf macro-structure).
3. Global Response Normalization (GRN) to prevent feature collapse on fine-grained lesion spots.
4. Dual-Head architecture: Decoupled Crop identification and Disease pathology heads.
"""
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GRN(nn.Module):
    """Global Response Normalization layer (ConvNeXt-V2).

    Normalizes channel activations across spatial dimensions, increasing
    feature diversity and contrast between fine lesion textures and leaf surfaces.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, H, W, C) in channels-last format
        # Compute spatial L2 norm in FP32 with interior epsilon for numerical stability and AMP safety
        gx = torch.sqrt(torch.sum(x.float() ** 2, dim=(1, 2), keepdim=True) + self.eps).type_as(x)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt-V2 Inverted Bottleneck Block with GRN."""
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_res = x
        x = self.dwconv(x)
        # Permute to channels-last for LayerNorm and Linear: (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        # Permute back: (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return input_res + x


class ConvNeXtV2(nn.Module):
    """Full ConvNeXt-V2 Backbone and Classifier."""
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 38,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
    ):
        super().__init__()
        self.depths = depths
        self.dims = dims

        # Stem: 4x4 downsample with stride 4
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)

        # 3 Downsample layers between stages
        for i in range(3):
            downsample = nn.Sequential(
                nn.LayerNorm(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample)

        # 4 Stages of ConvNeXt-V2 blocks
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtV2Block(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through stem
        x = self.downsample_layers[0][0](x)
        x = x.permute(0, 2, 3, 1)
        x = self.downsample_layers[0][1](x)
        x = x.permute(0, 3, 1, 2)
        x = self.stages[0](x)

        # Pass through remaining stages
        for i in range(1, 4):
            x = x.permute(0, 2, 3, 1)
            x = self.downsample_layers[i][0](x)
            x = x.permute(0, 3, 1, 2)
            x = self.downsample_layers[i][1](x)
            x = self.stages[i](x)

        # Global average pool: (N, C, H, W) -> (N, C)
        return x.mean([-2, -1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        norm_feat = self.norm(features)
        return self.head(norm_feat)


class DualHeadCNN(nn.Module):
    """Dual-Head CNN for Hierarchical Plant & Disease Diagnosis.

    Head 1 outputs logits for Crop Species (e.g. Tomato, Potato, Apple).
    Head 2 outputs multi-label logits for Pathology (Healthy, Early Blight, Late Blight, etc.).
    """
    def __init__(self, backbone: nn.Module, feature_dim: int, num_crops: int, num_diseases: int):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(feature_dim, eps=1e-6)
        self.crop_head = nn.Linear(feature_dim, num_crops)
        self.disease_head = nn.Linear(feature_dim, num_diseases)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.backbone, "forward_features"):
            feat = self.backbone.forward_features(x)
        else:
            feat = self.backbone(x)
        feat_norm = self.norm(feat)
        crop_logits = self.crop_head(feat_norm)
        disease_logits = self.disease_head(feat_norm)
        return crop_logits, disease_logits


def build_convnext_v2_tiny(num_classes: int = 38, pretrained: bool = False) -> nn.Module:
    """Build ConvNeXt-V2 Tiny model.

    If pretrained is requested, initializes stem and stages from torchvision
    ConvNeXt-Tiny weights while adding Global Response Normalization (GRN).
    """
    model = ConvNeXtV2(
        in_chans=3,
        num_classes=num_classes,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768]
    )
    if pretrained:
        # Load torchvision ConvNeXt-Tiny weights into compatible layers
        tv_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        tv_dict = tv_model.state_dict()
        model_dict = model.state_dict()
        # Transfer matching weights (dwconv, pwconv, norm)
        matched = {}
        for k, v in tv_dict.items():
            # Skip classifier head
            if "classifier" in k:
                continue
            # Map downsample stem
            if k.startswith("features.0.0."):
                m_k = k.replace("features.0.0.", "downsample_layers.0.0.")
            elif k.startswith("features.0.1."):
                m_k = k.replace("features.0.1.", "downsample_layers.0.1.")
            elif k.startswith("features.2.0."):
                m_k = k.replace("features.2.0.", "downsample_layers.1.0.")
            elif k.startswith("features.2.1."):
                m_k = k.replace("features.2.1.", "downsample_layers.1.1.")
            elif k.startswith("features.4.0."):
                m_k = k.replace("features.4.0.", "downsample_layers.2.0.")
            elif k.startswith("features.4.1."):
                m_k = k.replace("features.4.1.", "downsample_layers.2.1.")
            elif k.startswith("features.6.0."):
                m_k = k.replace("features.6.0.", "downsample_layers.3.0.")
            elif k.startswith("features.6.1."):
                m_k = k.replace("features.6.1.", "downsample_layers.3.1.")
            # Map block stages
            elif k.startswith("features.1."):
                m_k = k.replace("features.1.", "stages.0.")
            elif k.startswith("features.3."):
                m_k = k.replace("features.3.", "stages.1.")
            elif k.startswith("features.5."):
                m_k = k.replace("features.5.", "stages.2.")
            elif k.startswith("features.7."):
                m_k = k.replace("features.7.", "stages.3.")
            else:
                m_k = k

            # Map torchvision sequential block submodules to ConvNeXtV2Block named attributes
            m_k = (
                m_k.replace(".block.0.", ".dwconv.")
                .replace(".block.2.", ".norm.")
                .replace(".block.3.", ".pwconv1.")
                .replace(".block.5.", ".pwconv2.")
            )

            if m_k in model_dict and model_dict[m_k].shape == v.shape:
                matched[m_k] = v

        model_dict.update(matched)
        model.load_state_dict(model_dict)

    return model
