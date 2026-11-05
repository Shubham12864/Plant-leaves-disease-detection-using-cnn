"""Unified Gated Plant CNN Architecture (End-to-End Pure 2D CNN).

Unifies spatial leaf filtering, lesion saliency gating, and hierarchical dual-head
classification into a single differentiable neural network.

Replaces discrete multi-tool pipelines (YOLO cropping + classifier) with an
end-to-end architecture:
1. Multi-Scale Pure 2D CNN Backbone (ConvNeXt-V2 with GRN).
2. Differentiable Spatial Saliency Gating Module (filters background soil/weeds).
3. Cross-Task Conditioned Dual Heads (Crop Species + Disease Pathology).
4. Free Energy Head for Out-of-Distribution / Non-Plant Rejection.
"""
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

try:
    from .convnext_v2 import ConvNeXtV2Block, GRN
except ImportError:
    from src.plant_disease.models.convnext_v2 import ConvNeXtV2Block, GRN


class SpatialSaliencyGating(nn.Module):
    """Differentiable Spatial Saliency & Foliage Gating Module.

    Learns to distinguish healthy/diseased leaf foliage from non-foliar
    background (soil, sky, hands, pots) directly in feature space.
    Outputs a continuous attention mask M in [0, 1] and scales feature
    maps via controlled attenuation: F_gated = F * (alpha + (1 - alpha) * M).
    """
    def __init__(self, in_channels: int, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        reduced = max(16, in_channels // 4)
        self.conv1 = nn.Conv2d(in_channels, reduced, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(1, reduced)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(reduced, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute spatial saliency mask M in [0, 1]
        mask = self.sigmoid(self.conv2(self.act(self.norm(self.conv1(x)))))
        # Attenuation gating: background (M->0) suppressed by factor alpha, foliage (M->1) retained at 1.0
        gain = self.alpha + (1.0 - self.alpha) * mask
        gated = x * gain
        return gated, mask


class HierarchicalConditionedHead(nn.Module):
    """Anatomical Hierarchical Classification Head.

    Jointly predicts:
    1. Crop Family (k=14): Apple, Tomato, Corn, etc.
    2. Pathological Condition (k=38): Exact disease.

    Conditioning: Disease logits are conditioned on the soft Crop probabilities,
    mathematically enforcing biological consistency and suppressing cross-species
    hallucinations.
    """
    def __init__(self, in_features: int = 768, num_crops: int = 14, num_diseases: int = 38):
        super().__init__()
        self.norm = nn.LayerNorm(in_features, eps=1e-6)
        self.crop_head = nn.Linear(in_features, num_crops)
        self.crop_proj = nn.Linear(num_crops, in_features)
        self.disease_head = nn.Linear(in_features, num_diseases)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.norm(feat)
        crop_logits = self.crop_head(feat)
        crop_probs = torch.softmax(crop_logits, dim=-1)
        # Condition disease representation on crop taxonomy
        conditioned_feat = feat + self.crop_proj(crop_probs)
        disease_logits = self.disease_head(conditioned_feat)
        return disease_logits, crop_logits


class UnifiedGatedPlantCNN(nn.Module):
    """Unified Gated Plant CNN for In-the-Wild Agricultural Diagnostics.

    A 100% pure 2D CNN combining multi-scale ConvNeXt-V2 feature extraction,
    spatial saliency filtering, and hierarchical multi-task classification.
    """
    def __init__(
        self,
        in_chans: int = 3,
        num_crops: int = 14,
        num_diseases: int = 38,
        depths: Tuple[int, ...] = (3, 3, 9, 3),
        dims: Tuple[int, ...] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        default_single_output: bool = False,
    ):
        super().__init__()
        self.num_crops = num_crops
        self.num_diseases = num_diseases
        self.default_single_output = default_single_output

        # Stem: 4x downsampling
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0]),
        )
        self.downsample_layers.append(stem)

        # Intermediate downsamplers (2x each)
        for i in range(3):
            downsample = nn.Sequential(
                nn.GroupNorm(1, dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample)

        # 4 Stages of ConvNeXt-V2 Blocks with GRN
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    ConvNeXtV2Block(dim=dims[i], drop_path=dp_rates[cur + j])
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Spatial Saliency & Leaf Gating Module (placed after Stage 3, dim=384)
        self.saliency_gating = SpatialSaliencyGating(in_channels=dims[2])

        # Global Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Hierarchical Conditioned Diagnostic Head
        self.head = HierarchicalConditionedHead(
            in_features=dims[3],
            num_crops=num_crops,
            num_diseases=num_diseases,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract multi-scale representations and apply spatial saliency gating."""
        # Stage 0
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)

        # Stage 1
        x = self.downsample_layers[1](x)
        x = self.stages[1](x)

        # Stage 2 (dim=384, spatial resolution H/16 x W/16)
        x = self.downsample_layers[2](x)
        x = self.stages[2](x)

        # Saliency Gating: filter non-foliar background
        x, saliency_mask = self.saliency_gating(x)

        # Stage 3 (dim=768, spatial resolution H/32 x W/32)
        x = self.downsample_layers[3](x)
        x = self.stages[3](x)

        # Global average pool: (N, 768, 1, 1) -> (N, 768)
        pooled = self.avgpool(x).flatten(1)
        return pooled, saliency_mask

    def forward(
        self,
        x: torch.Tensor,
        return_mask: bool = True,
        single_output: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass.

        Args:
            x: Input tensor (N, 3, H, W)
            return_mask: If True (default), returns (disease_logits, crop_logits, saliency_mask).
                         If False, returns (disease_logits, crop_logits).
            single_output: If True (or if default_single_output=True and single_output is None),
                           returns only disease_logits to conform to standard model interfaces.

        Returns:
            Tuple of outputs or single disease_logits tensor.
        """
        use_single = self.default_single_output if single_output is None else single_output
        feat, saliency_mask = self.forward_features(x)
        disease_logits, crop_logits = self.head(feat)

        if use_single:
            return disease_logits
        if return_mask:
            return disease_logits, crop_logits, saliency_mask
        return disease_logits, crop_logits

    def compute_energy(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Compute free Helmholtz energy score for out-of-distribution detection.

        E(x) = -T * logsumexp(logits / T)
        """
        return -temperature * torch.logsumexp(logits / temperature, dim=-1)

    def load_pretrained_backbone(self, ckpt_path: str) -> Dict[str, int]:
        """Load weights from a trained ConvNeXt checkpoint into the backbone."""
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state, dict):
            raise ValueError(f"Invalid checkpoint format from {ckpt_path}")

        # Clean DDP module prefix
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

        model_dict = self.state_dict()
        matched = {}
        for k, v in state.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                matched[k] = v
                continue

            # Map torchvision features to downsample layers and stages
            m_k = k
            if k.startswith("features.0."):
                m_k = k.replace("features.0.", "downsample_layers.0.")
            elif k.startswith("features.1."):
                m_k = k.replace("features.1.", "stages.0.")
            elif k.startswith("features.2."):
                m_k = k.replace("features.2.", "downsample_layers.1.")
            elif k.startswith("features.3."):
                m_k = k.replace("features.3.", "stages.1.")
            elif k.startswith("features.4."):
                m_k = k.replace("features.4.", "downsample_layers.2.")
            elif k.startswith("features.5."):
                m_k = k.replace("features.5.", "stages.2.")
            elif k.startswith("features.6."):
                m_k = k.replace("features.6.", "downsample_layers.3.")
            elif k.startswith("features.7."):
                m_k = k.replace("features.7.", "stages.3.")

            m_k = (
                m_k.replace(".block.0.", ".dwconv.")
                .replace(".block.2.", ".norm.")
                .replace(".block.3.", ".pwconv1.")
                .replace(".block.5.", ".pwconv2.")
            )
            if m_k in model_dict and model_dict[m_k].shape == v.shape:
                matched[m_k] = v

        missing = len(model_dict) - len(matched)
        model_dict.update(matched)
        self.load_state_dict(model_dict)
        return {"matched": len(matched), "missing": missing}


def build_unified_plant_cnn(
    num_crops: int = 14,
    num_diseases: int = 38,
    pretrained_backbone_path: Optional[str] = None,
    single_output: bool = False,
) -> UnifiedGatedPlantCNN:
    """Factory function for UnifiedGatedPlantCNN."""
    model = UnifiedGatedPlantCNN(
        num_crops=num_crops,
        num_diseases=num_diseases,
        default_single_output=single_output,
    )
    if pretrained_backbone_path:
        stats = model.load_pretrained_backbone(pretrained_backbone_path)
        print(f"Loaded pretrained backbone from {pretrained_backbone_path}: matched {stats['matched']} layers.")
    return model
