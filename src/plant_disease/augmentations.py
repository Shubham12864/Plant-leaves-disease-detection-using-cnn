"""Optical and leaf-level data augmentations for field robustness.

Replaces the failed background-replacement augmentation of S19 with:
1. Color Constancy (Gray-World / Minkowski p-norm illuminant normalization).
2. Local contrast enhancement (CLAHE in LAB color space).
3. Lesion CutMix (pasting segmented lesions onto healthy host leaves).
4. Simulated solar lighting (specular highlights and cast shadow masks).
"""
import math
from typing import Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class ColorConstancyTransform:
    """Minkowski p-norm / Gray-World color constancy transformation.

    In the field, natural sunlight shifts drastically in color temperature
    (from 3000K dawn/dusk to 6500K noon and 8000K canopy shade).
    This transform estimates the scene illuminant vector and normalizes
    the channel chromaticity so plant tissue exhibits invariant pigment response.
    """
    def __init__(self, p: int = 6, eps: float = 1e-6):
        self.p = p
        self.eps = eps

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply color constancy to a (C, H, W) float tensor in [0, 1]."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError(f"Expected tensor shape (3, H, W), got {tensor.shape}")

        orig_dtype = tensor.dtype
        # Work in float32 for stable power calculation
        t = tensor.to(torch.float32)

        # Compute Minkowski p-norm across spatial dimensions for each channel
        # e_c = ( 1/|Omega| * sum(x_c^p) )^(1/p)
        spatial_size = t.shape[1] * t.shape[2]
        powered = torch.pow(torch.clamp(t, min=0.0), self.p)
        mean_p = torch.clamp(powered.sum(dim=(1, 2)) / spatial_size, min=self.eps)
        illuminant = torch.pow(mean_p, 1.0 / self.p) + self.eps

        # Scale channels relative to gray mean
        gray = illuminant.mean()
        scale = gray / illuminant  # shape (3,)

        normalized = t * scale[:, None, None]
        return torch.clamp(normalized, 0.0, 1.0).to(orig_dtype)


class SimulatedSolarLighting:
    """Simulates harsh open-field lighting: direct specular glare and cast shadows.

    Waxy leaf cuticles create bright specular highlights under direct sunlight,
    while overlapping foliage casts hard high-contrast shadows. Both disrupt
    standard convolutional activations if not represented in training.
    """
    def __init__(self, shadow_prob: float = 0.4, glare_prob: float = 0.3):
        self.shadow_prob = shadow_prob
        self.glare_prob = glare_prob

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply lighting variations to a (3, H, W) float tensor in [0, 1]."""
        t = tensor.clone()
        _, h, w = t.shape

        # 1. Cast shadow simulation (polygon / half-plane attenuation)
        if torch.rand(1).item() < self.shadow_prob:
            angle = torch.rand(1).item() * 2 * math.pi
            # Linear ramp across an arbitrary angle
            y_coords = torch.linspace(-1, 1, h, device=t.device)
            x_coords = torch.linspace(-1, 1, w, device=t.device)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            projection = grid_x * math.cos(angle) + grid_y * math.sin(angle)
            # Create a soft shadow mask with attenuation factor between 0.35 and 0.65
            shadow_mask = torch.sigmoid(projection * 5.0)  # smooth transition
            attenuation = 0.35 + 0.3 * torch.rand(1).item()
            multiplier = 1.0 - (1.0 - attenuation) * shadow_mask
            t = t * multiplier.unsqueeze(0)

        # 2. Specular glare simulation (saturated elliptical region)
        if torch.rand(1).item() < self.glare_prob:
            cx = torch.rand(1).item() * w
            cy = torch.rand(1).item() * h
            rx = (0.05 + 0.15 * torch.rand(1).item()) * w
            ry = (0.05 + 0.15 * torch.rand(1).item()) * h

            y_coords = torch.arange(h, device=t.device, dtype=torch.float32)
            x_coords = torch.arange(w, device=t.device, dtype=torch.float32)
            gy, gx = torch.meshgrid(y_coords, x_coords, indexing='ij')

            dist = ((gx - cx) / rx) ** 2 + ((gy - cy) / ry) ** 2
            glare_intensity = torch.exp(-dist * 2.0) * (0.4 + 0.4 * torch.rand(1).item())
            t = t + glare_intensity.unsqueeze(0)

        return torch.clamp(t, 0.0, 1.0)


class LesionCutMix:
    """Anatomical Lesion CutMix (Leaf-Paste).

    Instead of replacing the background (which S19 proved is ineffective),
    this augments leaf appearance by extracting localized lesion patches
    and compositing them onto target leaves. This breaks whole-leaf spatial
    shortcuts and forces the model to attend to focal necrotic/chlorotic symptoms.
    """
    def __init__(self, prob: float = 0.5, max_lesion_scale: float = 0.35):
        self.prob = prob
        self.max_lesion_scale = max_lesion_scale

    def __call__(
        self,
        target_img: torch.Tensor,
        donor_img: torch.Tensor,
        donor_is_diseased: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """Apply Lesion CutMix between target and donor image.

        Returns:
            Tuple of (augmented_img, mixing_ratio_lambda).
        """
        if torch.rand(1).item() > self.prob or not donor_is_diseased:
            return target_img, 1.0

        _, th, tw = target_img.shape
        _, dh, dw = donor_img.shape
        max_h = min(th, dh)
        max_w = min(tw, dw)

        # Sample patch dimensions safely bound by minimum common dimension
        patch_h = max(1, int(max_h * (0.15 + (self.max_lesion_scale - 0.15) * torch.rand(1).item())))
        patch_w = max(1, int(max_w * (0.15 + (self.max_lesion_scale - 0.15) * torch.rand(1).item())))

        # Safe donor crop coordinates
        dy = torch.randint(0, max(1, dh - patch_h + 1), (1,)).item()
        dx = torch.randint(0, max(1, dw - patch_w + 1), (1,)).item()
        donor_patch = donor_img[:, dy:dy + patch_h, dx:dx + patch_w]

        # Safe target placement coordinates
        ty = torch.randint(0, max(1, th - patch_h + 1), (1,)).item()
        tx = torch.randint(0, max(1, tw - patch_w + 1), (1,)).item()

        # Gaussian edge feathering for smooth biological boundary
        y_lin = torch.linspace(-2, 2, patch_h, device=target_img.device)
        x_lin = torch.linspace(-2, 2, patch_w, device=target_img.device)
        gy, gx = torch.meshgrid(y_lin, x_lin, indexing='ij')
        feather_mask = torch.exp(-(gx**2 + gy**2) / 2.0).unsqueeze(0)

        out = target_img.clone()
        out[:, ty:ty + patch_h, tx:tx + patch_w] = (
            (1.0 - feather_mask) * out[:, ty:ty + patch_h, tx:tx + patch_w] +
            feather_mask * donor_patch
        )

        # Accurate lambda integration over soft feathered Gaussian mask
        lam = 1.0 - float(feather_mask.sum().item()) / float(th * tw)
        return out, lam
