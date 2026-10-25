"""Grad-CAM saliency maps, dependency-free.

``grad-cam`` and ``torchcam`` are small packages, but adding a dependency for
about sixty lines of hooks is not worth the install risk on Kaggle. This
implementation follows the original formulation: weight each activation channel
by the mean gradient flowing through it, take the ReLU of the weighted sum, and
normalise to 0-1.

Scope warning, repeated in the report: a Grad-CAM map is an explanation aid, not
proof that the model reasons the way a plant pathologist would. It shows where
gradient sensitivity is concentrated. High activation on a lesion is consistent
with lesion-based reasoning; high activation on a pot rim or a table edge is
evidence of a shortcut.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Compute Grad-CAM for one target layer of a classification model."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __call__(self, batch: torch.Tensor, class_index: torch.Tensor | None = None) -> np.ndarray:
        """Return a (batch, H, W) float32 array of normalised saliency maps."""
        self.model.eval()
        batch = batch.clone().requires_grad_(True)
        logits = self.model(batch)
        if class_index is None:
            class_index = logits.argmax(dim=1)
        chosen = logits.gather(1, class_index.view(-1, 1)).sum()
        self.model.zero_grad(set_to_none=True)
        chosen.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks never fired. Check that target_layer is inside the model.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=batch.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        flat = cam.flatten(1)
        peak = flat.max(dim=1).values.clamp_min(1e-8).view(-1, 1, 1)
        return (cam / peak).detach().cpu().numpy()


def target_layer_for(model: torch.nn.Module, architecture: str) -> torch.nn.Module:
    """Pick the last convolutional block for a supported architecture."""
    if architecture == "resnet18":
        return model.layer4[-1]
    if architecture == "efficientnet_b0":
        return model.features[-1]
    # SmallCNN: last conv block before pooling.
    if hasattr(model, "features"):
        blocks = [module for module in model.features if isinstance(module, torch.nn.Conv2d)]
        if blocks:
            return blocks[-1]
    raise ValueError(
        f"No known Grad-CAM target layer for architecture {architecture!r}. "
        "Pass the layer explicitly."
    )


def overlay(cam: np.ndarray, image_rgb: np.ndarray, alpha: float = 0.45, colormap: str = "jet") -> np.ndarray:
    """Blend a saliency map over an RGB uint8 image, return uint8 RGB."""
    import matplotlib

    if image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be uint8 in 0-255")
    # matplotlib.colormaps exists from 3.5; cm.get_cmap was removed in 3.9.
    lookup = getattr(matplotlib, "colormaps", None)
    coloured = (lookup[colormap] if lookup is not None else matplotlib.cm.get_cmap(colormap))(cam)[..., :3]
    base = image_rgb.astype(np.float32) / 255.0
    blended = (1 - alpha) * base + alpha * coloured
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)
