import numpy as np
import pytest
import torch
from torch import nn

from plant_disease.gradcam import GradCAM, overlay, target_layer_for


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1))
        self.head = nn.Linear(4, 3)

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))


def test_gradcam_output_matches_input_size_and_is_normalised():
    torch.manual_seed(0)
    model = _Tiny()
    cam = GradCAM(model, target_layer_for(model, "smallcnn"))
    try:
        maps = cam(torch.randn(2, 3, 12, 12))
    finally:
        cam.remove()
    assert maps.shape == (2, 12, 12)
    assert maps.dtype == np.float32
    assert maps.min() >= 0.0
    assert maps.max() <= 1.0 + 1e-6


def test_gradcam_remove_detaches_hooks():
    model = _Tiny()
    layer = target_layer_for(model, "smallcnn")
    cam = GradCAM(model, layer)
    cam.remove()
    assert not layer._forward_hooks
    assert not layer._backward_hooks


def test_target_layer_for_smallcnn_is_last_conv():
    model = _Tiny()
    assert target_layer_for(model, "smallcnn") is model.features[2]


def test_target_layer_for_unknown_architecture_without_features_raises():
    with pytest.raises(ValueError):
        target_layer_for(nn.Linear(2, 2), "mystery")


def test_overlay_returns_uint8_rgb_and_rejects_float_images():
    cam = np.random.default_rng(0).random((8, 8)).astype(np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    out = overlay(cam, image)
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8
    with pytest.raises(ValueError):
        overlay(cam, image.astype(np.float32))
