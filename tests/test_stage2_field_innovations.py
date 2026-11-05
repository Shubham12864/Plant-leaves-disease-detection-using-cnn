"""Unit tests for the Stage 2 Field Accuracy Innovations:
1. Optical augmentations (Color constancy, Solar lighting, Lesion CutMix).
2. Pure CNN ConvNeXt-V2 and Dual-Head architecture forward passes.
3. Asymmetric Loss & Hierarchical Crop-Disease loss.
4. Energy-based OOD scoring and Split Conformal Prediction.
5. Two-stage inference pipeline.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from plant_disease.augmentations import ColorConstancyTransform, SimulatedSolarLighting, LesionCutMix
from plant_disease.losses import AsymmetricLoss, HierarchicalCropDiseaseLoss, EnergyRegularizedLoss
from plant_disease.models.convnext_v2 import ConvNeXtV2, DualHeadCNN, build_convnext_v2_tiny
from plant_disease.calibration import compute_energy_scores, conformal_calibrate, conformal_predict, evaluate_conformal_safety
from plant_disease.pipeline import TwoStagePlantDetector


def test_color_constancy_transform():
    transform = ColorConstancyTransform(p=6)
    # Create image tensor with heavy yellow/red cast (simulating sunset / golden hour)
    tensor = torch.zeros(3, 64, 64)
    tensor[0, :, :] = 0.9  # Red
    tensor[1, :, :] = 0.8  # Green
    tensor[2, :, :] = 0.2  # Blue (low)

    normalized = transform(tensor)
    assert normalized.shape == (3, 64, 64)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    # Channel means should be much closer after Gray-World normalization
    orig_spread = tensor.mean(dim=(1, 2)).max() - tensor.mean(dim=(1, 2)).min()
    norm_spread = normalized.mean(dim=(1, 2)).max() - normalized.mean(dim=(1, 2)).min()
    assert norm_spread < orig_spread


def test_solar_lighting_augmentation():
    sim = SimulatedSolarLighting(shadow_prob=1.0, glare_prob=1.0)
    tensor = torch.ones(3, 64, 64) * 0.5
    augmented = sim(tensor)
    assert augmented.shape == (3, 64, 64)
    assert augmented.min() >= 0.0
    assert augmented.max() <= 1.0


def test_lesion_cutmix():
    cutmix = LesionCutMix(prob=1.0, max_lesion_scale=0.3)
    target = torch.zeros(3, 64, 64)
    donor = torch.ones(3, 64, 64)
    mixed, lam = cutmix(target, donor, donor_is_diseased=True)
    assert mixed.shape == (3, 64, 64)
    assert 0.0 < lam < 1.0
    # Since target was 0 and donor was 1, mixed should have non-zero pixels
    assert mixed.sum() > 0.0


def test_asymmetric_loss():
    asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    loss = asl(logits, targets)
    assert loss.item() < 0.05  # near perfect predictions, very low loss
    assert loss.requires_grad

    # Wrong predictions should produce significant positive loss
    wrong_targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    wrong_loss = asl(logits, wrong_targets)
    assert wrong_loss.item() > loss.item()


def test_hierarchical_loss():
    crit = HierarchicalCropDiseaseLoss()
    crop_logits = torch.randn(4, 5, requires_grad=True)
    disease_logits = torch.randn(4, 10, requires_grad=True)
    crop_targets = torch.tensor([0, 1, 2, 3])
    disease_targets = torch.zeros(4, 10)
    disease_targets[0, 1] = 1.0
    disease_targets[1, 3] = 1.0

    out = crit(crop_logits, disease_logits, crop_targets, disease_targets)
    assert "total_loss" in out
    assert "crop_loss" in out
    assert "disease_loss" in out
    assert out["total_loss"].item() > 0.0


def test_convnext_v2_forward_pass():
    # Use smaller depths/dims for fast testing
    model = ConvNeXtV2(in_chans=3, num_classes=38, depths=[1, 1, 1, 1], dims=[32, 64, 128, 256])
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (2, 38)


def test_dual_head_cnn():
    backbone = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )
    dual_model = DualHeadCNN(backbone=backbone, feature_dim=16, num_crops=14, num_diseases=24)
    x = torch.randn(3, 3, 64, 64)
    crop_logits, disease_logits = dual_model(x)
    assert crop_logits.shape == (3, 14)
    assert disease_logits.shape == (3, 24)


def test_energy_scores_and_conformal():
    logits = np.array([
        [10.0, 0.0, 0.0],   # sharp in-distribution
        [0.1, 0.1, 0.1],    # uniform / out-of-distribution
    ])
    energies = compute_energy_scores(logits, temperature=1.0)
    assert len(energies) == 2
    # Confident in-distribution prediction has much lower energy than uniform OOD
    assert energies[0] < energies[1]

    # Conformal calibration test
    val_probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.95, 0.05],
        [0.85, 0.15],
    ])
    val_y = np.array([0, 0, 0, 0])
    q_hat = conformal_calibrate(val_probs, val_y, alpha=0.1)
    assert 0.0 < q_hat <= 1.0

    test_probs = np.array([[0.95, 0.05], [0.5, 0.5]])
    sets = conformal_predict(test_probs, q_hat)
    assert len(sets) == 2
    assert 0 in sets[0]  # First test sample has true class included


def test_two_stage_pipeline():
    class DummyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3 * 64 * 64, 3)

        def forward(self, x):
            return self.linear(x.view(x.size(0), -1))

    classifier = DummyClassifier()
    pipeline = TwoStagePlantDetector(
        classifier=classifier,
        class_names=["Apple_scab", "Corn_blight", "Tomato_healthy"],
        color_constancy=True,
        image_size=64,
        device="cpu"
    )

    dummy_pil = Image.new("RGB", (128, 128), color=(40, 180, 50))
    report = pipeline.inspect(dummy_pil)
    assert report.leaves_detected > 0
    assert len(report.diagnoses) > 0
    assert report.primary_diagnosis is not None
