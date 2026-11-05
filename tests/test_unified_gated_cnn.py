"""Unit tests for the Unified Gated Plant CNN architecture."""
import pytest
import torch

try:
    from plant_disease.models import build_model
    from plant_disease.models.unified_gated_cnn import (
        SpatialSaliencyGating,
        HierarchicalConditionedHead,
        UnifiedGatedPlantCNN,
        build_unified_plant_cnn,
    )
except ImportError:
    from src.plant_disease.models import build_model
    from src.plant_disease.models.unified_gated_cnn import (
        SpatialSaliencyGating,
        HierarchicalConditionedHead,
        UnifiedGatedPlantCNN,
        build_unified_plant_cnn,
    )


def test_spatial_saliency_gating():
    in_channels = 384
    alpha = 0.1
    gating = SpatialSaliencyGating(in_channels=in_channels, alpha=alpha)
    x = torch.randn(2, in_channels, 14, 14)
    gated, mask = gating(x)

    assert gated.shape == x.shape
    assert mask.shape == (2, 1, 14, 14)
    assert (mask >= 0.0).all() and (mask <= 1.0).all()
    # Check attenuation scaling: gain in [alpha, 1.0] for non-negative activations
    relu_x = torch.relu(x)
    gated_relu, _ = gating(relu_x)
    assert (gated_relu <= relu_x + 1e-5).all()
    assert (gated_relu >= alpha * relu_x - 1e-5).all()


def test_hierarchical_conditioned_head():
    in_features = 768
    num_crops = 14
    num_diseases = 38
    head = HierarchicalConditionedHead(
        in_features=in_features, num_crops=num_crops, num_diseases=num_diseases
    )
    feat = torch.randn(4, in_features)
    disease_logits, crop_logits = head(feat)

    assert disease_logits.shape == (4, num_diseases)
    assert crop_logits.shape == (4, num_crops)


def test_unified_gated_plant_cnn_forward():
    model = UnifiedGatedPlantCNN(num_crops=14, num_diseases=38)
    x = torch.randn(2, 3, 224, 224)
    # Default return_mask=True
    disease_logits, crop_logits, mask = model(x)

    assert disease_logits.shape == (2, 38)
    assert crop_logits.shape == (2, 14)
    # Mask spatial dimensions should be H/16, W/16 = 14x14
    assert mask.shape == (2, 1, 14, 14)

    # Test return_mask=False
    out = model(x, return_mask=False)
    assert len(out) == 2
    assert out[0].shape == (2, 38)
    assert out[1].shape == (2, 14)


def test_unified_gated_plant_cnn_backward_gradients():
    model = UnifiedGatedPlantCNN(num_crops=14, num_diseases=38)
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    disease_logits, crop_logits, mask = model(x)

    loss = disease_logits.sum() + 0.3 * crop_logits.sum() + 0.1 * mask.sum()
    loss.backward()

    # Verify gradients flow into the gating module
    assert model.saliency_gating.conv1.weight.grad is not None
    assert model.saliency_gating.conv2.weight.grad is not None
    assert x.grad is not None


def test_energy_computation():
    model = UnifiedGatedPlantCNN(num_crops=14, num_diseases=38)
    logits = torch.randn(3, 38)
    energy = model.compute_energy(logits, temperature=1.0)

    assert energy.shape == (3,)
    # Free energy: -T * logsumexp(logits / T)
    expected = -1.0 * torch.logsumexp(logits, dim=-1)
    assert torch.allclose(energy, expected)


def test_build_model_integration():
    m = build_model("unified_gated_cnn", num_classes=38)
    assert isinstance(m, UnifiedGatedPlantCNN)
    assert m.num_diseases == 38
    assert m.num_crops == 14
