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


def test_load_pretrained_folds_layer_scale(tmp_path):
    """Verify torchvision ConvNeXt-Tiny transfer folds layer_scale into pwconv2 and transfers heads."""
    import torchvision.models as tv_models

    tv_model = tv_models.convnext_tiny(num_classes=38)
    with torch.no_grad():
        # Set distinct non-zero layer_scale and head weights to verify folding
        tv_model.features[1][0].layer_scale.fill_(0.25)
        tv_model.features[1][0].block[5].weight.fill_(2.0)
        tv_model.features[1][0].block[5].bias.fill_(4.0)
        tv_model.classifier[2].weight.fill_(0.5)

    ckpt_path = tmp_path / "convnext_tiny_mock.pt"
    torch.save({"model_state_dict": tv_model.state_dict()}, ckpt_path)

    gated = UnifiedGatedPlantCNN(num_crops=14, num_diseases=38)
    stats = gated.load_pretrained_backbone(str(ckpt_path))

    assert stats["folded_layer_scales"] == 18
    assert torch.allclose(gated.stages[0][0].pwconv2.weight, torch.full_like(gated.stages[0][0].pwconv2.weight, 0.5))
    assert torch.allclose(gated.stages[0][0].pwconv2.bias, torch.full_like(gated.stages[0][0].pwconv2.bias, 1.0))
    assert torch.allclose(gated.head.disease_head.weight, torch.full_like(gated.head.disease_head.weight, 0.5))


def test_taxonomy_and_pathogen_advisories():
    try:
        from plant_disease.taxonomy import (
            CLASSES_38,
            CROPS_14,
            CLASS_TO_CROP_IDX,
            get_treatment_for_disease,
            preprocess_pil_image,
        )
    except ImportError:
        from src.plant_disease.taxonomy import (
            CLASSES_38,
            CROPS_14,
            CLASS_TO_CROP_IDX,
            get_treatment_for_disease,
            preprocess_pil_image,
        )
    from PIL import Image

    assert len(CLASSES_38) == 38
    assert len(CROPS_14) == 14
    assert len(CLASS_TO_CROP_IDX) == 38

    virus_adv = get_treatment_for_disease("Tomato___Tomato_Yellow_Leaf_Curl_Virus")
    assert "fungicides are ineffective" in virus_adv["chemical"].lower()

    mite_adv = get_treatment_for_disease("Tomato___Spider_mites Two-spotted_spider_mite")
    assert "miticide" in mite_adv["chemical"].lower()

    # Non-square photo preprocessing produces exact 224x224 center crop
    wide_img = Image.new("RGB", (640, 360), color=(20, 120, 40))
    cropped = preprocess_pil_image(wide_img, target_size=224, center_crop=True)
    assert cropped.size == (224, 224)


def test_diagnostic_engine_outputs(tmp_path):
    try:
        from plant_disease.diagnostic_engine import PlantDiagnosticEngine
    except ImportError:
        from src.plant_disease.diagnostic_engine import PlantDiagnosticEngine
    from PIL import Image

    model = UnifiedGatedPlantCNN(num_crops=14, num_diseases=38)
    ckpt_path = tmp_path / "dummy_unified.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    engine = PlantDiagnosticEngine(checkpoint_path=ckpt_path, q_hat=0.9822)
    res = engine.diagnose_image(Image.new("RGB", (400, 300), color=(40, 100, 30)))

    assert res["saliency_mask"].shape == (224, 224)
    assert 1 <= res["conformal_set_size"] <= 38
    assert len(res["top3_candidates"]) == 3
    assert res["inference_ms"] >= 0.0
    assert "organic" in res["treatment_advisory"]
