"""Interactive Diagnostic Engine for Unified Gated Plant CNN.

Provides single-image inference, hierarchical diagnosis, calibrated conformal
prediction set computation, Helmholtz Free Energy OOD checks, and pathogen-accurate
agronomic advisories.
"""
from pathlib import Path
import time
from typing import Dict, Optional
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

try:
    from .models.unified_gated_cnn import build_unified_plant_cnn
    from .taxonomy import (
        CLASSES_38,
        CLASS_TO_CROP_IDX,
        CROPS_14,
        IMAGENET_MEAN,
        IMAGENET_STD,
        get_treatment_for_disease,
        preprocess_pil_image,
    )
except ImportError:
    from src.plant_disease.models.unified_gated_cnn import build_unified_plant_cnn
    from src.plant_disease.taxonomy import (
        CLASSES_38,
        CLASS_TO_CROP_IDX,
        CROPS_14,
        IMAGENET_MEAN,
        IMAGENET_STD,
        get_treatment_for_disease,
        preprocess_pil_image,
    )

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "reports" / "unified_extended" / "unified_gated_plant_cnn_extended_best.pt"
CROP_IDX_LIST = [CLASS_TO_CROP_IDX[i] for i in range(len(CLASSES_38))]


def compute_joint_probabilities(disease_probs: np.ndarray, crop_probs: np.ndarray) -> np.ndarray:
    """Apply hierarchical botanical conditioning: P_joint(d) proportional to P(d|x) * P(crop(d)|x)."""
    joint = disease_probs * crop_probs[CROP_IDX_LIST]
    total = float(joint.sum())
    if total > 0.0:
        return joint / total
    return disease_probs


class PlantDiagnosticEngine:
    """Production Diagnostic Engine for Agricultural Field Pathology."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        q_hat: float = 0.9822,
        energy_threshold: float = -3.5,
        center_crop: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        self.q_hat = float(q_hat)
        self.energy_threshold = float(energy_threshold)
        self.center_crop = center_crop

        self.model = build_unified_plant_cnn(num_crops=len(CROPS_14), num_diseases=len(CLASSES_38))
        try:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()

    def diagnose_image(self, img_pil: Image.Image) -> Dict:
        """Run single-pass unified inference on an image."""
        img_rgb = preprocess_pil_image(img_pil, target_size=224, center_crop=self.center_crop)
        tensor = TF.to_tensor(img_rgb)
        tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0)

        t0 = time.perf_counter()
        with torch.no_grad():
            d_logits, c_logits, mask = self.model(tensor)
            d_probs = torch.softmax(d_logits, dim=-1)[0].numpy()
            c_probs = torch.softmax(c_logits, dim=-1)[0].numpy()
            energy = self.model.compute_energy(d_logits).item()
        inference_ms = (time.perf_counter() - t0) * 1000.0

        # Upsample 14x14 mask to image size
        mask_high = F.interpolate(mask, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()

        # Hierarchical joint conditioning
        joint_probs = compute_joint_probabilities(d_probs, c_probs)
        top_d_idx = int(np.argmax(joint_probs))
        top_c_idx = int(np.argmax(c_probs))

        # Calibrated Adaptive Prediction Set (APS) over conditioned joint probabilities
        sorted_d = np.argsort(joint_probs)[::-1]
        cum_probs = np.cumsum(joint_probs[sorted_d])
        cutoff = min(int(np.searchsorted(cum_probs, self.q_hat)), len(sorted_d) - 1)
        conformal_indices = sorted_d[: cutoff + 1]

        conformal_set = [
            {
                "disease": CLASSES_38[idx],
                "clean_name": CLASSES_38[idx].split("___")[-1].replace("_", " "),
                "crop": CROPS_14[CLASS_TO_CROP_IDX[idx]].replace("_", " "),
                "confidence": float(joint_probs[idx]),
            }
            for idx in conformal_indices[:5]
        ]

        top3 = [
            {
                "disease": CLASSES_38[idx],
                "clean_name": CLASSES_38[idx].split("___")[-1].replace("_", " "),
                "crop": CROPS_14[CLASS_TO_CROP_IDX[idx]].replace("_", " "),
                "confidence": float(joint_probs[idx]),
            }
            for idx in sorted_d[:3]
        ]

        top5 = [
            {
                "disease": CLASSES_38[idx],
                "clean_name": CLASSES_38[idx].split("___")[-1].replace("_", " "),
                "crop": CROPS_14[CLASS_TO_CROP_IDX[idx]].replace("_", " "),
                "confidence": float(joint_probs[idx]),
            }
            for idx in sorted_d[:5]
        ]

        pred_disease = CLASSES_38[top_d_idx]
        is_in_distribution = bool(energy < self.energy_threshold)

        return {
            "preprocessed_image": img_rgb,
            "predicted_disease": pred_disease,
            "disease_clean_name": pred_disease.split("___")[-1].replace("_", " "),
            "disease_confidence": float(joint_probs[top_d_idx]),
            "marginal_disease_confidence": float(d_probs[top_d_idx]),
            "predicted_crop": CROPS_14[top_c_idx].replace("_", " "),
            "crop_confidence": float(c_probs[top_c_idx]),
            "conformal_prediction_set": conformal_set,
            "conformal_set_size": int(len(conformal_indices)),
            "top3_candidates": top3,
            "top5_candidates": top5,
            "free_energy": float(energy),
            "is_foliar_leaf": is_in_distribution,
            "saliency_mean": float(mask_high.mean()),
            "saliency_mask": mask_high,
            "inference_ms": float(inference_ms),
            "treatment_advisory": get_treatment_for_disease(pred_disease),
        }
