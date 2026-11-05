"""Interactive Diagnostic Engine for Unified Gated Plant CNN.

Provides single-image inference, hierarchical diagnosis, conformal prediction set
computation, and Free Helmholtz Energy out-of-distribution verification.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from src.plant_disease.models.unified_gated_cnn import build_unified_plant_cnn

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "reports" / "unified_extended" / "unified_gated_plant_cnn_extended_best.pt"

CLASSES_38 = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES_38)}
CROPS_14 = sorted(list(set(c.split('___')[0] for c in CLASSES_38)))
CROP_TO_IDX = {cr: i for i, cr in enumerate(CROPS_14)}
CLASS_TO_CROP_IDX = {i: CROP_TO_IDX[c.split('___')[0]] for i, c in enumerate(CLASSES_38)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PlantDiagnosticEngine:
    """Production Diagnostic Engine for Agricultural Field Pathology."""

    def __init__(self, checkpoint_path: Optional[Path] = None, q_hat: float = 0.9822, energy_threshold: float = -3.5):
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        self.q_hat = q_hat
        self.energy_threshold = energy_threshold

        self.model = build_unified_plant_cnn(num_crops=len(CROPS_14), num_diseases=len(CLASSES_38))
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()

    def diagnose_image(self, img_pil: Image.Image) -> Dict:
        """Run single-pass unified inference on an image."""
        img_rgb = img_pil.convert("RGB").resize((224, 224))
        tensor = TF.to_tensor(img_rgb)
        tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0)

        with torch.no_grad():
            d_logits, c_logits, mask = self.model(tensor)
            d_probs = torch.softmax(d_logits, dim=-1)[0].numpy()
            c_probs = torch.softmax(c_logits, dim=-1)[0].numpy()
            energy = self.model.compute_energy(d_logits).item()

        # Upsample mask to image size
        mask_high = F.interpolate(mask, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()

        # Top predictions
        top_d_idx = int(np.argmax(d_probs))
        top_c_idx = int(np.argmax(c_probs))

        # Conformal Prediction Set (APS)
        sorted_d = np.argsort(d_probs)[::-1]
        sorted_probs = d_probs[sorted_d]
        cum_probs = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cum_probs, self.q_hat)
        conformal_indices = sorted_d[:cutoff + 1]

        conformal_set = [
            {
                "disease": CLASSES_38[idx],
                "clean_name": CLASSES_38[idx].split("___")[-1].replace("_", " "),
                "crop": CROPS_14[CLASS_TO_CROP_IDX[idx]].replace("_", " "),
                "confidence": float(d_probs[idx])
            }
            for idx in conformal_indices[:5]  # Top 5 in set
        ]

        # Top 3 Candidates
        top3_indices = sorted_d[:3]
        top3 = [
            {
                "disease": CLASSES_38[idx],
                "clean_name": CLASSES_38[idx].split("___")[-1].replace("_", " "),
                "confidence": float(d_probs[idx])
            }
            for idx in top3_indices
        ]

        # OOD Status via Helmholtz Free Energy
        is_in_distribution = bool(energy < self.energy_threshold)

        return {
            "predicted_disease": CLASSES_38[top_d_idx],
            "disease_clean_name": CLASSES_38[top_d_idx].split("___")[-1].replace("_", " "),
            "disease_confidence": float(d_probs[top_d_idx]),
            "predicted_crop": CROPS_14[top_c_idx].replace("_", " "),
            "crop_confidence": float(c_probs[top_c_idx]),
            "conformal_prediction_set": conformal_set,
            "top3_candidates": top3,
            "free_energy": float(energy),
            "is_foliar_leaf": is_in_distribution,
            "saliency_mean": float(mask_high.mean()),
            "saliency_mask": mask_high
        }
