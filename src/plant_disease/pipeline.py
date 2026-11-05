"""Two-Stage Inference Pipeline for Field Plant Pathology.

Decouples detection into:
- Stage 1: Spatial Leaf Localization (isolates individual leaves from complex backgrounds).
- Stage 2: Optical Preprocessing & ConvNeXt-V2 Diagnosis with Energy OOD & Conformal Calibration.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF

from .augmentations import ColorConstancyTransform
from .calibration import compute_energy_scores, conformal_predict


@dataclass
class LeafDiagnosis:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    predicted_class: str
    class_id: int
    confidence: float
    energy_score: float
    is_ood: bool
    conformal_set: List[str] = field(default_factory=list)
    multi_label_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class FieldInspectionReport:
    image_path: Optional[str]
    image_size: Tuple[int, int]  # (W, H)
    leaves_detected: int
    diagnoses: List[LeafDiagnosis]
    primary_diagnosis: Optional[str] = None
    actionable_summary: str = ""


class TwoStagePlantDetector:
    """End-to-End Two-Stage Field Diagnosis System."""
    def __init__(
        self,
        classifier: nn.Module,
        class_names: List[str],
        detector: Optional[Any] = None,
        color_constancy: bool = True,
        energy_threshold: float = -10.0,
        conformal_q_hat: Optional[float] = None,
        device: str = "cpu",
        image_size: int = 224,
    ):
        self.classifier = classifier.to(device).eval()
        self.class_names = class_names
        self.detector = detector
        self.color_constancy = ColorConstancyTransform(p=6) if color_constancy else None
        self.energy_threshold = energy_threshold
        self.conformal_q_hat = conformal_q_hat
        self.device = torch.device(device)
        self.image_size = image_size

    def extract_leaf_crops(
        self,
        pil_img: Image.Image,
        padding_ratio: float = 0.10,
        min_box_size: int = 32
    ) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """Detect or segment individual leaves in the raw field photograph."""
        w, h = pil_img.size

        # If a detector (e.g. YOLO) is provided, use it
        if self.detector is not None:
            results = self.detector(pil_img)
            boxes = []
            for r in results:
                for box in r.boxes.xyxy.cpu().numpy():
                    boxes.append(tuple(map(int, box)))
        else:
            # Fallback when detector model not initialized:
            # Multi-scale focal grid crops (center crop + 4 quadrant crops)
            boxes = [
                (0, 0, w, h),  # full frame
                (int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85)),  # center focus
            ]

        crops = []
        for (x1, y1, x2, y2) in boxes:
            bw, bh = x2 - x1, y2 - y1
            if bw < min_box_size or bh < min_box_size:
                continue
            # Context padding to preserve lesion borders
            pad_x = int(bw * padding_ratio)
            pad_y = int(bh * padding_ratio)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)

            crop = pil_img.crop((cx1, cy1, cx2, cy2))
            crops.append((crop, (x1, y1, x2, y2)))

        return crops

    def preprocess_crop(self, crop_img: Image.Image) -> torch.Tensor:
        """Resize and apply optical illuminant normalization."""
        resized = crop_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        tensor = TF.to_tensor(resized)  # (3, H, W) in [0, 1]

        if self.color_constancy is not None:
            tensor = self.color_constancy(tensor)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor - mean) / std

    @torch.no_grad()
    def inspect(self, image_input: Any) -> FieldInspectionReport:
        """Run two-stage inspection on a field photograph."""
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
            path_str = str(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            path_str = None
        else:
            raise TypeError("Expected image path or PIL.Image")

        crops_and_boxes = self.extract_leaf_crops(pil_img)
        if not crops_and_boxes:
            return FieldInspectionReport(
                image_path=path_str,
                image_size=pil_img.size,
                leaves_detected=0,
                diagnoses=[],
                actionable_summary="No leaf candidate detected in the frame. Please take a closer photo."
            )

        tensors = torch.stack([self.preprocess_crop(c) for c, _ in crops_and_boxes]).to(self.device)

        # Forward pass through CNN
        logits = self.classifier(tensors)
        if isinstance(logits, tuple):  # Dual-Head output: (crop_logits, disease_logits)
            _, disease_logits = logits
            logits = disease_logits

        logits_np = logits.cpu().numpy()
        probs_np = torch.softmax(logits, dim=-1).cpu().numpy()
        energies = compute_energy_scores(logits_np)

        diagnoses: List[LeafDiagnosis] = []
        for i, (_, bbox) in enumerate(crops_and_boxes):
            pred_id = int(np.argmax(probs_np[i]))
            pred_class = self.class_names[pred_id] if pred_id < len(self.class_names) else f"class_{pred_id}"
            conf = float(probs_np[i, pred_id])
            energy = float(energies[i])
            is_ood = energy > self.energy_threshold

            conformal_set = []
            if self.conformal_q_hat is not None:
                inc_ids = conformal_predict(probs_np[i:i + 1], self.conformal_q_hat)[0]
                conformal_set = [self.class_names[j] for j in inc_ids if j < len(self.class_names)]

            diagnoses.append(LeafDiagnosis(
                bbox=bbox,
                predicted_class=pred_class if not is_ood else "Unrecognized / Out-Of-Distribution",
                class_id=pred_id,
                confidence=conf,
                energy_score=energy,
                is_ood=is_ood,
                conformal_set=conformal_set,
            ))

        valid = [d for d in diagnoses if not d.is_ood]
        primary = valid[0].predicted_class if valid else "Abstained (Uncertain / High Energy)"

        return FieldInspectionReport(
            image_path=path_str,
            image_size=pil_img.size,
            leaves_detected=len(diagnoses),
            diagnoses=diagnoses,
            primary_diagnosis=primary,
            actionable_summary=f"Analyzed {len(diagnoses)} leaf regions. Primary condition: {primary}."
        )
