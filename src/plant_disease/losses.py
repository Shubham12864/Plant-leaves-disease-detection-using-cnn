"""Loss functions designed for real-world field plant pathology.

Features:
1. AsymmetricLoss (ASL): Multi-label loss that prevents easy negative labels
   from overwhelming rare disease signals, supporting co-occurring infections.
2. HierarchicalCropDiseaseLoss: Multi-task loss decoupling species taxonomy
   (e.g., Tomato vs Apple) from pathological state (e.g., Blight vs Rust vs Healthy).
3. EnergyRegularizedLoss: Penalizes high energy on in-distribution leaf images
   to enable calibrated out-of-distribution (OOD) detection.
"""
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for Multi-Label Classification.

    Reference:
        Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021.

    In agricultural fields, a single leaf can exhibit multiple conditions
    (e.g., Early Blight + Septoria, or nutritional chlorosis + fungal lesions).
    ASL provides asymmetric focusing (gamma_neg > gamma_pos) and margin clipping
    to suppress easy negative gradient mass while maintaining sharp gradients
    for genuine co-occurring disease symptoms.
    """
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C) unnormalized class logits.
            targets: (N, C) binary ground truth labels (0 or 1).
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # Asymmetric Clipping for negative classes
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Basic Cross Entropy
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1.0 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        loss = -loss.sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class HierarchicalCropDiseaseLoss(nn.Module):
    """Hierarchical Crop + Disease Multi-Task Loss.

    Separates diagnosis into:
    - Head 1: Crop Species identification (Categorical Cross Entropy).
    - Head 2: Disease / Pathological condition (Asymmetric Multi-label Loss).
    """
    def __init__(self, crop_weight: float = 0.4, disease_weight: float = 1.0, clip: float = 0.05):
        super().__init__()
        self.crop_weight = crop_weight
        self.disease_weight = disease_weight
        self.crop_criterion = nn.CrossEntropyLoss()
        self.disease_criterion = AsymmetricLoss(clip=clip)

    def forward(
        self,
        crop_logits: torch.Tensor,
        disease_logits: torch.Tensor,
        crop_targets: torch.Tensor,
        disease_targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        loss_crop = self.crop_criterion(crop_logits, crop_targets)
        loss_disease = self.disease_criterion(disease_logits, disease_targets)
        total = self.crop_weight * loss_crop + self.disease_weight * loss_disease
        return {
            "total_loss": total,
            "crop_loss": loss_crop,
            "disease_loss": loss_disease,
        }


class EnergyRegularizedLoss(nn.Module):
    """Energy-Regularized Loss to enhance Out-Of-Distribution (OOD) separability.

    Pushes in-distribution samples to lower Helmholtz free energy while
    fitting cross-entropy classification labels.
    """
    def __init__(self, temperature: float = 1.0, energy_weight: float = 0.1, m_in: float = -12.0):
        super().__init__()
        self.T = temperature
        self.energy_weight = energy_weight
        self.m_in = m_in
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        # Energy = -T * log(sum(exp(logits / T)))
        energy = -self.T * torch.logsumexp(logits / self.T, dim=-1)
        # Penalize energy above target margin m_in
        loss_energy = torch.mean(F.relu(energy - self.m_in) ** 2)
        return loss_ce + self.energy_weight * loss_energy
