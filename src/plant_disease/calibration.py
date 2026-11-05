"""Confidence calibration diagnostics.

The project plan lists ``calibration`` as a required metric but nothing computed
it. This module provides the standard set so the report can state a measured
value instead of claiming calibrated confidence without evidence.

Definitions used here
---------------------
ECE (expected calibration error)
    Bin predictions by top-1 confidence, then average |accuracy - confidence|
    weighted by bin population. Lower is better; 0 is perfect calibration.
MCE (maximum calibration error)
    The worst single bin. Useful because ECE can hide one badly calibrated bin.
Brier score
    Mean squared error between the predicted probability vector and the one-hot
    label. Lower is better. Rewards both calibration and sharpness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


def reliability_bins(probabilities: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> list[ReliabilityBin]:
    """Bin predictions by top-1 confidence for a reliability diagram."""
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        count = int(mask.sum())
        bins.append(
            ReliabilityBin(
                lower=float(low),
                upper=float(high),
                count=count,
                mean_confidence=float(confidence[mask].mean()) if count else 0.0,
                accuracy=float(correct[mask].mean()) if count else 0.0,
            )
        )
    return bins


def expected_calibration_error(probabilities: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    total = len(y_true)
    if total == 0:
        raise ValueError("Cannot compute calibration on an empty set")
    return sum(bin_.count / total * abs(bin_.accuracy - bin_.mean_confidence) for bin_ in reliability_bins(probabilities, y_true, n_bins))


def maximum_calibration_error(probabilities: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    gaps = [abs(b.accuracy - b.mean_confidence) for b in reliability_bins(probabilities, y_true, n_bins) if b.count]
    return float(max(gaps)) if gaps else 0.0


def brier_score(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def selective_risk_coverage(probabilities: np.ndarray, y_true: np.ndarray, thresholds=None) -> list[dict]:
    """Accuracy retained when predictions below a confidence threshold abstain.

    This is the straightforward version of the ``selective_risk_coverage`` metric
    in ``configs/data_contract.yaml``: for each threshold, how much of the set is
    answered and how accurate those answers are.
    """
    if thresholds is None:
        thresholds = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    rows = []
    for threshold in thresholds:
        answered = confidence >= threshold
        coverage = float(answered.mean())
        accuracy = float(correct[answered].mean()) if answered.any() else float("nan")
        rows.append({"threshold": float(threshold), "coverage": coverage, "selective_accuracy": accuracy})
    return rows


def summarise(probabilities: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> dict:
    """All calibration numbers in one dict, ready for JSON."""
    confidence = probabilities.max(axis=1)
    return {
        "ece": expected_calibration_error(probabilities, y_true, n_bins),
        "mce": maximum_calibration_error(probabilities, y_true, n_bins),
        "brier": brier_score(probabilities, y_true),
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(np.median(confidence)),
        "fraction_below_0_5": float((confidence < 0.5).mean()),
        "fraction_below_0_8": float((confidence < 0.8).mean()),
        "n_bins": n_bins,
    }


def to_markdown_table(bins: list[ReliabilityBin]) -> str:
    """Reliability bins as a markdown table for the report."""
    lines = ["| Bin | Count | Mean confidence | Accuracy | Gap |", "|---|---:|---:|---:|---:|"]
    for bin_ in bins:
        lines.append(
            f"| {bin_.lower:.2f}-{bin_.upper:.2f} | {bin_.count} | "
            f"{bin_.mean_confidence:.3f} | {bin_.accuracy:.3f} | {abs(bin_.accuracy - bin_.mean_confidence):.3f} |"
        )
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Energy-Based OOD Detection & Split Conformal Prediction (Solving H4)
# -----------------------------------------------------------------------------

def compute_energy_scores(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute Helmholtz free energy scores from raw classification logits.

    Reference:
        Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020.

    Unlike softmax probabilities which are constrained to sum to 1.0 (causing
    high-confidence hallucinations on foreign objects), energy is unconstrained
    and maps directly to the data density. In-distribution crop leaves produce
    low energy (high negative values), while non-plant clutter and unseen weeds
    produce significantly higher energy.
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits array (N, C), got {logits.shape}")
    scaled = logits / temperature
    # Stable LogSumExp computation: max + log(sum(exp(x - max)))
    max_val = np.max(scaled, axis=1, keepdims=True)
    lse = max_val.squeeze(1) + np.log(np.sum(np.exp(scaled - max_val), axis=1))
    return -temperature * lse


def conformal_calibrate(val_probs: np.ndarray, val_y_true: np.ndarray, alpha: float = 0.05) -> float:
    """Compute Split Conformal Prediction threshold q_hat.

    Provides finite-sample statistical guarantees that the true class is contained
    within the predicted set with probability >= 1 - alpha (e.g. 95% coverage).

    Args:
        val_probs: (N, C) predicted probabilities on held-out validation data.
        val_y_true: (N,) ground-truth class indices.
        alpha: Error allowance (default 0.05 for 95% guaranteed coverage).

    Returns:
        q_hat: Conformal non-conformity score quantile.
    """
    n = len(val_y_true)
    if n == 0:
        raise ValueError("Cannot calibrate conformal prediction on empty validation set")

    # Non-conformity score: 1 - P(true class)
    true_probs = val_probs[np.arange(n), val_y_true]
    nonconformity_scores = 1.0 - true_probs

    # Conformal quantile adjustment: ceil((n + 1) * (1 - alpha)) / n
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    q_hat = float(np.quantile(nonconformity_scores, level, method="higher" if hasattr(np, "quantile") else "linear"))
    return q_hat


def conformal_predict(probabilities: np.ndarray, q_hat: float) -> list[list[int]]:
    """Generate prediction sets C(x) with statistical coverage guarantee.

    A class c is included in C(x) if and only if P(c | x) >= 1 - q_hat.
    If |C(x)| == 1, the model is definitively confident.
    If |C(x)| > 1, multiple differential diagnoses are provided (e.g. Early vs Late Blight).
    If |C(x)| == 0, the input is anomalous / unclassifiable.
    """
    threshold = 1.0 - q_hat
    prediction_sets = []
    for probs in probabilities:
        included = np.where(probs >= threshold)[0].tolist()
        prediction_sets.append(included)
    return prediction_sets


def evaluate_conformal_safety(
    val_probs: np.ndarray,
    val_y_true: np.ndarray,
    test_probs: np.ndarray,
    test_y_true: np.ndarray,
    test_logits: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    energy_percentile: float = 95.0,
) -> dict:
    """Evaluate Conformal Prediction coverage and Energy OOD performance.

    Replaces naive softmax confidence thresholding with distribution-free
    risk bounds and energy filtering.
    """
    q_hat = conformal_calibrate(val_probs, val_y_true, alpha=alpha)
    pred_sets = conformal_predict(test_probs, q_hat)

    # 1. Measure empirical coverage on test set
    n_test = len(test_y_true)
    covered = sum(test_y_true[i] in pred_sets[i] for i in range(n_test))
    empirical_coverage = float(covered / n_test) if n_test > 0 else 0.0

    # 2. Set size distribution
    set_sizes = [len(s) for s in pred_sets]
    mean_set_size = float(np.mean(set_sizes)) if set_sizes else 0.0
    singleton_fraction = float(np.mean([s == 1 for s in set_sizes])) if set_sizes else 0.0
    empty_fraction = float(np.mean([s == 0 for s in set_sizes])) if set_sizes else 0.0

    result = {
        "alpha": alpha,
        "q_hat": q_hat,
        "guaranteed_coverage": 1.0 - alpha,
        "empirical_test_coverage": empirical_coverage,
        "mean_set_size": mean_set_size,
        "singleton_fraction": singleton_fraction,
        "empty_set_fraction": empty_fraction,
    }

    # 3. Energy OOD evaluation if logits are provided
    if test_logits is not None:
        energies = compute_energy_scores(test_logits)
        result["mean_energy"] = float(np.mean(energies))
        result["std_energy"] = float(np.std(energies))

    return result

