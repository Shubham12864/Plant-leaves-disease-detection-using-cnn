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
