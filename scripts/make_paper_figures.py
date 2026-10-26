#!/usr/bin/env python3
"""Figures for the results section (roadmap S27).

Three that do not exist yet: the PlantDoc confusion matrix, the coverage-accuracy
curve, and a saliency grid. Axes start at zero or say where they are truncated.

    python scripts/make_paper_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
FIELD = REPO / "reports" / "field_gap"
MECH = REPO / "reports" / "mechanism"
OUT = REPO / "reports" / "paper" / "figures"


def confusion_figure() -> None:
    path = FIELD / "b0_plantdoc_test" / "confusion.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    labels = [r["true\\predicted"] for r in rows]
    matrix = np.array([[int(r[c]) for c in labels] for r in rows])

    fig, ax = plt.subplots(figsize=(11, 9.5))
    ax.imshow(matrix, cmap="magma_r", aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    short = [l.split("___")[0][:14] + "/" + l.split("___")[1][:16] if "___" in l else l[:26] for l in labels]
    ax.set_xticklabels(short, rotation=90, fontsize=5.5)
    ax.set_yticklabels(short, fontsize=5.5)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"EfficientNet-B0 on PlantDoc, {int(matrix.sum())} field images\n"
                 f"{int(np.trace(matrix))} correct on the diagonal "
                 f"({np.trace(matrix)/max(matrix.sum(),1):.1%} accuracy)")
    fig.tight_layout()
    fig.savefig(OUT / "plantdoc_confusion.png", dpi=175)
    plt.close(fig)
    print("wrote plantdoc_confusion.png")


def coverage_figure() -> None:
    summary = json.loads((FIELD / "calibration_summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for model, d in summary.items():
        rows = list(csv.DictReader((FIELD / (model.replace("efficientnet_b0", "b0") + "_plantdoc_test")
                                    / "predictions.csv").open(encoding="utf-8")))
        pairs = sorted(((float(r["confidence"]), int(r["correct"] == "1")) for r in rows), reverse=True)
        coverages = np.arange(0.1, 1.001, 0.05)
        accs = [np.mean([ok for _, ok in pairs[:max(1, int(round(c * len(pairs))))]]) for c in coverages]
        ax.plot(coverages * 100, accs, marker="o", markersize=3, label=model)
    ax.axvline(90, linestyle="--", color="grey", linewidth=1)
    ax.annotate("90% coverage", xy=(90, 0.02), xytext=(72, 0.06), fontsize=8, color="grey")
    ax.set_xlabel("coverage (% of field images retained, most confident first)")
    ax.set_ylabel("accuracy on retained images")
    ax.set_ylim(0, 0.55)
    ax.set_xlim(10, 100)
    ax.set_title("Rejecting low-confidence field images barely raises accuracy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "coverage_curve.png", dpi=170)
    plt.close(fig)
    print("wrote coverage_curve.png")


def saliency_grid() -> None:
    directory = MECH / "gradcam_b0_plantdoc" / "overlays"
    index = list(csv.DictReader((MECH / "gradcam_b0_plantdoc" / "index.csv").open(encoding="utf-8")))
    if not index:
        print("no overlays to grid")
        return
    cols, rows_n = 6, 3
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.3, rows_n * 2.5))
    for ax in axes.ravel():
        ax.axis("off")
    for i, record in enumerate(index[: cols * rows_n]):
        ax = axes[i // cols][i % cols]
        with Image.open(directory / record["overlay"]) as image:
            ax.imshow(np.asarray(image))
        ax.set_title(f"{record['group'].replace('_', ' ')}\ntrue {record['true_class'].split('___')[-1][:16]}"
                     f"\npred {record['predicted_class'].split('___')[-1][:16]} ({record['confidence']})",
                     fontsize=5.5)
        ax.axis("off")
    fig.suptitle("Grad-CAM on EfficientNet-B0 field predictions (PlantDoc)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "saliency_grid.png", dpi=170)
    plt.close(fig)
    print("wrote saliency_grid.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    confusion_figure()
    coverage_figure()
    saliency_grid()


if __name__ == "__main__":
    main()
