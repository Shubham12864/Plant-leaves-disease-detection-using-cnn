#!/usr/bin/env python3
"""Two figures that carry the field-gap result.

(a) reliability on PlantDoc versus PlantVillage, which is where the model stops
    being merely wrong and starts being confidently wrong;
(b) per-class recall, lab versus field, for every mapped class.

    python scripts/plot_field_gap.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIELD = REPO / "reports" / "field_gap" / "b0_plantdoc_test"
PV_VAL = REPO / "reports" / "model_comparison" / "locked_test_efficientnet_b0" / "validation"
OUT = REPO / "reports" / "field_gap"


def reliability_bins(pairs, edges):
    """pairs: list of (confidence, correct). Returns (conf, acc, weight) per bin."""
    conf_out, acc_out, weight = [], [], []
    total = len(pairs)
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = [(c, ok) for c, ok in pairs if lo <= c < hi]
        if not selected:
            continue
        conf_out.append(float(np.mean([c for c, _ in selected])))
        acc_out.append(float(np.mean([ok for _, ok in selected])))
        weight.append(len(selected) / total)
    return conf_out, acc_out, weight


def main() -> None:
    field_rows = list(csv.DictReader((FIELD / "predictions.csv").open(encoding="utf-8")))
    field_pairs = [(float(r["confidence"]), int(r["correct"] == "1")) for r in field_rows]
    gap = json.loads((FIELD / "gap_analysis.json").read_text(encoding="utf-8"))

    # PlantVillage reliability from the stored bins of the S1 evaluation
    pv_bins = list(csv.DictReader((PV_VAL / "reliability_bins.csv").open(encoding="utf-8")))
    edges = np.linspace(0, 1, 11)
    f_conf, f_acc, _ = reliability_bins(field_pairs, edges)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="perfect calibration")
    ax.plot(f_conf, f_acc, marker="o", linewidth=2,
            label=f"PlantDoc field, ECE {gap['ece_plantdoc']:.3f}", color="#c0392b")
    if pv_bins:
        key_conf = next((k for k in pv_bins[0] if "conf" in k.lower()), None)
        key_acc = next((k for k in pv_bins[0] if "acc" in k.lower()), None)
        if key_conf and key_acc:
            # Bins with no images carry mean_confidence 0.0 and would plot at the
            # origin as a fake data point. Filter on the count, not on truthiness.
            # On a 99.5%-accurate model only 48 of 6,544 images fall below 0.9
            # confidence, so those bins hold 2-29 images and their accuracy is
            # noise. Plot only bins with at least 25 images and state the rule.
            def kept_bin(row):
                raw = row.get("count") or row.get("n") or 0
                return float(raw) >= 25
            kept = [r for r in pv_bins if kept_bin(r)]
            xs = [float(r[key_conf]) for r in kept]
            ys = [float(r[key_acc]) for r in kept]
            ax.plot(xs, ys, marker="s", linewidth=2, color="#27ae60",
                    label="PlantVillage lab, ECE 0.0026 (bins with n≥25)")
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Confidence means the same thing in both domains\n(it does not)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1]
    per_class = sorted(gap["per_class"], key=lambda r: r["pv_recall"] - r["pd_recall"], reverse=True)
    labels = [r["class"].replace("___", " / ").replace("_", " ") for r in per_class]
    pv_recalls = [r["pv_recall"] for r in per_class]
    pd_recalls = [r["pd_recall"] for r in per_class]
    y = np.arange(len(per_class))
    for i in y:
        ax.plot([pv_recalls[i], pd_recalls[i]], [i, i], color="#bbb", linewidth=1, zorder=1)
    ax.scatter(pv_recalls, y, s=26, color="#27ae60", label="PlantVillage (lab)", zorder=2)
    ax.scatter(pd_recalls, y, s=26, color="#c0392b", label="PlantDoc (field)", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("recall")
    ax.set_xlim(-0.02, 1.02)
    ax.set_title(f"Recall collapsed in all {len(per_class)} mapped classes\n"
                 f"mean {np.mean(pv_recalls):.3f} to {np.mean(pd_recalls):.3f}")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = OUT / "field_gap.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"wrote {path}")

    axes[1].set_yticklabels([])
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot([0, 1], [np.mean(pv_recalls)] * 2, color="#27ae60", linewidth=2,
             label=f"PlantVillage mean {np.mean(pv_recalls):.3f}")
    ax2.plot([0, 1], [np.mean(pd_recalls)] * 2, color="#c0392b", linewidth=2,
             label=f"PlantDoc mean {np.mean(pd_recalls):.3f}")
    ax2.set_xticks([]);
    ax2.set_ylabel("per-class recall")
    ax2.set_title("Field gap")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUT / "field_gap_bars.png", dpi=160)
    plt.close(fig2)
    print(f"wrote {OUT / 'field_gap_bars.png'}")


if __name__ == "__main__":
    main()
