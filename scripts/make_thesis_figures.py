#!/usr/bin/env python3
"""Build the figures used in the dissertation.

Every figure is drawn from a stored artifact. Nothing is sketched by hand, so a
figure cannot disagree with the table it sits beside.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "thesis" / "figures"
REPORTS = ROOT / "reports"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def reliability(path: Path, out: Path, title: str) -> None:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    keep = [r for r in rows if int(r["count"]) > 0]
    conf = [float(r["mean_confidence"]) for r in keep]
    acc = [float(r["accuracy"]) for r in keep]
    n = [int(r["count"]) for r in keep]
    total = sum(n)

    # Scatter with area proportional to bin count, and no connecting line. A line
    # would let a bin holding one image draw the same visual weight as the bin
    # holding 10,590, which reads as miscalibration that is not there.
    size = [18 + 320 * (c / total) ** 0.5 for c in n]

    fig, ax = plt.subplots(figsize=(5.9, 4.9))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="0.55", label="perfect calibration")
    ax.scatter(conf, acc, s=size, color="#1f4e79", alpha=0.75,
               edgecolor="white", linewidth=0.8, label="measured bins")
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.text(0.98, 0.03, f"marker area $\\propto$ bin count\nlargest bin: {max(n):,} of {total:,} images",
            transform=ax.transAxes, fontsize=7.5, color="0.35", ha="right")
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}  ({len(keep)} non-empty bins, {total:,} images)")


def swap_chart(path: Path, out: Path) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    labels = ["chance\n(1 in 38)", "same-class\ncontrol", "background\nswap", "leaf\nswap"]
    vals = [d["chance_adoption"], d["same_class_control_flip"],
            d["background_swap_adoption"], d["leaf_swap_adoption"]]
    errs = [0.0] + [
        (d["same_class_control_flip_ci95"][1] - d["same_class_control_flip_ci95"][0]) / 2,
        (d["background_swap_adoption_ci95"][1] - d["background_swap_adoption_ci95"][0]) / 2,
        (d["leaf_swap_adoption_ci95"][1] - d["leaf_swap_adoption_ci95"][0]) / 2,
    ]
    colors = ["0.7", "0.55", "#c55a11", "#1f4e79"]

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    bars = ax.bar(labels, vals, yerr=errs, capsize=4, color=colors,
                  edgecolor="black", linewidth=0.6)
    for b, v, e in zip(bars, vals, errs):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v + e),
                    textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=9)
    ax.set_ylabel("adoption rate (prediction follows donor)")
    ax.set_ylim(0, max(vals) * 1.28)
    ax.set_title("What moves the prediction, 298 validation images", fontsize=9)
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


def downscale(src: Path, out: Path, max_w: int = 1600) -> None:
    im = Image.open(src)
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    im.save(out, optimize=True)
    print(f"  {out.name}  ({im.width}x{im.height})")


def copy(src: Path, out: Path) -> None:
    shutil.copy(src, out)
    print(f"  {out.name}  (copied from {src.relative_to(ROOT)})")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lock = REPORTS / "model_comparison" / "locked_test_efficientnet_b0"
    print("building figures:")

    reliability(lock / "validation" / "reliability_bins.csv",
                OUT / "fig_reliability_val.png", "PlantVillage validation")
    if (lock / "test" / "reliability_bins.csv").exists():
        reliability(lock / "test" / "reliability_bins.csv",
                    OUT / "fig_reliability_test.png", "PlantVillage locked test")
    swap_chart(REPORTS / "mechanism" / "swap_test_b0" / "swap_test.json",
               OUT / "fig_swap.png")
    copy(REPORTS / "field_gap" / "field_gap_bars.png", OUT / "fig_field_gap.png")
    copy(REPORTS / "paper" / "figures" / "plantdoc_confusion.png", OUT / "fig_confusion.png")
    copy(REPORTS / "paper" / "figures" / "coverage_curve.png", OUT / "fig_coverage.png")
    downscale(REPORTS / "paper" / "figures" / "saliency_grid.png", OUT / "fig_saliency.png")
    print(f"done: {len(list(OUT.glob('*.png')))} figures in thesis/figures/")


if __name__ == "__main__":
    main()
