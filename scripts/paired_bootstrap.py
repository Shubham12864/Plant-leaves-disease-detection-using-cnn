#!/usr/bin/env python3
"""Paired bootstrap on the PlantVillage to PlantDoc drop (roadmap S19 gate).

Two models are compared on the same 236 field images, so the comparison is
paired: for each image both models either succeed or fail, and what matters is the
difference in the drop from the laboratory benchmark.

Two quantities are reported and they answer different questions.

**Field accuracy difference.** How much better is the treatment in the field,
signed. Bootstrap over images, resampling the *same* images for both models so the
pairing is respected.

**Drop difference.** How much less does the treatment lose going from lab to
field. This is the H3 quantity: if augmentation helps, the treatment's drop should
be smaller than the baseline's.

Reporting only the first would let a treatment look good while actually being worse
in the lab. Reporting both is the honest pairing.

    python scripts/paired_bootstrap.py \
        --baseline reports/field_gap/b0_plantdoc_test/predictions.csv \
        --treatment reports/field_gap/composite_b0_plantdoc_test/predictions.csv \
        --baseline-lab 0.99280 --treatment-lab 0.99 \
        --out reports/field_gap/paired_b0.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict[str, tuple[int, float]]:
    """relative_path -> (correct, confidence). Keyed on path so pairing is exact."""
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["relative_path"]] = (int(row["correct"] == "1"), float(row["confidence"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--baseline-lab", type=float, required=True,
                        help="baseline PlantVillage accuracy")
    parser.add_argument("--treatment-lab", type=float, required=True,
                        help="treatment PlantVillage accuracy")
    parser.add_argument("--rounds", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base, treat = load(args.baseline), load(args.treatment)
    shared = sorted(set(base) & set(treat))
    if len(shared) != len(base) or len(shared) != len(treat):
        raise SystemExit(
            f"the two prediction files cover different images: baseline {len(base)}, "
            f"treatment {len(treat)}, shared {len(shared)}. A paired test needs identical images."
        )

    b_ok = np.array([base[p][0] for p in shared], dtype=np.float64)
    t_ok = np.array([treat[p][0] for p in shared], dtype=np.float64)
    n = len(shared)

    base_acc, treat_acc = b_ok.mean(), t_ok.mean()
    base_drop = args.baseline_lab - base_acc
    treat_drop = args.treatment_lab - treat_acc

    rng = np.random.default_rng(args.seed)
    deltas = np.empty(args.rounds)
    drop_deltas = np.empty(args.rounds)
    for r in range(args.rounds):
        idx = rng.integers(0, n, n)          # same resample for both models
        deltas[r] = t_ok[idx].mean() - b_ok[idx].mean()
        drop_deltas[r] = ((args.baseline_lab - b_ok[idx].mean())
                          - (args.treatment_lab - t_ok[idx].mean()))

    def ci(sample):
        return [float(np.percentile(sample, 2.5)), float(np.percentile(sample, 97.5))]

    both_wrong = int(((b_ok == 0) & (t_ok == 0)).sum())
    treatment_fixes = int(((b_ok == 0) & (t_ok == 1)).sum())
    treatment_breaks = int(((b_ok == 1) & (t_ok == 0)).sum())

    summary = {
        "images": n,
        "baseline_lab_accuracy": args.baseline_lab,
        "treatment_lab_accuracy": args.treatment_lab,
        "baseline_field_accuracy": round(float(base_acc), 6),
        "treatment_field_accuracy": round(float(treat_acc), 6),
        "field_delta_pp": round(float((treat_acc - base_acc) * 100), 2),
        "field_delta_ci95_pp": [round(v * 100, 2) for v in ci(deltas)],
        "baseline_drop_pp": round(float(base_drop * 100), 2),
        "treatment_drop_pp": round(float(treat_drop * 100), 2),
        "drop_improvement_pp": round(float((base_drop - treat_drop) * 100), 2),
        "drop_improvement_ci95_pp": [round(v * 100, 2) for v in ci(drop_deltas)],
        "rounds": args.rounds,
        "paired_movements": {
            "both_wrong": both_wrong,
            "treatment_fixes": treatment_fixes,
            "treatment_breaks": treatment_breaks,
        },
        "verdict": None,
    }

    low, high = summary["drop_improvement_ci95_pp"]
    if low > 0:
        summary["verdict"] = "H3 supported: the drop is smaller for the treatment, CI excludes zero"
    elif high < 0:
        summary["verdict"] = "H3 contradicted: the treatment's drop is larger, CI excludes zero"
    else:
        summary["verdict"] = "H3 not supported: the CI on the drop improvement includes zero"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"images                    {n}")
    print(f"baseline field accuracy   {base_acc:.4f}  (lab {args.baseline_lab:.4f}, drop {base_drop*100:.1f} pp)")
    print(f"treatment field accuracy  {treat_acc:.4f}  (lab {args.treatment_lab:.4f}, drop {treat_drop*100:.1f} pp)")
    print(f"field delta               {summary['field_delta_pp']:+.2f} pp  "
          f"CI {summary['field_delta_ci95_pp']}")
    print(f"drop improvement          {summary['drop_improvement_pp']:+.2f} pp  "
          f"CI {summary['drop_improvement_ci95_pp']}")
    print(f"paired movements          fixes {treatment_fixes}, breaks {treatment_breaks}, "
          f"both wrong {both_wrong}")
    print(f"\n{summary['verdict']}")


if __name__ == "__main__":
    main()
