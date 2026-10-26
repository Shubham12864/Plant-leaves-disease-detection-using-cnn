#!/usr/bin/env python3
"""Score every sweep checkpoint on PlantDoc (S11b).

Stage C evaluated one checkpoint per architecture, which left the field ranking
carrying no seed variance. The ConvNeXt lead of 8.9 points was therefore a single
observation. This closes that gap: it takes every checkpoint produced by the
fourteen-run sweep, scores each on PlantDoc test, and reports per-architecture
field accuracy with the spread across seeds.

    python scripts/evaluate_field_all_seeds.py --bundle ../overnight_runs.zip

Runs on CPU. 14 checkpoints x 236 images takes a few minutes, not hours.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "reports" / "field_gap" / "all_seeds"
STORE = ROOT / "checkpoints" / "sweep"


def extract(bundle: Path) -> list[tuple[str, int, Path]]:
    """Pull every <arch>_best.pt out of the sweep archive into checkpoints/sweep/."""
    found = []
    with zipfile.ZipFile(bundle) as zf:
        for name in zf.namelist():
            if not name.endswith("_best.pt") or "/seed" not in name:
                continue
            parts = name.split("/")
            if "_gate" in parts or "pilot" in parts:
                continue                     # the gate is a 2-epoch check, not a result
            if len(parts) != 3:
                continue
            arch, seed_dir = parts[0], parts[1]
            seed = int(seed_dir.replace("seed", ""))
            target = STORE / arch / seed_dir / Path(name).name
            if target.is_file():
                found.append((arch, seed, target))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(target, "wb") as out:
                out.write(src.read())
            found.append((arch, seed, target))
            print(f"extracted {arch} seed {seed}")
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    if not args.bundle.is_file():
        raise SystemExit(f"bundle not found: {args.bundle}")

    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints = extract(args.bundle)
    print(f"{len(checkpoints)} checkpoints to score")

    results = []
    for arch, seed, path in checkpoints:
        run_out = OUT / f"{arch}_seed{seed}"
        cached = run_out / "field_metrics.json"
        if cached.is_file():
            metrics = json.loads(cached.read_text(encoding="utf-8"))
            results.append({
                "architecture": arch, "seed": seed,
                "accuracy": round(metrics["accuracy"], 6),
                "accuracy_ci95": [round(v, 6) for v in metrics["accuracy_ci95"]],
                "macro_f1": round(metrics["macro_f1"], 6),
                "mean_confidence": round(metrics["mean_confidence"], 6),
                "out_of_mapping_rate": round(metrics["out_of_mapping_rate"], 6),
            })
            print(f"  {arch:16s} seed {seed:2d}  accuracy {metrics['accuracy']:.4f}  (cached)")
            continue
        command = [PY, str(ROOT / "scripts" / "evaluate_plantdoc.py"),
                   "--checkpoint", str(path), "--arch", arch, "--split", "test",
                   "--device", "cpu", "--bootstrap", str(args.bootstrap),
                   "--out", str(run_out)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FAILED {arch} seed {seed}")
            print(result.stderr[-800:])
            continue
        metrics = json.loads((run_out / "field_metrics.json").read_text(encoding="utf-8"))
        results.append({
            "architecture": arch, "seed": seed,
            "accuracy": round(metrics["accuracy"], 6),
            "accuracy_ci95": [round(v, 6) for v in metrics["accuracy_ci95"]],
            "macro_f1": round(metrics["macro_f1"], 6),
            "mean_confidence": round(metrics["mean_confidence"], 6),
            "out_of_mapping_rate": round(metrics["out_of_mapping_rate"], 6),
        })
        print(f"  {arch:16s} seed {seed:2d}  accuracy {metrics['accuracy']:.4f}")

    summary = {}
    for arch in sorted({r["architecture"] for r in results}):
        rows = [r for r in results if r["architecture"] == arch]
        accs = [r["accuracy"] for r in rows]
        mean = sum(accs) / len(accs)
        sd = (sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)) ** 0.5 if len(accs) > 1 else 0.0
        summary[arch] = {
            "seeds": [r["seed"] for r in rows],
            "n": len(rows),
            "accuracy_mean": round(mean, 6),
            "accuracy_sd": round(sd, 6),
            "accuracy_min": round(min(accs), 6),
            "accuracy_max": round(max(accs), 6),
            "best_seed": rows[accs.index(max(accs))]["seed"],
        }

    payload = {"bundle": str(args.bundle), "images": 236, "classes": 27,
               "per_run": results, "per_architecture": summary}
    (OUT / "field_all_seeds.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{'architecture':16s} {'n':>2s} {'mean':>7s} {'sd':>7s} {'min':>7s} {'max':>7s}")
    for arch, d in sorted(summary.items(), key=lambda kv: -kv[1]["accuracy_mean"]):
        print(f"{arch:16s} {d['n']:2d} {d['accuracy_mean']:7.4f} {d['accuracy_sd']:7.4f} "
              f"{d['accuracy_min']:7.4f} {d['accuracy_max']:7.4f}")
    print(f"\nwrote {OUT / 'field_all_seeds.json'}")


if __name__ == "__main__":
    main()
