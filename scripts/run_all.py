#!/usr/bin/env python3
"""Reproduce every number in the project from a fresh clone (roadmap S25).

Two levels, because the expensive part is training and the cheap part is
everything else.

    python scripts/run_all.py --check        # minutes: tests, tables, registry
    python scripts/run_all.py --evaluate     # + field evaluation on PlantDoc (CPU)
    python scripts/run_all.py --all          # everything local; training is NOT included

Training is deliberately excluded. The 14-run sweep needs Kaggle GPUs and lives
in notebooks/08_Overnight_Sweep.ipynb; a script that silently starts a six-hour
job on a laptop would be a worse tool.

Each step prints PASS/FAIL and the script exits non-zero if any step fails, so it
can be used as a pre-submission check.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(label: str, command: list[str]) -> bool:
    print(f"\n=== {label} ===")
    print("  " + " ".join(command))
    result = subprocess.run(command, cwd=ROOT)
    ok = result.returncode == 0
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="tests, tables and registry only")
    parser.add_argument("--evaluate", action="store_true", help="also re-run the PlantDoc field evaluation")
    parser.add_argument("--all", action="store_true", help="everything local")
    args = parser.parse_args()

    if not any((args.check, args.evaluate, args.all)):
        parser.print_help()
        return

    steps: list[tuple[str, list[str]]] = [
        ("unit and contract tests", [PY, "-m", "pytest", "tests/", "-q"]),
        ("table build", [PY, "scripts/build_tables.py"]),
        ("registry check", [PY, "scripts/build_tables.py", "--verify"]),
        ("batch composition check", [PY, "scripts/check_batch_purity.py"]),
        ("worker command contract", [PY, "scripts/test_sweep_commands.py"]),
        ("overnight failure paths", [PY, "scripts/test_overnight_logic.py"]),
    ]

    if args.evaluate or args.all:
        steps.append((
            "field evaluation (reference model)",
            [PY, "scripts/evaluate_plantdoc.py",
             "--checkpoint", "checkpoints/efficientnet_b0/efficientnet_b0_best.pt",
             "--arch", "efficientnet_b0", "--split", "test", "--device", "cpu",
             "--out", "reports/field_gap/b0_plantdoc_test"],
        ))

    results = [(label, run(label, command)) for label, command in steps]

    print("\n" + "=" * 66)
    failed = [label for label, ok in results if not ok]
    for label, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print("=" * 66)
    if failed:
        print(f"{len(failed)} step(s) failed: {failed}")
        raise SystemExit(1)
    print(f"all {len(results)} steps passed")

    print("\nNot reproduced here, by design:")
    print("  - the 14-run training sweep (Kaggle, 2x T4): notebooks/08_Overnight_Sweep.ipynb")
    print("  - locked-test evaluation of new checkpoints (Kaggle): notebooks/04_Locked_Test_Evaluation.ipynb")
    print("  - background augmentation (S19): blocked on S18, no licence-clean corpus")


if __name__ == "__main__":
    main()
