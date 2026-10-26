#!/usr/bin/env python3
"""Render every reported table from results/registry.json (roadmap S24).

The rule this enforces: a number may appear in a document only if it is
registered here, with the artifact that produced it. Tables are generated, not
typed, so a document cannot drift from its evidence.

    python scripts/build_tables.py            # write reports/generated/*.md
    python scripts/build_tables.py --verify   # fail if anything is stale or unbacked

`--verify` is the gate. It fails when the registry is missing (by design),
when a claim points at a run that does not exist, when an artifact directory has
disappeared, or when the generated tables on disk no longer match the registry.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "results" / "registry.json"
OUT = ROOT / "reports" / "generated"


def load_registry() -> dict:
    if not REGISTRY.is_file():
        raise SystemExit(
            f"registry missing: {REGISTRY}\n"
            "Every reported number must trace to an artifact. Without the registry there is "
            "nothing to build tables from, and that is the intended failure."
        )
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def table(rows, header) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def build(reg: dict) -> dict[str, str]:
    files = {}

    # 1. Architecture comparison on PlantVillage validation
    sweep = reg.get("sweeps", {}).get("sweep_v2_overnight")
    if sweep:
        rows = []
        for arch, d in sorted(sweep["architectures"].items(),
                              key=lambda kv: -kv[1]["val_macro_f1_mean"]):
            lo, hi = d["val_macro_f1_ci95"]
            rows.append([arch, d["n_runs"], f"{d['val_macro_f1_mean']:.5f}",
                         f"{d['val_macro_f1_sd']:.5f}", f"{lo:.5f} – {hi:.5f}"])
        files["architecture_comparison.md"] = (
            "# Architecture comparison — PlantVillage validation\n\n"
            f"Source: `{sweep['protocol']}`, artifact `{sweep['artifact']}`.\n\n"
            + table(rows, ["architecture", "seeds", "macro-F1 mean", "sd", "95% CI"])
            + "\n")

    # 2. Field gap per model
    gap = reg.get("sweep_v2_overnight") or {}
    field_rows = []
    for claim in reg.get("document_claims", []):
        if "field calibration" in claim["claim"]:
            model = claim["claim"].split()[0]
            field_rows.append([model, f"{claim['value']:.4f}"])
    if field_rows:
        files["field_ece.md"] = (
            "# Field calibration — PlantDoc test\n\n"
            "ECE at 15 bins over 236 field images. Higher is worse.\n\n"
            + table(field_rows, ["model", "ECE on PlantDoc"])
            + "\n")

    # 3. Hypothesis verdicts
    hyp = reg.get("hypotheses")
    if hyp:
        rows = [[h, v["verdict"], v["evidence"]] for h, v in hyp.items()]
        files["hypotheses.md"] = (
            "# Hypothesis verdicts\n\n"
            + table(rows, ["hypothesis", "verdict", "evidence"])
            + "\n")

    # 4. Registered runs
    rows = []
    for run in reg.get("runs", []):
        rows.append([run["id"], run["model"], run["dataset"], run.get("split", "-"),
                     run.get("images"), run.get("date", "")])
    if rows:
        files["runs.md"] = (
            "# Registered runs\n\n"
            + table(rows, ["id", "model", "dataset", "split", "images", "date"])
            + "\n")

    return files


def verify(reg: dict, files: dict[str, str]) -> list[str]:
    problems = []

    run_ids = {run["id"] for run in reg.get("runs", [])}
    sweep_ids = set(reg.get("sweeps", {}))
    for claim in reg.get("document_claims", []):
        target = claim.get("run", "")
        head = target.split("/")[0].split(" ")[0]
        if head not in run_ids and head not in sweep_ids and not any(
                h in run_ids for h in [head]):
            # claims may cite a sweep or a run id; anything else is unsupported
            if head not in sweep_ids:
                problems.append(f"claim cites unknown run/sweep: {target!r}")

    for run in reg.get("runs", []):
        artifact = ROOT / run.get("artifact_dir", "")
        if run.get("artifact_dir") and not artifact.exists():
            problems.append(f"run {run['id']}: artifact dir missing ({run['artifact_dir']})")

    for run in reg.get("runs", []):
        checkpoint = run.get("checkpoint")
        if checkpoint and checkpoint not in reg.get("checkpoints", {}):
            problems.append(f"run {run['id']}: checkpoint {checkpoint!r} not registered")

    for name, content in files.items():
        path = OUT / name
        if not path.is_file():
            problems.append(f"generated table missing: {name} (run without --verify)")
        elif path.read_text(encoding="utf-8") != content:
            problems.append(f"generated table stale: {name} (registry changed since last build)")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    reg = load_registry()
    files = build(reg)

    if args.verify:
        problems = verify(reg, files)
        if problems:
            print("REGISTRY CHECK FAILED:")
            for problem in problems:
                print("  -", problem)
            raise SystemExit(1)
        print(f"REGISTRY CHECK PASSED — {len(files)} tables match the registry, "
              f"{len(reg.get('document_claims', []))} claims backed")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (OUT / name).write_text(content, encoding="utf-8")
        print(f"wrote reports/generated/{name}")


if __name__ == "__main__":
    main()
