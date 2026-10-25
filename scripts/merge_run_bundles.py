#!/usr/bin/env python3
"""Merge several run bundles into one zip, with a provenance manifest.

Kaggle hands back one zip per session (``training_runs_pilot.zip``,
``sweep_all.zip``, and later the PlantDoc bundles). This merges them into a
single archive without ever guessing: every entry is deduplicated by content
hash, and any path that appears twice with *different* content is reported as a
conflict instead of being silently overwritten.

    python scripts/merge_run_bundles.py out.zip a.zip b.zip [--root runs]

Layout: entries keep their path relative to the archive root; the output also
carries ``MANIFEST.csv`` (path, source archive, sha256, bytes) so any number in
a document can be traced back to the file and the session it came from.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

# Checkpoints are large and are not needed to reproduce a table, but they are
# part of the record, so they are kept by default and only skipped on request.
SKIP_SUFFIXES = (".pyc", ".pyo")
SKIP_PARTS = ("__pycache__", ".ipynb_checkpoints")


def should_skip(name: str) -> bool:
    if name.endswith(SKIP_SUFFIXES):
        return True
    return any(part in name.split("/") for part in SKIP_PARTS)


def normalise(name: str, strip_prefixes: tuple[str, ...]) -> str:
    """Drop a leading wrapper folder so bundles from different tools line up."""
    parts = [p for p in name.split("/") if p not in ("", ".")]
    while parts and (parts[0] in strip_prefixes or parts[0].startswith("runs")):
        if len(parts) == 1:
            break
        parts = parts[1:]
    return "/".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--strip", nargs="*", default=["kaggle/working", "working"],
                        help="leading folders to drop from entry names")
    parser.add_argument("--skip-checkpoints", action="store_true",
                        help="omit *.pt files to keep the archive small")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    missing = [str(p) for p in args.inputs if not p.is_file()]
    if missing:
        raise SystemExit(f"missing input archive(s): {missing}")

    # path -> list of (sha, source archive, payload bytes)
    collected: dict[str, list[tuple[str, str, bytes]]] = defaultdict(list)
    read_order: list[str] = []

    for archive in args.inputs:
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = normalise(info.filename, tuple(args.strip))
                if not name or should_skip(name):
                    continue
                if args.skip_checkpoints and name.endswith(".pt"):
                    continue
                payload = zf.read(info)
                digest = hashlib.sha256(payload).hexdigest()
                collected[name].append((digest, archive.name, payload))
                read_order.append(name)

        print(f"read {archive.name}: {len(zf.namelist())} entries")

    conflicts, merged = [], {}
    for name, versions in collected.items():
        digests = {digest for digest, _, _ in versions}
        if len(digests) == 1:
            # Identical content from every archive that carried it.
            merged[name] = (versions[0][0], "+".join(sorted({src for _, src, _ in versions})),
                            versions[0][2])
        else:
            conflicts.append((name, [(digest[:12], src, len(payload))
                                     for digest, src, payload in versions]))
            # Last archive wins, but the conflict is reported and recorded.
            digest, src, payload = versions[-1]
            merged[name] = (digest, src, payload)

    total_bytes = sum(len(payload) for _, _, payload in merged.values())
    print(f"\nunique entries: {len(merged)}  ({total_bytes/1e6:.1f} MB uncompressed)")
    if conflicts:
        print(f"CONFLICTS: {len(conflicts)} path(s) differ between archives")
        for name, versions in conflicts[:20]:
            print(f"  {name}")
            for digest, src, size in versions:
                print(f"      {digest}  {size:>10d}  {src}")
        if len(conflicts) > 20:
            print(f"  ... and {len(conflicts) - 20} more")
    else:
        print("no conflicts")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = io.StringIO()
    writer = csv.writer(manifest)
    writer.writerow(["path", "source_archives", "sha256", "bytes"])
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        for name in sorted(merged):
            digest, src, payload = merged[name]
            out.writestr(name, payload)
            writer.writerow([name, src, digest, len(payload)])
        out.writestr("MANIFEST.csv", manifest.getvalue())
        if conflicts:
            lines = ["path,sha12,bytes,source_archive"]
            for name, versions in conflicts:
                for digest, src, size in versions:
                    lines.append(f"{name},{digest},{size},{src}")
            out.writestr("CONFLICTS.csv", "\n".join(lines) + "\n")

    print(f"\nwrote {args.output} ({args.output.stat().st_size/1e6:.1f} MB on disk)")
    print("includes MANIFEST.csv" + (" and CONFLICTS.csv" if conflicts else ""))


if __name__ == "__main__":
    main()
