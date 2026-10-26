#!/usr/bin/env python3
"""Prove a background corpus does not overlap PlantDoc (roadmap S18).

H3 hangs on this check. If a single augmentation background came from PlantDoc,
the field evaluation would be seeing training data and the −78.4 pp result would
be void. "They are different datasets" is not evidence; a hash comparison is.

    python scripts/check_background_disjointness.py --backgrounds data/raw/DeepWeeds

Exit code 1 means overlap was found and training must not start.

What this does and does not catch
---------------------------------
It compares SHA-256 of every background file against the SHA-256 values the
PlantDoc manifests already record per image, and against PlantVillage for good
measure. That catches the same file, and any file byte-identical after
re-encoding.

It does **not** catch a near-duplicate that was re-saved at a different quality,
cropped, or resized. Catching that needs perceptual hashing, which is a different
tool and is not implemented here. The output says so, so nobody reads a pass as
stronger than it is.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_hashes(names: list[str]) -> tuple[dict[str, str], int]:
    """Collect the recorded sha256 values from the given manifest files."""
    known, missing = {}, 0
    for name in names:
        path = MANIFESTS / name
        if not path.is_file():
            print(f"  warning: {name} not found, skipping")
            continue
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                digest = (row.get("sha256") or "").strip().lower()
                if digest:
                    known[digest] = row.get("relative_path", "?")
                else:
                    missing += 1
    return known, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backgrounds", type=Path, required=True,
                        help="directory containing candidate background images")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "mechanism" / "disjointness.json")
    args = parser.parse_args()

    if not args.backgrounds.is_dir():
        raise SystemExit(f"background directory not found: {args.backgrounds}")

    plantdoc, pd_missing = manifest_hashes(["plantdoc_development.csv", "plantdoc_test.csv"])
    plantvillage, pv_missing = manifest_hashes(["plantvillage_development.csv", "plantvillage_test.csv"])
    print(f"PlantDoc hashes on record : {len(plantdoc):,} ({pd_missing} rows had none)")
    print(f"PlantVillage hashes       : {len(plantvillage):,} ({pv_missing} rows had none)")

    candidates = [p for p in args.backgrounds.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not candidates:
        raise SystemExit(f"no images found under {args.backgrounds}")
    print(f"candidate backgrounds     : {len(candidates):,}")

    overlaps, unreadable = [], []
    for i, path in enumerate(candidates, start=1):
        try:
            digest = sha256_of(path)
        except OSError as exc:
            unreadable.append((str(path), repr(exc)))
            continue
        if digest in plantdoc:
            overlaps.append({"background": str(path.relative_to(args.backgrounds)),
                             "collides_with": plantdoc[digest], "corpus": "PlantDoc"})
        elif digest in plantvillage:
            overlaps.append({"background": str(path.relative_to(args.backgrounds)),
                             "collides_with": plantvillage[digest], "corpus": "PlantVillage"})
        if i % 2000 == 0:
            print(f"  hashed {i:,}/{len(candidates):,}", flush=True)

    verdict = "FAIL: overlap found, do not train" if overlaps else "PASS: no byte-identical overlap"
    payload = {
        "backgrounds": str(args.backgrounds),
        "candidate_images": len(candidates),
        "plantdoc_hashes": len(plantdoc),
        "plantvillage_hashes": len(plantvillage),
        "overlaps": overlaps,
        "unreadable": unreadable,
        "verdict": verdict,
        "limitation": ("exact SHA-256 only; a re-encoded, cropped or resized near-duplicate "
                       "would not be caught, and no perceptual hashing was performed"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{verdict}")
    if overlaps:
        for hit in overlaps[:20]:
            print(f"  {hit['background']} == {hit['collides_with']} ({hit['corpus']})")
        if len(overlaps) > 20:
            print(f"  ... and {len(overlaps) - 20} more")
    print(f"wrote {args.out}")
    if overlaps:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
