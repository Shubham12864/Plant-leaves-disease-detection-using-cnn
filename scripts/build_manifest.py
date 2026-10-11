"""Build a provenance-rich CSV manifest from class-folder image data.

Splits are optional. When requested, every image must have a stable group_id in
metadata CSV. Exact duplicate files and shared group IDs are kept together.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    result: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "relative_path" not in reader.fieldnames:
            raise ValueError("metadata CSV must include a relative_path column")
        for row in reader:
            key = (row.get("relative_path") or "").replace("\\", "/").strip()
            if not key:
                continue
            if key in result:
                raise ValueError(f"duplicate relative_path in metadata CSV: {key}")
            result[key] = {k: (v or "").strip() for k, v in row.items() if k}
    return result


def connected_components(rows: list[dict[str, str]]) -> dict[str, str]:
    """Join images sharing a group_id or exact file hash; return stable component IDs."""
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    seen: dict[tuple[str, str], int] = {}
    for i, row in enumerate(rows):
        for key in (("group", row["group_id"]), ("hash", row["sha256"])):
            if key[1]:
                if key in seen:
                    union(i, seen[key])
                else:
                    seen[key] = i
    roots: dict[int, list[str]] = defaultdict(list)
    for i, row in enumerate(rows):
        roots[find(i)].append(row["relative_path"])
    return {
        path: "grp_" + hashlib.sha256("\n".join(sorted(paths)).encode()).hexdigest()[:12]
        for paths in roots.values()
        for path in paths
    }


def assign_group_splits(rows: list[dict[str, str]], seed: int, ratios: tuple[float, float, float]) -> None:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["split_group"]].append(row)
    names = list(groups)
    random.Random(seed).shuffle(names)
    targets = [ratios[i] * len(rows) for i in range(3)]
    counts = [0, 0, 0]
    # Greedy size balancing keeps each connected group intact; class balance is audited after.
    for group in sorted(names, key=lambda name: -len(groups[name])):
        active = [i for i, ratio in enumerate(ratios) if ratio > 0]
        idx = min(active, key=lambda i: counts[i] / max(targets[i], 1))
        label = ("train", "validation", "test")[idx]
        for row in groups[group]:
            row["split"] = label
        counts[idx] += len(groups[group])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True, help="Folder with one subfolder per class")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. PlantVillage")
    parser.add_argument("--domain", required=True, choices=("controlled", "field", "other"))
    parser.add_argument("--output", type=Path, required=True, help="Output manifest CSV path")
    parser.add_argument("--metadata", type=Path, help="CSV with relative_path and group_id columns")
    parser.add_argument("--source-split", choices=("train", "validation", "test"), help="Preserve an upstream split when scanning one split folder")
    parser.add_argument("--assign-splits", action="store_true", help="Create group-preserving provisional splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.70, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    args = parser.parse_args()

    root = args.image_root.resolve()
    if not root.is_dir():
        parser.error(f"image root does not exist or is not a directory: {root}")
    if any(r < 0 for r in args.ratios) or sum(r > 0 for r in args.ratios) < 2 or abs(sum(args.ratios) - 1) > 1e-6:
        parser.error("split ratios must be non-negative, at least two must be positive, and they must sum to 1")
    if args.source_split and args.assign_splits:
        parser.error("choose either --source-split to preserve upstream splits or --assign-splits to create provisional splits")

    try:
        metadata = load_metadata(args.metadata)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        parser.error("no supported images found under image root")

    rows: list[dict[str, str]] = []
    unreadable: list[str] = []
    for path in paths:
        rel = path.relative_to(root).as_posix()
        if len(Path(rel).parts) < 2:
            parser.error(f"image must be inside a class subfolder: {rel}")
        label = Path(rel).parts[0]
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            unreadable.append(f"{rel}: {exc}")
            continue
        meta = metadata.get(rel, {})
        rows.append({
            "dataset": args.dataset,
            "domain": args.domain,
            "relative_path": rel,
            "label": label,
            "sha256": sha256_file(path),
            "width": str(width),
            "height": str(height),
            "group_id": meta.get("group_id", ""),
            "source_id": meta.get("source_id", ""),
            "plant_id": meta.get("plant_id", ""),
            "split": args.source_split or "",
        })

    if not rows:
        parser.error("no readable images found")
    if args.assign_splits and any(not row["group_id"] for row in rows):
        parser.error("--assign-splits requires a non-empty group_id for every readable image in --metadata")
    if args.assign_splits:
        components = connected_components(rows)
        for row in rows:
            row["split_group"] = components[row["relative_path"]]
        assign_group_splits(rows, args.seed, tuple(args.ratios))
        for row in rows:
            del row["split_group"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["dataset", "domain", "relative_path", "label", "sha256", "width", "height", "group_id", "source_id", "plant_id", "split"]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    class_counts = Counter(row["label"] for row in rows)
    summary = {
        "dataset": args.dataset,
        "domain": args.domain,
        "images_indexed": len(rows),
        "unreadable_images": len(unreadable),
        "classes": dict(sorted(class_counts.items())),
        "splits": dict(Counter(row["split"] for row in rows if row["split"])),
        "seed": args.seed if args.assign_splits else None,
        "split_warning": "Group-preserving size-balanced split; inspect per-class counts and group boundaries before freezing." if args.assign_splits else ("Upstream split preserved; audit cross-split duplicates and grouping." if args.source_split else "No split assigned."),
        "unreadable": unreadable,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(args.output), "summary": str(summary_path), **{k: summary[k] for k in ("images_indexed", "unreadable_images", "classes", "splits")}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
