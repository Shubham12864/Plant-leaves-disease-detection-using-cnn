"""Extract PlantDoc train/test images and make non-Windows-safe filenames reversible."""
from __future__ import annotations
import argparse
import csv
import hashlib
import re
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

INVALID = re.compile(r'[<>:"/\\|?*]')
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_filename(name: str, source_path: str) -> str:
    candidate = INVALID.sub("_", name).rstrip(" .")
    if not candidate:
        candidate = "image"
    stem, suffix = Path(candidate).stem, Path(candidate).suffix
    if stem.upper() in RESERVED:
        stem = "_" + stem
    if len(candidate) > 180:
        candidate = stem[:145] + "__" + hashlib.sha256(source_path.encode()).hexdigest()[:12] + suffix[:16]
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/PlantDoc/source/PlantDoc-Dataset.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/PlantDoc"))
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"archive not found: {args.archive}")

    mappings = []
    used: dict[str, str] = {}
    counts = {"train": 0, "test": 0}
    with ZipFile(args.archive) as zf:
        for member in zf.infolist():
            raw = member.filename
            parts = PurePosixPath(raw).parts
            split_at = next((i for i, part in enumerate(parts[:-1]) if part in ("train", "test")), None)
            if split_at is None or len(parts) < split_at + 3 or member.is_dir():
                continue
            split = parts[split_at]
            class_name, original_name = parts[split_at + 1], parts[-1]
            if Path(original_name).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            renamed = safe_filename(original_name, raw)
            relative = Path(split) / class_name / renamed
            key = relative.as_posix().casefold()
            if key in used and used[key] != raw:
                stem, suffix = Path(renamed).stem, Path(renamed).suffix
                renamed = f"{stem}__{hashlib.sha256(raw.encode()).hexdigest()[:12]}{suffix}"
                relative = Path(split) / class_name / renamed
                key = relative.as_posix().casefold()
                if key in used:
                    raise RuntimeError(f"filename collision after disambiguation: {raw}")
            used[key] = raw
            destination = args.output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, destination.open("wb") as target:
                target.write(source.read())
            mappings.append({"source_path": raw, "saved_relative_path": relative.as_posix(), "source_filename": original_name})
            counts[split] += 1

        # Preserve source attribution files beside the archive.
        source_dir = args.output_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        for member in zf.infolist():
            name = PurePosixPath(member.filename).name
            if name.lower() in {"readme.md", "license.txt"} and not member.is_dir():
                (source_dir / ("UPSTREAM_" + name)).write_bytes(zf.read(member))

    map_path = args.output_dir / "source" / "filename_map.csv"
    with map_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("source_path", "saved_relative_path", "source_filename"))
        writer.writeheader()
        writer.writerows(mappings)
    print(f"train: {counts['train']} images; test: {counts['test']} images")
    print(f"filename map: {map_path}")

if __name__ == "__main__":
    main()
