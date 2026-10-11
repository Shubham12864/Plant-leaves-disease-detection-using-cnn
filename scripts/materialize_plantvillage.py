"""Extract the upstream PlantVillage color split files into class-folder layout."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/PlantVillage/source"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/PlantVillage"))
    args = parser.parse_args()
    archive = args.source_dir / "data.zip"
    if not archive.is_file():
        parser.error(f"download the upstream data.zip first: {archive}")
    with ZipFile(archive) as zf:
        available = set(zf.namelist())
        leaf_map_path = args.source_dir / "leaf-map.json"
        leaf_map = json.loads(leaf_map_path.read_text(encoding="utf-8")) if leaf_map_path.exists() else {}
        metadata_rows = []
        for split in ("train", "test"):
            listing = args.source_dir / f"color_{split}.txt"
            if not listing.is_file():
                parser.error(f"missing upstream split file: {listing}")
            files = [line.strip().replace("\\", "/") for line in listing.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            written = 0
            for member in files:
                posix = PurePosixPath(member)
                if posix.is_absolute() or ".." in posix.parts:
                    raise ValueError(f"unsafe path in upstream split list: {member}")
                if member not in available:
                    raise FileNotFoundError(f"split-listed image not found in archive: {member}")
                if len(posix.parts) < 4 or posix.parts[:2] != ("raw", "color"):
                    raise ValueError(f"unexpected path in upstream split list: {member}")
                label, filename = posix.parts[2], posix.parts[-1]
                destination = args.output_dir / split / label / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    with zf.open(member) as src, destination.open("wb") as dst:
                        dst.write(src.read())
                image_id = filename.replace("_final_masked", "")
                if "___" in image_id:
                    image_id = image_id.split("___")[-1]
                image_id = image_id.split("copy")[0]
                for extension in (".jpg", ".JPG", ".png", ".PNG"):
                    image_id = image_id.replace(extension, "")
                key = image_id.lower().strip()
                suggestions = leaf_map.get(key, [])
                if len(suggestions) == 1:
                    group_id = suggestions[0]
                else:
                    group_id = next((s for s in suggestions if label in s), f"fallback_{image_id}")
                metadata_rows.append({
                    "relative_path": f"{label}/{filename}",
                    "group_id": group_id,
                    "source_id": "PlantVillage-HF-color",
                    "plant_id": group_id,
                    "notes": "Upstream leaf map where matched; otherwise upstream filename fallback",
                })
                written += 1
            print(f"{split}: extracted {written} images into {args.output_dir / split}")
        metadata_path = args.output_dir / "metadata.csv"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("relative_path", "group_id", "source_id", "plant_id", "notes"))
            writer.writeheader()
            writer.writerows(metadata_rows)
        print(f"metadata: wrote {len(metadata_rows)} group records to {metadata_path}")

if __name__ == "__main__":
    main()
