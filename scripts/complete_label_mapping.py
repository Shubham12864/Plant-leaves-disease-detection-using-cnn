#!/usr/bin/env python3
"""Resolve every PlantDoc class to a PlantVillage class, or record why not.

The audit left this file as the blocker for all field-domain evaluation:
``data/label_mapping_review.csv`` mapped 16 of 28 PlantDoc classes and marked
every one of them ``review_required``, so nothing could be scored.

This script writes a complete, decided map. It does not invent evidence: each
row carries the relationship type and the specific caveat that still applies.
Where a PlantDoc label is genuinely ambiguous (for example "Corn leaf blight"
can mean Northern corn leaf blight or southern corn leaf blight), the row says
so and names the assumption being made, so a reviewer can reject it.

Counts come from the PlantDoc manifests, so the file also shows the statistical
support behind each class. That matters: ``Tomato two spotted spider mites leaf``
has 2 training images, which cannot support a per-class claim.

Usage:
    python scripts/complete_label_mapping.py
    python scripts/complete_label_mapping.py --dataset-root data --check-output
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# PlantDoc label -> (PlantVillage label, relationship, status, note)
# status: confirmed | confirmed_with_caveat | ambiguous_rejected
MAPPING = {
    # --- apple -------------------------------------------------------------
    "Apple Scab Leaf": ("Apple___Apple_scab", "same disease", "confirmed",
                        "Venturia inaequalis scab on apple foliage; both datasets name the disease."),
    "Apple rust leaf": ("Apple___Cedar_apple_rust", "synonym", "confirmed",
                        "Gymnosporangium cedar-apple rust; PlantDoc drops the alternate host."),
    "Apple leaf": ("Apple___healthy", "healthy class", "confirmed",
                   "Healthy reference imagery; no lesion present by construction."),
    # --- bell pepper -------------------------------------------------------
    "Bell_pepper leaf": ("Pepper,_bell___healthy", "healthy class", "confirmed",
                         "Healthy bell pepper foliage."),
    "Bell_pepper leaf spot": ("Pepper,_bell___Bacterial_spot", "likely synonym", "confirmed_with_caveat",
                              "PlantDoc does not name the pathogen. Bacterial spot (Xanthomonas) is the "
                              "matching PlantVillage class, but a viral or fungal spot would be mislabelled. "
                              "Report per-class recall with this caveat attached."),
    # --- blueberry, cherry, peach, raspberry, strawberry -------------------
    "Blueberry leaf": ("Blueberry___healthy", "healthy class", "confirmed", "Healthy highbush blueberry."),
    "Cherry leaf": ("Cherry_(including_sour)___healthy", "healthy class", "confirmed",
                    "Healthy cherry foliage; PlantVillage does not separate sweet and sour here."),
    "Peach leaf": ("Peach___healthy", "healthy class", "confirmed", "Healthy peach foliage."),
    "Raspberry leaf": ("Raspberry___healthy", "healthy class", "confirmed", "Healthy raspberry foliage."),
    "Strawberry leaf": ("Strawberry___healthy", "healthy class", "confirmed", "Healthy strawberry foliage."),
    # --- corn --------------------------------------------------------------
    "Corn Gray leaf spot": ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "same disease", "confirmed",
                            "Cercospora zeae-maydis; PlantVillage concatenates the two names."),
    "Corn rust leaf": ("Corn_(maize)___Common_rust_", "synonym", "confirmed",
                       "Puccinia sorghi common rust."),
    "Corn leaf blight": ("Corn_(maize)___Northern_Leaf_Blight", "assumed synonym", "confirmed_with_caveat",
                         "Ambiguous: 'leaf blight' could be northern (Exserohilum turcicum) or southern "
                         "(Bipolaris maydis) leaf blight. PlantVillage only has northern. Assumed northern; "
                         "state the assumption wherever this class is reported."),
    # --- potato ------------------------------------------------------------
    "Potato leaf early blight": ("Potato___Early_blight", "same disease", "confirmed",
                                 "Alternaria solani."),
    "Potato leaf late blight": ("Potato___Late_blight", "same disease", "confirmed",
                                "Phytophthora infestans."),
    # --- soybean, squash, grape -------------------------------------------
    "Soyabean leaf": ("Soybean___healthy", "healthy class", "confirmed",
                      "PlantDoc spells it 'Soyabean'; the class is healthy soybean foliage."),
    "Squash Powdery mildew leaf": ("Squash___Powdery_mildew", "same disease", "confirmed",
                                   "Powdery mildew on cucurbit foliage."),
    "grape leaf": ("Grape___healthy", "healthy class", "confirmed", "Healthy grape foliage."),
    "grape leaf black rot": ("Grape___Black_rot", "same disease", "confirmed",
                             "Guignardia bidwellii black rot; the audit flagged this as unmapped despite "
                             "the PlantVillage class existing."),
    # --- tomato ------------------------------------------------------------
    "Tomato Early blight leaf": ("Tomato___Early_blight", "same disease", "confirmed", "Alternaria solani on tomato."),
    "Tomato Septoria leaf spot": ("Tomato___Septoria_leaf_spot", "same disease", "confirmed",
                                  "Septoria lycopersici."),
    "Tomato leaf": ("Tomato___healthy", "healthy class", "confirmed", "Healthy tomato foliage."),
    "Tomato leaf bacterial spot": ("Tomato___Bacterial_spot", "same disease", "confirmed",
                                   "Xanthomonas bacterial spot/speck on tomato."),
    "Tomato leaf late blight": ("Tomato___Late_blight", "same disease", "confirmed", "Phytophthora infestans on tomato."),
    "Tomato leaf mosaic virus": ("Tomato___Tomato_mosaic_virus", "same disease", "confirmed",
                                 "Tomato mosaic virus."),
    "Tomato leaf yellow virus": ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "synonym", "confirmed",
                                 "PlantDoc's 'yellow virus' is tomato yellow leaf curl virus."),
    "Tomato mold leaf": ("Tomato___Leaf_Mold", "synonym", "confirmed",
                         "Passalora fulva leaf mould."),
    "Tomato two spotted spider mites leaf": ("Tomato___Spider_mites Two-spotted_spider_mite", "same disease",
                                             "confirmed_with_caveat",
                                             "Tetranychus urticae damage. Only 2 training images in PlantDoc, "
                                             "so this class cannot support a per-class claim; report it as "
                                             "anecdotal and keep it out of macro averages or flag it."),
}


def read_plantdoc_counts(dataset_root: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    manifest_dir = dataset_root / "data" / "manifests"
    for split, filename in (("train", "plantdoc_train.csv"), ("test", "plantdoc_test.csv")):
        path = manifest_dir / filename
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                label = row.get("label") or row.get("plantdoc_label") or ""
                counts.setdefault(label, {"train": 0, "test": 0})
                counts[label][split] = counts[label].get(split, 0) + 1
    return counts


def plantvillage_classes(dataset_root: Path) -> set[str]:
    path = dataset_root / "data" / "manifests" / "plantvillage_train.csv"
    if not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["label"] for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=ROOT,
                        help="Repository root containing data/manifests")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--check-output", action="store_true",
                        help="Validate every PlantVillage target exists in the PlantVillage manifest")
    args = parser.parse_args()

    output = args.output or (args.dataset_root / "data" / "label_mapping.csv")
    counts = read_plantdoc_counts(args.dataset_root)
    pv_classes = plantvillage_classes(args.dataset_root)

    rows = []
    for plantdoc_label, (pv_label, relationship, status, note) in sorted(MAPPING.items()):
        split_counts = counts.get(plantdoc_label, {})
        rows.append(
            {
                "plantdoc_label": plantdoc_label,
                "plantvillage_label": pv_label,
                "relationship": relationship,
                "status": status,
                "review_note": note,
                "plantdoc_train_images": split_counts.get("train", 0),
                "plantdoc_test_images": split_counts.get("test", 0),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    confirmed = sum(1 for r in rows if r["status"].startswith("confirmed"))
    print(f"Wrote {output} with {len(rows)} PlantDoc classes")
    print(f"  confirmed: {confirmed}  caveated: {sum(1 for r in rows if r['status'].endswith('caveat'))}")

    if counts:
        unmapped = sorted(set(counts) - set(MAPPING))
        if unmapped:
            print(f"  WARNING PlantDoc labels present in manifests but not mapped: {unmapped}")

    if args.check_output:
        if not pv_classes:
            print("  PlantVillage manifest not found; skipped target validation")
            return
        unknown = sorted({r["plantvillage_label"] for r in rows} - pv_classes)
        if unknown:
            raise SystemExit(f"These PlantVillage targets do not exist in the manifest: {unknown}")
        print(f"  all {len({r['plantvillage_label'] for r in rows})} PlantVillage targets exist in the manifest")
        covered = {r["plantvillage_label"] for r in rows}
        print(f"  PlantVillage classes with no PlantDoc counterpart ({len(pv_classes - covered)}): "
              f"{sorted(pv_classes - covered)}")


if __name__ == "__main__":
    main()
