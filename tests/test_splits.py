"""Split integrity. If these fail, every number in the project is suspect."""
from pathlib import Path

import pandas as pd


def test_no_image_appears_in_two_splits(manifests: Path):
    dev = pd.read_csv(manifests / "plantvillage_development.csv")
    sets = {name: set(group.relative_path) for name, group in dev.groupby("split")}
    assert set(sets) == {"train", "validation"}, sorted(sets)
    overlap = sets["train"] & sets["validation"]
    assert not overlap, f"{len(overlap)} paths appear in both splits"


def test_no_duplicate_rows_within_a_split(manifests: Path):
    dev = pd.read_csv(manifests / "plantvillage_development.csv")
    duplicated = dev.duplicated(subset=["relative_path"]).sum()
    assert duplicated == 0, f"{duplicated} duplicated relative_path rows"


def test_frozen_counts_are_what_the_project_reports(manifests: Path):
    dev = pd.read_csv(manifests / "plantvillage_development.csv")
    counts = dev.split.value_counts().to_dict()
    assert counts["train"] == 36762, counts
    assert counts["validation"] == 6544, counts


def test_every_class_appears_in_both_splits(manifests: Path):
    dev = pd.read_csv(manifests / "plantvillage_development.csv")
    classes = set(dev.label)
    assert len(classes) == 38, len(classes)
    for split, group in dev.groupby("split"):
        missing = classes - set(group.label)
        assert not missing, f"{split} is missing {sorted(missing)}"


def test_locked_test_split_is_disjoint_from_development(manifests: Path):
    dev = pd.read_csv(manifests / "plantvillage_development.csv")
    test = pd.read_csv(manifests / "plantvillage_test.csv")
    overlap = set(dev.relative_path) & set(test.relative_path)
    assert not overlap, f"{len(overlap)} paths leak between development and the locked test split"
    assert len(test) == 10709, len(test)


def test_plantdoc_mapping_is_complete_and_decided(manifests: Path):
    mapping = pd.read_csv(manifests / "plantdoc_to_plantvillage_map.csv")
    assert len(mapping) == 28, len(mapping)
    assert mapping.plantdoc_label.is_unique
    assert mapping.plantvillage_label.notna().all()
    # No row may be left in the old undecided state.
    assert "review_required" not in set(mapping.status), set(mapping.status)
    caveated = mapping[mapping.status == "confirmed_with_caveat"]
    assert len(caveated) == 3, f"expected 3 caveated mappings, found {len(caveated)}"
