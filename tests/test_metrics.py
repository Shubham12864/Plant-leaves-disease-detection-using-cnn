"""Metric sanity. These catch the traps this project actually fell into."""
from pathlib import Path
import csv
import json

import numpy as np
import pytest


def test_balanced_accuracy_equals_macro_recall_by_construction():
    """The project once reported these as two independent pieces of evidence."""
    from plant_disease.metrics import classification_metrics

    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["balanced_accuracy"] == pytest.approx(metrics["macro_recall"], abs=1e-12)


def test_macro_average_is_meaningless_when_classes_have_no_support():
    """Why every evaluator here restricts macro averages to present classes."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 1, 1])
    present_only = 0.75                      # recall of class 0
    naive_macro = (1.0 + 0.0 + 0.0) / 3      # includes two absent classes
    assert naive_macro < present_only
    assert not np.isclose(naive_macro, present_only)


def test_field_metrics_json_has_the_required_fields(repo: Path):
    path = repo / "reports" / "field_gap" / "b0_plantdoc_test" / "field_metrics.json"
    if not path.is_file():
        pytest.skip("field metrics not generated")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    for key in ("accuracy", "accuracy_ci95", "macro_f1", "out_of_mapping_rate",
                "images_scored", "plantdoc_classes_scored"):
        assert key in metrics, key
    low, high = metrics["accuracy_ci95"]
    assert low <= metrics["accuracy"] <= high
    assert metrics["images_scored"] == 236
    assert metrics["class_count_note"]


def test_field_per_class_only_contains_classes_with_support(repo: Path):
    path = repo / "reports" / "field_gap" / "b0_plantdoc_test" / "per_class.csv"
    if not path.is_file():
        pytest.skip("per-class report not generated")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows
    for row in rows:
        assert int(row["support"]) > 0, row["class"]


def test_predictions_cover_every_scored_image(repo: Path):
    directory = repo / "reports" / "field_gap" / "b0_plantdoc_test"
    if not directory.is_dir():
        pytest.skip("field predictions not generated")
    rows = list(csv.DictReader((directory / "predictions.csv").open(encoding="utf-8")))
    metrics = json.loads((directory / "field_metrics.json").read_text(encoding="utf-8"))
    assert len(rows) == metrics["images_scored"]
    recomputed = sum(int(r["correct"]) for r in rows) / len(rows)
    assert recomputed == pytest.approx(metrics["accuracy"], abs=1e-9)
