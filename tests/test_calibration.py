import numpy as np
import pytest

from plant_disease.calibration import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
    selective_risk_coverage,
    summarise,
    to_markdown_table,
)


def _one_hot(labels, n_classes=3):
    probs = np.zeros((len(labels), n_classes))
    probs[np.arange(len(labels)), labels] = 1.0
    return probs


def test_ece_is_zero_for_confident_and_correct_predictions():
    y = np.array([0, 1, 2, 1])
    assert expected_calibration_error(_one_hot(y), y) == pytest.approx(0.0)


def test_ece_is_high_for_confident_and_wrong_predictions():
    y = np.array([0, 1, 2, 1])
    wrong = _one_hot((y + 1) % 3)
    assert expected_calibration_error(wrong, y) == pytest.approx(1.0)


def test_ece_stays_within_unit_interval_on_random_input():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(5), size=200)
    y = rng.integers(0, 5, size=200)
    assert 0.0 <= expected_calibration_error(probs, y) <= 1.0
    assert 0.0 <= maximum_calibration_error(probs, y) <= 1.0


def test_ece_rejects_an_empty_set():
    with pytest.raises(ValueError):
        expected_calibration_error(np.zeros((0, 3)), np.array([], dtype=int))


def test_reliability_bins_cover_every_sample_once():
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(4), size=50)
    y = rng.integers(0, 4, size=50)
    bins = reliability_bins(probs, y, n_bins=10)
    assert len(bins) == 10
    assert sum(b.count for b in bins) == 50


def test_brier_score_bounds():
    y = np.array([0, 1, 2])
    assert brier_score(_one_hot(y), y) == pytest.approx(0.0)
    assert brier_score(_one_hot((y + 1) % 3), y) == pytest.approx(2.0)


def test_selective_risk_coverage_drops_coverage_as_threshold_rises():
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.55, 0.45], [0.2, 0.8]])
    y = np.array([0, 0, 1, 1])
    rows = selective_risk_coverage(probs, y, thresholds=[0.0, 0.7, 0.99])
    assert rows[0]["coverage"] == pytest.approx(1.0)
    assert rows[1]["coverage"] == pytest.approx(0.5)
    assert rows[2]["coverage"] == pytest.approx(0.0)
    assert np.isnan(rows[2]["selective_accuracy"])


def test_summarise_and_markdown_table_agree_on_bin_count():
    rng = np.random.default_rng(2)
    probs = rng.dirichlet(np.ones(3), size=40)
    y = rng.integers(0, 3, size=40)
    summary = summarise(probs, y, n_bins=5)
    assert summary["n_bins"] == 5
    assert {"ece", "mce", "brier"} <= set(summary)
    table = to_markdown_table(reliability_bins(probs, y, n_bins=5))
    assert len(table.splitlines()) == 2 + 5
