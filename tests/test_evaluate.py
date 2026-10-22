import json

import numpy as np
import pandas as pd
import pytest

from kp_classifier.evaluate import (
    bootstrap_ci,
    build_predictions_frame,
    compare_predictions,
    compare_runs,
    compute_metrics,
    coverage_summary,
    error_table,
    evaluate_predictions,
    expected_calibration_error,
    mcnemar_test,
    risk_coverage_curve,
    top_confusions,
    top_k_accuracy,
)

NAMES = ["A", "B", "C"]
LABEL_MAP = [{"index": i, "group_id": 100 + i, "group_name": n} for i, n in enumerate(NAMES)]


def one_hot(labels, n=3, p=1.0):
    proba = np.full((len(labels), n), (1 - p) / (n - 1))
    proba[np.arange(len(labels)), labels] = p
    return proba


@pytest.fixture
def df():
    return pd.DataFrame({
        "name": [f"Naslov {i}" for i in range(6)],
        "text": [f"naslov {i}" for i in range(6)],
        "label": [0, 0, 1, 1, 2, 2],
    })


def test_perfect_predictions():
    y = np.array([0, 1, 2, 0, 1, 2])
    m = compute_metrics(y, y, NAMES, one_hot(y))
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["errors"] == 0
    assert m["top2_accuracy"] == 1.0
    assert m["ece"] == pytest.approx(0.0)


def test_known_errors():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    m = compute_metrics(y_true, y_pred, NAMES)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["errors"] == 2
    assert m["per_class"]["B"]["recall"] == 1.0
    assert m["per_class"]["B"]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"]["A"]["support"] == 2
    assert "log_loss" not in m


def test_top_k_accuracy():
    y = np.array([0, 1])
    proba = np.array([[0.1, 0.6, 0.3], [0.5, 0.4, 0.1]])
    assert top_k_accuracy(y, proba, 1) == 0.0
    assert top_k_accuracy(y, proba, 2) == 0.5


def test_ece_overconfident_model():
    y = np.array([0, 1, 2, 0])
    proba = one_hot(np.array([0, 0, 0, 0]), p=0.99)
    assert expected_calibration_error(y, proba) == pytest.approx(0.99 - 0.5, abs=1e-6)


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 1000)
    y_pred = np.where(rng.random(1000) < 0.9, y_true, (y_true + 1) % 3)
    ci = bootstrap_ci(y_true, y_pred, 3, n_boot=300)
    acc = (y_true == y_pred).mean()
    assert ci["accuracy"][0] < acc < ci["accuracy"][1]
    assert 0.02 < ci["accuracy"][1] - ci["accuracy"][0] < 0.06
    perfect = bootstrap_ci(y_true, y_true, 3, n_boot=50)
    assert perfect["accuracy"] == [1.0, 1.0]


def test_mcnemar():
    y = np.zeros(100, dtype=int)
    a = np.zeros(100, dtype=int)
    b = a.copy()
    b[:15] = 1
    result = mcnemar_test(y, a, b)
    assert result["only_a_correct"] == 15
    assert result["only_b_correct"] == 0
    assert result["p_value"] < 0.001

    c = a.copy()
    c[:5] = 1
    d = a.copy()
    d[5:10] = 1
    assert mcnemar_test(y, c, d)["p_value"] == pytest.approx(1.0)
    assert mcnemar_test(y, a, a)["p_value"] == 1.0


def test_compare_predictions_aligns_by_text(df):
    a = build_predictions_frame(df, one_hot(df["label"].to_numpy(), p=0.9))
    b = build_predictions_frame(df, one_hot(np.array([1, 0, 1, 1, 2, 2]), p=0.9))
    result = compare_predictions(a, b.sample(frac=1, random_state=0))
    assert result["only_a_correct"] == 1
    assert result["only_b_correct"] == 0
    with pytest.raises(ValueError):
        compare_predictions(a, b.iloc[:3])


def test_risk_coverage():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    proba = one_hot(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1]), p=0.9)
    proba[8:] = [[0.4, 0.5, 0.1]] * 2
    curve = risk_coverage_curve(y, proba)
    assert curve["accuracy"].iloc[7] == 1.0
    assert curve["accuracy"].iloc[-1] == pytest.approx(0.8)
    summary = coverage_summary(curve, coverages=(1.0, 0.8), targets=(0.99,))
    assert summary["at_coverage"][0]["accuracy"] == pytest.approx(0.8)
    assert summary["at_coverage"][1]["accuracy"] == 1.0
    assert summary["for_target_accuracy"][0]["coverage"] == pytest.approx(0.8)


def test_top_confusions_and_error_table(df):
    proba = one_hot(np.array([0, 1, 1, 1, 2, 0]), p=0.9)
    proba[1] = [0.05, 0.95, 0.0]
    preds = build_predictions_frame(df, proba)

    conf = top_confusions(preds["label"], preds["pred"], NAMES)
    assert set(zip(conf["stvarna"], conf["predvidjena"])) == {("A", "B"), ("C", "A")}

    errors = error_table(preds, NAMES)
    assert len(errors) == 2
    assert errors.loc[0, "name"] == "Naslov 1"
    assert errors["confidence"].is_monotonic_decreasing


def test_evaluate_predictions_writes_files(df, tmp_path):
    preds = build_predictions_frame(df, one_hot(df["label"].to_numpy(), p=0.8))
    run_dir = tmp_path / "runs" / "model_a"
    m = evaluate_predictions(preds, LABEL_MAP, run_dir, "model_a", "val", verbose=False, n_boot=50)
    assert m["accuracy"] == 1.0
    assert "ci" in m and "risk_coverage" in m
    for name in ["val_metrics.json", "val_per_class.csv", "val_top_confusions.csv",
                 "val_errors.csv", "val_predictions.parquet", "val_confusion_matrix.png",
                 "val_reliability.png", "val_risk_coverage.png"]:
        assert (run_dir / name).exists()
    saved = json.loads((run_dir / "val_metrics.json").read_text(encoding="utf-8"))
    assert saved["model"] == "model_a"

    worse = build_predictions_frame(df, one_hot(np.array([0, 1, 1, 1, 2, 2]), p=0.8))
    evaluate_predictions(worse, LABEL_MAP, tmp_path / "runs" / "model_b", "model_b", "val", verbose=False, n_boot=50)
    table = compare_runs(tmp_path / "runs", "val")
    assert table["run"].tolist() == ["model_a", "model_b"]
    assert "acc_ci_low" in table.columns
