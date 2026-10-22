import json

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from kp_classifier.ensemble import apply_temperature
from kp_classifier.models.baseline import build_pipeline
from kp_classifier.predict import Predictor, format_result, read_titles

LABEL_MAP = [
    {"index": 0, "group_id": 405, "group_name": "Kreveti i dušeci"},
    {"index": 1, "group_id": 973, "group_name": "Električne brusilice"},
    {"index": 2, "group_id": 489, "group_name": "Apple iPhone"},
]


@pytest.fixture
def models_dir(tmp_path):
    vocab = [["krevet", "dusek", "bracni"], ["brusilica", "makita", "ugaona"], ["iphone", "pro", "max"]]
    rows = [{"text": f"{w} {v}", "label": l} for l, ws in enumerate(vocab) for w in ws for v in ws]
    df = pd.DataFrame(rows)
    for name, C in [("lr_a", 10.0), ("lr_b", 1.0)]:
        pipeline = build_pipeline({"word_ngram_range": [1, 1], "char_ngram_range": [2, 3], "min_df": 1,
                                   "sublinear_tf": True, "max_iter": 500}, C=C)
        pipeline.fit(df["text"], df["label"])
        (tmp_path / name).mkdir()
        joblib.dump(pipeline, tmp_path / name / "model.joblib")
        (tmp_path / name / "label_map.json").write_text(json.dumps(LABEL_MAP), encoding="utf-8")
    (tmp_path / "ens").mkdir()
    (tmp_path / "ens" / "label_map.json").write_text(json.dumps(LABEL_MAP), encoding="utf-8")
    (tmp_path / "ens" / "ensemble.yaml").write_text(
        yaml.safe_dump({"members": ["lr_a", "lr_b"], "temperature": 0.7}), encoding="utf-8")
    return tmp_path


def test_single_model_prediction(models_dir):
    predictor = Predictor(models_dir / "lr_a")
    results = predictor.predict(["Ugaona BRUSILICA Makita", "iPhone 15 Pro, zvati 064/123-45-67"], top_k=2)
    assert results[0]["group_name"] == "Električne brusilice"
    assert results[0]["group_id"] == 973
    assert results[1]["group_name"] == "Apple iPhone"
    assert "<phone>" in results[1]["normalized"]
    assert len(results[0]["top_k"]) == 2
    assert results[0]["top_k"][0]["probability"] >= results[0]["top_k"][1]["probability"]


def test_ensemble_matches_manual_average(models_dir):
    titles = ["bračni krevet", "makita"]
    a = Predictor(models_dir / "lr_a").predict_proba(titles)[1]
    b = Predictor(models_dir / "lr_b").predict_proba(titles)[1]
    texts, ens = Predictor(models_dir / "ens").predict_proba(titles)
    assert texts == ["bracni krevet", "makita"]
    assert np.allclose(ens, apply_temperature((a + b) / 2, 0.7))
    assert np.allclose(ens.sum(axis=1), 1.0)


def test_needs_review(models_dir):
    predictor = Predictor(models_dir / "lr_a")
    assert predictor.predict(["!!!"], threshold=0.0)[0]["needs_review"]
    assert predictor.predict(["iphone pro max"], threshold=1.01)[0]["needs_review"]
    assert not predictor.predict(["iphone pro max"], threshold=0.0)[0]["needs_review"]


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Predictor(tmp_path / "nema")


def test_read_titles_and_format(tmp_path, models_dir):
    txt = tmp_path / "in.txt"
    txt.write_text("krevet\n\niphone 13\n", encoding="utf-8")
    assert read_titles(txt) == ["krevet", "iphone 13"]
    csv = tmp_path / "in.csv"
    pd.DataFrame({"name": ["makita", "dusek"]}).to_csv(csv, index=False)
    assert read_titles(csv) == ["makita", "dusek"]
    out = format_result(Predictor(models_dir / "lr_a").predict(["makita"])[0])
    assert "Električne brusilice (973)" in out
