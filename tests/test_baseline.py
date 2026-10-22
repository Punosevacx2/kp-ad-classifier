import numpy as np
import pandas as pd

from kp_classifier.models.baseline import (
    build_pipeline,
    feature_names,
    feature_weight,
    prior_predictions,
    select_C,
    top_features,
)

CONFIG = {
    "word_ngram_range": [1, 2],
    "char_ngram_range": [2, 4],
    "min_df": 1,
    "sublinear_tf": True,
    "max_iter": 1000,
    "C_grid": [0.1, 10.0],
    "top_features": 5,
}
NAMES = ["Kreveti", "Brusilice", "Telefoni"]


def make_df(n=20, seed=0):
    rng = np.random.default_rng(seed)
    vocab = [["krevet", "dusek", "sprat"], ["brusilica", "makita", "ugaona"], ["iphone", "pro", "<phone>"]]
    rows = []
    for label, words in enumerate(vocab):
        for _ in range(n):
            text = " ".join(rng.choice(words, size=3)) + f" {rng.integers(1000)}"
            rows.append({"name": text, "text": text, "label": label})
    return pd.DataFrame(rows)


def test_pipeline_learns_simple_task():
    train, val = make_df(seed=0), make_df(seed=1)
    pipeline = build_pipeline(CONFIG, C=10.0)
    pipeline.fit(train["text"], train["label"])
    assert (pipeline.predict(val["text"]) == val["label"]).mean() == 1.0
    assert pipeline.predict_proba(val["text"]).shape == (len(val), 3)


def test_special_tokens_are_single_features():
    pipeline = build_pipeline(CONFIG, C=1.0)
    pipeline.fit(make_df()["text"], make_df()["label"])
    names = set(feature_names(pipeline))
    assert "word:<phone>" in names
    assert "word:phone" not in names


def test_select_C_and_top_features():
    best_C, pipeline, search = select_C(make_df(seed=0), make_df(seed=1), CONFIG, verbose=False)
    assert best_C in CONFIG["C_grid"]
    assert len(search) == len(CONFIG["C_grid"])
    features = top_features(pipeline, NAMES, k=5)
    assert len(features) == 15
    weights = feature_weight(pipeline, NAMES, "word:<phone>")
    assert weights.index[0] == "Telefoni"


def test_prior_predictions_predict_majority():
    train = pd.DataFrame({"label": [0, 0, 0, 1, 2]})
    df = make_df(n=2)
    preds = prior_predictions(train, df, 3)
    assert (preds["pred"] == 0).all()
    assert np.allclose(preds[["proba_0", "proba_1", "proba_2"]].iloc[0], [0.6, 0.2, 0.2])
    