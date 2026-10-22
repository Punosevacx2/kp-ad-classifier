import numpy as np
import pandas as pd
import pytest

from kp_classifier.data.dataset import encode_split
from kp_classifier.data.tokenizer import Tokenizer
from kp_classifier.ensemble import apply_temperature, ensemble_proba, fit_temperature, member_proba, predict_any
from kp_classifier.models.baseline import build_pipeline
from kp_classifier.train import save_model, train_model


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def test_apply_temperature_identity_and_sharpening():
    p = np.array([[0.7, 0.2, 0.1], [0.3, 0.3, 0.4]])
    assert np.allclose(apply_temperature(p, 1.0), p)
    sharp = apply_temperature(p, 0.5)
    assert sharp[0, 0] > p[0, 0]
    assert np.allclose(sharp.sum(axis=1), 1.0)
    assert (sharp.argmax(axis=1) == p.argmax(axis=1)).all()


@pytest.mark.parametrize("true_t", [0.4, 3.0])
def test_fit_temperature_recovers_scale(true_t):
    rng = np.random.default_rng(0)
    z = rng.normal(size=(20000, 5)) * 2
    labels = np.array([rng.choice(5, p=row) for row in softmax(z)])
    miscalibrated = softmax(z / true_t)
    assert fit_temperature(miscalibrated, labels) == pytest.approx(1 / true_t, rel=0.1)


def test_ensemble_mixes_sklearn_and_torch(tmp_path):
    import joblib
    vocab = [["krevet", "dusek"], ["brusilica", "makita"], ["iphone", "pro"]]
    rows = [{"name": f"{w} {i}", "text": f"{w} {i}", "label": l}
            for l, ws in enumerate(vocab) for w in ws for i in range(15)]
    df = pd.DataFrame(rows)

    pipeline = build_pipeline({"word_ngram_range": [1, 1], "char_ngram_range": [2, 3], "min_df": 1,
                               "sublinear_tf": True, "max_iter": 500}, C=10.0)
    pipeline.fit(df["text"], df["label"])
    (tmp_path / "lr").mkdir()
    joblib.dump(pipeline, tmp_path / "lr" / "model.joblib")

    config = {
        "name": "t", "seed": 0,
        "tokenizer": {"min_word_freq": 1, "min_char_freq": 1, "max_words": 4, "max_chars": 20},
        "model": {"word_embedding_dim": 8, "word_kernel_sizes": [1], "word_filters": 8,
                  "char_embedding_dim": 4, "char_kernel_sizes": [3], "char_filters": 8, "hidden_dim": 8},
        "training": {"batch_size": 16, "max_epochs": 5, "learning_rate": 0.01, "weight_decay": 0.0,
                     "grad_clip": 1.0, "early_stopping_patience": 5, "lr_patience": 5, "lr_factor": 0.5,
                     "device": "cpu"},
    }
    tok = Tokenizer.build(df["text"], min_word_freq=1, min_char_freq=1, max_words=4, max_chars=20)
    model, *_ = train_model(config, encode_split(tok, df), encode_split(tok, df), 3,
                            tok.word_vocab_size, tok.char_vocab_size, verbose=False)
    label_map = tmp_path / "label_map.json"
    label_map.write_text('[{"index":0,"group_id":1,"group_name":"a"},{"index":1,"group_id":2,"group_name":"b"},'
                         '{"index":2,"group_id":3,"group_name":"c"}]', encoding="utf-8")
    save_model(tmp_path / "nn", model.cpu(), tok, label_map, config, {})

    p_lr = member_proba(tmp_path / "lr", df)
    p_nn = member_proba(tmp_path / "nn", df)
    avg, members = ensemble_proba([tmp_path / "lr", tmp_path / "nn"], df)
    assert avg.shape == (len(df), 3)
    assert np.allclose(avg, (p_lr + p_nn) / 2)
    assert len(members) == 2
    import yaml
    (tmp_path / "ens").mkdir()
    (tmp_path / "ens" / "ensemble.yaml").write_text(
        yaml.safe_dump({"members": ["lr", "nn"], "temperature": 0.5}), encoding="utf-8")
    p_ens = predict_any(tmp_path / "ens", df)
    assert np.allclose(p_ens, apply_temperature(avg, 0.5))
    assert np.allclose(predict_any(tmp_path / "lr", df), p_lr)
