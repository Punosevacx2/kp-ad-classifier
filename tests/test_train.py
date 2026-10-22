import numpy as np
import pandas as pd
import torch

from kp_classifier.data.dataset import encode_split, make_loader
from kp_classifier.data.tokenizer import Tokenizer
from kp_classifier.train import apply_overrides, load_trained, predict_proba, save_model, train_model

CONFIG = {
    "name": "test",
    "seed": 0,
    "tokenizer": {"min_word_freq": 1, "min_char_freq": 1, "max_words": 6, "max_chars": 30},
    "model": {
        "word_embedding_dim": 16, "word_kernel_sizes": [1, 2], "word_filters": 16,
        "char_embedding_dim": 8, "char_kernel_sizes": [3], "char_filters": 16,
        "hidden_dim": 16, "dropout": 0.1, "hidden_dropout": 0.1, "word_dropout": 0.1,
    },
    "training": {
        "batch_size": 16, "max_epochs": 15, "learning_rate": 0.01, "weight_decay": 0.0,
        "grad_clip": 1.0, "early_stopping_patience": 15, "lr_patience": 5, "lr_factor": 0.5,
        "device": "cpu",
    },
}
VOCAB = [["krevet", "dusek", "sprat"], ["brusilica", "makita", "ugaona"], ["iphone", "pro", "max"]]


def make_df(n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for label, words in enumerate(VOCAB):
        for _ in range(n):
            text = " ".join(rng.choice(words, size=3))
            rows.append({"name": text, "text": text, "label": label})
    return pd.DataFrame(rows)


def test_training_learns_and_model_roundtrips(tmp_path):
    train, val = make_df(40, 0), make_df(10, 1)
    tok = Tokenizer.build(train["text"], min_word_freq=1, min_char_freq=1, max_words=6, max_chars=30)
    train_ds, val_ds = encode_split(tok, train), encode_split(tok, val)

    model, history, best_epoch, device = train_model(
        CONFIG, train_ds, val_ds, 3, tok.word_vocab_size, tok.char_vocab_size, verbose=False
    )
    assert history["val_accuracy"].max() == 1.0
    assert 1 <= best_epoch <= len(history)
    assert history["train_loss"].iloc[-1] < history["train_loss"].iloc[0]

    loader = make_loader(val_ds, 64, shuffle=False)
    proba = predict_proba(model, loader, device)
    assert proba.shape == (len(val), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    label_map = tmp_path / "label_map.json"
    label_map.write_text('[{"index":0,"group_id":1,"group_name":"a"},{"index":1,"group_id":2,"group_name":"b"},'
                         '{"index":2,"group_id":3,"group_name":"c"}]', encoding="utf-8")
    save_model(tmp_path / "m", model.cpu(), tok, label_map, CONFIG, {"best_epoch": best_epoch})
    loaded, loaded_tok, loaded_map, loaded_cfg = load_trained(tmp_path / "m")
    proba_loaded = predict_proba(loaded, make_loader(encode_split(loaded_tok, val), 64, shuffle=False), "cpu")
    assert np.allclose(proba, proba_loaded, atol=1e-5)
    assert loaded_cfg["best_epoch"] == best_epoch
    assert len(loaded_map) == 3


def test_training_is_reproducible():
    train, val = make_df(20, 0), make_df(5, 1)
    tok = Tokenizer.build(train["text"], min_word_freq=1, min_char_freq=1, max_words=6, max_chars=30)
    cfg = {**CONFIG, "training": {**CONFIG["training"], "max_epochs": 3}}
    runs = [
        train_model(cfg, encode_split(tok, train), encode_split(tok, val), 3,
                    tok.word_vocab_size, tok.char_vocab_size, verbose=False)[1]
        for _ in range(2)
    ]
    assert np.allclose(runs[0]["train_loss"], runs[1]["train_loss"])


def test_apply_overrides():
    import copy
    import pytest
    cfg = copy.deepcopy(CONFIG)
    apply_overrides(cfg, ["seed=7", "model.dropout=0.2", "training.learning_rate=0.001",
                          "model.word_kernel_sizes=[1, 2, 3]"])
    assert cfg["seed"] == 7
    assert cfg["model"]["dropout"] == 0.2
    assert cfg["training"]["learning_rate"] == 0.001
    assert cfg["model"]["word_kernel_sizes"] == [1, 2, 3]
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["model.nepostojeci=1"])
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["seed"])
