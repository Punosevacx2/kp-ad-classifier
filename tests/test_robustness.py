import numpy as np
import pandas as pd

from kp_classifier.robustness import (
    add_typos,
    build_variants,
    class_word_counts,
    drop_word,
    is_typo_candidate,
    make_typo,
    shuffle_words,
    strongest_word_index,
)


def test_make_typo_always_changes_word():
    rng = np.random.default_rng(0)
    for _ in range(500):
        word = "brusilica"
        new = make_typo(word, rng)
        assert new != word
        assert abs(len(new) - len(word)) <= 1


def test_add_typos_skips_special_tokens_and_numbers():
    rng = np.random.default_rng(1)
    assert not is_typo_candidate("<phone>")
    assert not is_typo_candidate("125")
    assert not is_typo_candidate("pro")
    text, ok = add_typos("iphone 13 <phone>", 2, rng)
    assert ok
    assert "13" in text and "<phone>" in text and "iphone" not in text.split()
    assert add_typos("13 <phone> 506", 1, rng) == ("13 <phone> 506", False)


def test_shuffle_words_keeps_tokens():
    rng = np.random.default_rng(2)
    text, ok = shuffle_words("ugaona brusilica makita 125", rng)
    assert ok and sorted(text.split()) == sorted("ugaona brusilica makita 125".split())
    assert shuffle_words("krevet", rng) == ("krevet", False)


def test_strongest_word_is_class_specific():
    train = pd.DataFrame({
        "text": ["krevet bracni", "krevet samac", "na prodaju plac", "plac na prodaju", "na stanju iphone"],
        "label": [0, 0, 1, 1, 2],
    })
    counts = class_word_counts(train)
    idx = strongest_word_index("prodajem krevet na sprat", 0, counts, 3)
    assert "prodajem krevet na sprat".split()[idx] == "krevet"
    assert strongest_word_index("krevet", 0, counts, 3) is None
    assert drop_word("prodajem krevet na sprat", idx) == "prodajem na sprat"


def test_build_variants_alignment():
    train = pd.DataFrame({"text": ["krevet bracni", "brusilica makita", "iphone pro"] * 5, "label": [0, 1, 2] * 5})
    val = pd.DataFrame({
        "name": ["a", "b", "c", "d"],
        "text": ["krevet bracni novi", "brusilica nepoznatarec", "iphone", "13 506"],
        "label": [0, 1, 2, 2],
        "group_id": [1, 2, 3, 3],
    }, index=[10, 11, 12, 13])
    variants = build_variants(val, train, seed=0)
    assert list(variants["original"].index) == [0, 1, 2, 3]
    assert set(variants["typo_1"].index) == {0, 1, 2}
    assert set(variants["drop_keyword"].index) == {0, 1}
    assert set(variants["has_unknown_word"].index) | set(variants["no_unknown_word"].index) == {0, 1, 2, 3}
    for name, df in variants.items():
        assert (df["label"].values == variants["original"].loc[df.index, "label"].values).all()
