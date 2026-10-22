import numpy as np
import pytest

from kp_classifier.data.tokenizer import (
    PAD_ID,
    UNK_ID,
    Tokenizer,
    char_sequence,
    coverage_stats,
    tokenize_words,
)

TRAIN = [
    "ugaona brusilica makita 125",
    "ugaona brusilica bosch 125",
    "krevet na sprat 90x200",
    "krevet bracni 160x200",
    "iphone 13 pro <phone>",
    "iphone 13 mini",
    "plac 1234 m2 <id>",
]


@pytest.fixture
def tok():
    return Tokenizer.build(TRAIN, min_word_freq=2, min_char_freq=1, max_words=6, max_chars=20)


def test_tokenize_keeps_special_tokens():
    assert tokenize_words("iphone 13 <phone> x") == ["iphone", "13", "<phone>", "x"]
    assert char_sequence("a <phone> b") == "a \u0001 b"


def test_reserved_ids(tok):
    assert tok.word_itos[PAD_ID] == "<pad>"
    assert tok.word_itos[UNK_ID] == "<unk>"
    assert tok.char_itos[PAD_ID] == "<pad>"
    assert tok.char_itos[UNK_ID] == "<unk>"
    for t in ["<num1>", "<num2>", "<num3>", "<num4+>", "<phone>", "<id>"]:
        assert t in tok.word_stoi


def test_min_freq_and_numbers(tok):
    assert "brusilica" in tok.word_stoi
    assert "makita" not in tok.word_stoi
    assert "125" in tok.word_stoi
    assert tok.word_id("125") == tok.word_stoi["125"]
    assert tok.word_id("7") == tok.word_stoi["<num1>"]
    assert tok.word_id("506") == tok.word_stoi["<num3>"]
    assert tok.word_id("123456") == tok.word_stoi["<num4+>"]
    assert tok.word_id("nepoznato") == UNK_ID
    assert tok.word_id("128gb") == UNK_ID


def test_encode_pads_and_truncates(tok):
    words, chars = tok.encode(["krevet", "krevet krevet krevet krevet krevet krevet krevet krevet"])
    assert words.shape == (2, 6) and chars.shape == (2, 20)
    assert words.dtype == np.int64
    assert words[0, 0] == tok.word_stoi["krevet"]
    assert (words[0, 1:] == PAD_ID).all()
    assert (words[1] != PAD_ID).all()
    assert (chars[1] != PAD_ID).all()


def test_special_token_is_single_char(tok):
    _, chars = tok.encode(["iphone <phone>"])
    decoded = tok.decode_chars(chars[0])
    assert decoded == "iphone <phone>"
    assert (chars[0] != PAD_ID).sum() == len("iphone ") + 1


def test_unknown_char(tok):
    _, chars = tok.encode(["qqq"])
    assert chars[0, 0] == UNK_ID


def test_save_load_roundtrip(tok, tmp_path):
    path = tmp_path / "vocab.json"
    tok.save(path)
    loaded = Tokenizer.load(path)
    texts = ["ugaona brusilica 999 nepoznato", "iphone 13 <phone>"]
    for a, b in zip(tok.encode(texts), loaded.encode(texts)):
        assert (a == b).all()
    assert loaded.max_words == 6 and loaded.max_chars == 20


def test_coverage_stats(tok):
    stats = coverage_stats(tok, ["krevet nepoznato", "krevet 7"])
    assert stats["tokens"] == 4
    assert stats["unk_token_rate"] == pytest.approx(0.25)
    assert stats["num_shape_token_rate"] == pytest.approx(0.25)
    assert stats["titles_with_unk_rate"] == pytest.approx(0.5)


def test_build_is_deterministic():
    a = Tokenizer.build(TRAIN, min_char_freq=1)
    b = Tokenizer.build(list(reversed(TRAIN)), min_char_freq=1)
    assert a.word_itos == b.word_itos
    assert a.char_itos == b.char_itos
