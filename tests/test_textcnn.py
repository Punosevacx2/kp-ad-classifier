import pytest
import torch

from kp_classifier.data.tokenizer import PAD_ID, UNK_ID
from kp_classifier.models.textcnn import HybridTextCNN, build_model, parameter_table

WORD_VOCAB, CHAR_VOCAB, CLASSES = 50, 20, 4


def batch(batch_size=3, words=8, chars=15, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(2, WORD_VOCAB, (batch_size, words), generator=g)
    c = torch.randint(2, CHAR_VOCAB, (batch_size, chars), generator=g)
    w[:, 5:] = PAD_ID
    c[:, 10:] = PAD_ID
    return w, c


def small_model(**kwargs):
    defaults = dict(word_embedding_dim=8, word_filters=6, char_embedding_dim=4, char_filters=5, hidden_dim=7)
    defaults.update(kwargs)
    return HybridTextCNN(WORD_VOCAB, CHAR_VOCAB, CLASSES, **defaults)


def test_output_shape():
    w, c = batch()
    assert small_model()(w, c).shape == (3, CLASSES)


@pytest.mark.parametrize("use_words,use_chars,dim", [(True, False, 18), (False, True, 15), (True, True, 33)])
def test_branch_variants(use_words, use_chars, dim):
    model = small_model(use_words=use_words, use_chars=use_chars)
    assert model.feature_dim == dim
    w, c = batch()
    assert model(w, c).shape == (3, CLASSES)


def test_no_branch_raises():
    with pytest.raises(ValueError):
        small_model(use_words=False, use_chars=False)


def test_padding_does_not_change_output():
    torch.manual_seed(0)
    model = small_model().eval()
    w, c = batch()
    longer_w = torch.cat([w, torch.full((3, 12), PAD_ID)], dim=1)
    longer_c = torch.cat([c, torch.full((3, 25), PAD_ID)], dim=1)
    assert torch.allclose(model(w, c), model(longer_w, longer_c), atol=1e-6)


def test_short_and_empty_inputs_are_finite():
    model = small_model().eval()
    one_word = torch.tensor([[7, 0, 0]])
    empty = torch.zeros((1, 3), dtype=torch.long)
    chars = torch.tensor([[3, 4]])
    for w in (one_word, empty):
        assert torch.isfinite(model(w, chars)).all()


def test_word_dropout_only_in_training():
    model = small_model(word_dropout=1.0)
    w, _ = batch()
    model.train()
    dropped = model.apply_word_dropout(w)
    assert ((dropped == UNK_ID) == (w != PAD_ID)).all()
    model.eval()
    assert torch.equal(model.apply_word_dropout(w), w)


def test_eval_is_deterministic():
    model = small_model().eval()
    w, c = batch()
    assert torch.equal(model(w, c), model(w, c))


def test_build_model_and_parameter_count():
    config = {"word_embedding_dim": 8, "word_kernel_sizes": [1, 2], "word_filters": 4,
              "char_embedding_dim": 4, "char_kernel_sizes": [3], "char_filters": 5, "hidden_dim": 0}
    model = build_model(config, WORD_VOCAB, CHAR_VOCAB, CLASSES)
    expected = (
        WORD_VOCAB * 8 + (8 * 4 * 1 + 4) + (8 * 4 * 2 + 4)
        + CHAR_VOCAB * 4 + (4 * 5 * 3 + 5)
        + (8 + 5) * CLASSES + CLASSES
    )
    table = dict(parameter_table(model))
    assert table["UKUPNO"] == expected
    assert sum(n for k, n in table.items() if k != "UKUPNO") == expected


def test_max_mean_pooling():
    torch.manual_seed(0)
    model = small_model(pooling="max_mean").eval()
    assert model.feature_dim == 2 * (3 * 6 + 3 * 5)
    w, c = batch()
    longer_w = torch.cat([w, torch.full((3, 12), PAD_ID)], dim=1)
    longer_c = torch.cat([c, torch.full((3, 25), PAD_ID)], dim=1)
    assert torch.allclose(model(w, c), model(longer_w, longer_c), atol=1e-6)
    with pytest.raises(ValueError):
        small_model(pooling="avg")