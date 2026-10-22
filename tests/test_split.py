import pandas as pd
import pytest

from kp_classifier.data.split import (
    add_labels,
    build_label_map,
    check_no_leakage,
    class_distribution,
    clean,
    max_proportion_gap,
    per_class_split,
)


@pytest.fixture
def raw_df():
    rows = []
    groups = [(489, "Apple iPhone", 137), (405, "Kreveti i dušeci", 93), (232, "Placevi i zemljište", 211)]
    for group_id, group_name, count in groups:
        for i in range(count):
            rows.append((f"{group_name} oglas {i}", group_id, group_name))
    rows += [
        ("Dušek 160x200", 405, "Kreveti i dušeci"),
        ("dusek 160x200", 405, "Kreveti i dušeci"),
        ("Душек 160х200", 405, "Kreveti i dušeci"),
        ("prodajem", 405, "Kreveti i dušeci"),
        ("Prodajem", 232, "Placevi i zemljište"),
        ("!!!", 489, "Apple iPhone"),
        ("#ERROR!", 489, "Apple iPhone"),
    ]
    return pd.DataFrame(rows, columns=["name", "group_id", "group_name"])


def prepare(raw_df):
    df, _ = clean(raw_df)
    label_map = build_label_map(df)
    return add_labels(df, label_map), label_map


def test_clean_removes_duplicates_conflicts_and_empty(raw_df):
    df, stats = clean(raw_df)
    assert stats["invalid_rows"] == 1
    assert stats["empty_after_normalization"] == 1
    assert stats["conflict_texts"] == 1
    assert stats["conflict_rows"] == 2
    assert stats["duplicate_rows"] == 2
    assert stats["clean_rows"] == 442
    assert df["text"].is_unique
    assert "prodajem" not in set(df["text"])
    assert "error" not in set(df["text"])


def test_label_map_sorted_by_group_id(raw_df):
    _, label_map = prepare(raw_df)
    assert [item["group_id"] for item in label_map] == [232, 405, 489]
    assert [item["index"] for item in label_map] == [0, 1, 2]


def test_per_class_split_counts(raw_df):
    df, _ = prepare(raw_df)
    splits, dropped = per_class_split(df, val_size=0.1, test_size=0.1, seed=42)
    check_no_leakage(splits)
    assert dropped == 0
    assert sum(len(s) for s in splits.values()) == len(df)
    for label, n in df["label"].value_counts().items():
        assert (splits["test"]["label"] == label).sum() == round(n * 0.1)
        assert (splits["val"]["label"] == label).sum() == round(n * 0.1)


def test_val_and_test_have_identical_class_counts(raw_df):
    df, label_map = prepare(raw_df)
    splits, _ = per_class_split(df, val_size=0.1, test_size=0.1, seed=42)
    distribution = class_distribution(splits, label_map)
    assert (distribution["val"] == distribution["test"]).all()
    for label, n in df["label"].value_counts().items():
        expected = n * 0.1
        assert abs((splits["val"]["label"] == label).sum() - expected) <= 0.5


def test_exact_mode_gives_identical_proportions(raw_df):
    df, label_map = prepare(raw_df)
    splits, dropped = per_class_split(df, val_size=0.1, test_size=0.1, seed=42, exact=True)
    check_no_leakage(splits)
    assert dropped > 0
    assert max_proportion_gap(class_distribution(splits, label_map)) < 1e-9


def test_split_is_reproducible(raw_df):
    df, _ = prepare(raw_df)
    a, _ = per_class_split(df, val_size=0.1, test_size=0.1, seed=42)
    b, _ = per_class_split(df, val_size=0.1, test_size=0.1, seed=42)
    for name in a:
        assert a[name]["text"].tolist() == b[name]["text"].tolist()


def test_split_does_not_depend_on_input_order(raw_df):
    shuffled = raw_df.sample(frac=1, random_state=0).reset_index(drop=True)
    results = []
    for raw in (raw_df, shuffled):
        df, _ = prepare(raw)
        splits, _ = per_class_split(df, val_size=0.1, test_size=0.1, seed=42)
        results.append({name: sorted(s["text"]) for name, s in splits.items()})
    assert results[0] == results[1]


def test_check_no_leakage_raises():
    a = pd.DataFrame({"text": ["x", "y"]})
    b = pd.DataFrame({"text": ["y", "z"]})
    with pytest.raises(ValueError):
        check_no_leakage({"train": a, "test": b})


def test_removing_row_from_one_class_does_not_change_other_classes(raw_df):
    df, _ = prepare(raw_df)
    full, _ = per_class_split(df, val_size=0.1, test_size=0.1, seed=42)

    iphone_label = df.loc[df["group_id"] == 489, "label"].iloc[0]
    dropped_index = df.index[df["label"] == iphone_label][0]
    reduced, _ = per_class_split(df.drop(index=dropped_index), val_size=0.1, test_size=0.1, seed=42)

    for name in full:
        a = full[name][full[name]["label"] != iphone_label]
        b = reduced[name][reduced[name]["label"] != iphone_label]
        assert sorted(a["text"]) == sorted(b["text"])
