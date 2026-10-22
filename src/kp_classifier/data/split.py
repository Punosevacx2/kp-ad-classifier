import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from kp_classifier.data.preprocess import normalize_text

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "ml_dataset.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_NAMES = ["train", "val", "test"]


def load_raw(path):
    df = pd.read_parquet(path)
    return df[["name", "group_id", "group_name"]].copy()


def clean(df):
    stats = {"raw_rows": len(df)}

    df = df.copy()
    df["text"] = df["name"].map(normalize_text)
    df = df.sort_values(["group_id", "text", "name"], kind="stable").reset_index(drop=True)

    empty = df["text"] == ""
    stats["empty_after_normalization"] = int(empty.sum())
    df = df[~empty]

    groups_per_text = df.groupby("text")["group_id"].nunique()
    conflict_texts = set(groups_per_text[groups_per_text > 1].index)
    conflict_mask = df["text"].isin(conflict_texts)
    stats["conflict_texts"] = len(conflict_texts)
    stats["conflict_rows"] = int(conflict_mask.sum())
    df = df[~conflict_mask]

    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    stats["duplicate_rows"] = before - len(df)
    stats["clean_rows"] = len(df)

    return df.reset_index(drop=True), stats


def build_label_map(df):
    groups = (
        df[["group_id", "group_name"]]
        .drop_duplicates()
        .sort_values("group_id")
        .reset_index(drop=True)
    )
    return [
        {"index": i, "group_id": int(row.group_id), "group_name": row.group_name}
        for i, row in groups.iterrows()
    ]


def add_labels(df, label_map):
    group_to_index = {item["group_id"]: item["index"] for item in label_map}
    df = df.copy()
    df["label"] = df["group_id"].map(group_to_index).astype(int)
    return df


def exact_block_size(val_size, test_size):
    denominators = [Fraction(val_size).limit_denominator(1000).denominator,
                    Fraction(test_size).limit_denominator(1000).denominator]
    return math.lcm(*denominators)


def per_class_split(df, val_size, test_size, seed, exact=False):
    rng = np.random.default_rng(seed)
    block = exact_block_size(val_size, test_size) if exact else 1
    parts = {name: [] for name in SPLIT_NAMES}
    dropped = 0

    for label in sorted(df["label"].unique()):
        class_df = df[df["label"] == label]
        order = rng.permutation(len(class_df))
        class_df = class_df.iloc[order]

        n = len(class_df) - len(class_df) % block
        dropped += len(class_df) - n
        class_df = class_df.iloc[:n]

        n_test = round(n * test_size)
        n_val = round(n * val_size)
        parts["test"].append(class_df.iloc[:n_test])
        parts["val"].append(class_df.iloc[n_test:n_test + n_val])
        parts["train"].append(class_df.iloc[n_test + n_val:])

    splits = {
        name: pd.concat(chunks).sample(frac=1, random_state=seed).reset_index(drop=True)
        for name, chunks in parts.items()
    }
    return splits, dropped


def check_no_leakage(splits):
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(splits[a]["text"]) & set(splits[b]["text"])
            if overlap:
                raise ValueError(f"Isti tekst u skupovima {a} i {b}: {len(overlap)} primera")


def class_distribution(splits, label_map):
    index_to_name = {item["index"]: item["group_name"] for item in label_map}
    table = pd.DataFrame({
        name: split["label"].value_counts().sort_index() for name, split in splits.items()
    }).fillna(0).astype(int)
    table.index = [index_to_name[i] for i in table.index]
    table["ukupno"] = table.sum(axis=1)
    return table


def class_proportions(distribution):
    counts = distribution[SPLIT_NAMES]
    return counts.div(counts.sum(), axis=1) * 100


def max_proportion_gap(distribution):
    proportions = class_proportions(distribution)
    return float((proportions.max(axis=1) - proportions.min(axis=1)).max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exact", action="store_true")
    args = parser.parse_args()

    raw = load_raw(args.input)
    df, stats = clean(raw)
    label_map = build_label_map(df)
    df = add_labels(df, label_map)

    splits, dropped = per_class_split(df, args.val_size, args.test_size, args.seed, args.exact)
    check_no_leakage(splits)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["name", "text", "group_id", "label"]
    for name, split in splits.items():
        split[columns].to_parquet(args.output_dir / f"{name}.parquet", index=False)

    with open(args.output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    distribution = class_distribution(splits, label_map)
    proportions = class_proportions(distribution)
    gap = max_proportion_gap(distribution)
    total = sum(len(s) for s in splits.values())

    stats.update({
        "seed": args.seed,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "exact": args.exact,
        "dropped_for_exact_proportions": dropped,
        "split_rows": {name: len(split) for name, split in splits.items()},
        "max_proportion_gap_pp": round(gap, 4),
        "class_distribution": distribution.to_dict(orient="index"),
        "class_proportions_pct": proportions.round(4).to_dict(orient="index"),
    })
    with open(args.output_dir / "split_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Čišćenje:")
    for key in ["raw_rows", "empty_after_normalization", "conflict_texts", "conflict_rows", "duplicate_rows", "clean_rows"]:
        print(f"  {key}: {stats[key]}")
    if args.exact:
        print(f"  dropped_for_exact_proportions: {dropped}")
    print("\nVeličine skupova:")
    for name, split in splits.items():
        print(f"  {name}: {len(split)} ({len(split) / total * 100:.1f}%)")
    print("\nRaspodela klasa:")
    print(distribution.to_string())
    print("\nUdeo klasa po skupovima (%):")
    print(proportions.round(3).to_string())
    print(f"\nNajveće odstupanje udela između skupova: {gap:.4f} procentnih poena")
    print(f"\nSačuvano u: {args.output_dir}")


if __name__ == "__main__":
    main()