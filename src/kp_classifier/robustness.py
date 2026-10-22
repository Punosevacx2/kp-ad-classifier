import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kp_classifier.data.tokenizer import UNK_ID, Tokenizer, tokenize_words
from kp_classifier.ensemble import predict_any
from kp_classifier.train import resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "reports" / "robustness"
LETTERS = "abcdefghijklmnopqrstuvwxyz"
TYPO_OPS = ("substitute", "delete", "insert", "transpose")


def is_typo_candidate(token):
    return token.isalpha() and len(token) >= 4


def make_typo(word, rng, max_tries=10):
    for _ in range(max_tries):
        op = TYPO_OPS[int(rng.integers(len(TYPO_OPS)))]
        i = int(rng.integers(len(word)))
        if op == "substitute":
            choices = [c for c in LETTERS if c != word[i]]
            new = word[:i] + choices[int(rng.integers(len(choices)))] + word[i + 1:]
        elif op == "delete":
            new = word[:i] + word[i + 1:]
        elif op == "insert":
            new = word[:i] + word[i] + word[i:]
        else:
            i = min(i, len(word) - 2)
            new = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        if new != word:
            return new
    return word[:-1]


def add_typos(text, n_words, rng):
    tokens = tokenize_words(text)
    candidates = [i for i, t in enumerate(tokens) if is_typo_candidate(t)]
    if not candidates:
        return text, False
    chosen = rng.choice(candidates, size=min(n_words, len(candidates)), replace=False)
    for i in chosen:
        tokens[i] = make_typo(tokens[i], rng)
    return " ".join(tokens), True


def shuffle_words(text, rng):
    tokens = tokenize_words(text)
    if len(tokens) < 2:
        return text, False
    return " ".join(tokens[i] for i in rng.permutation(len(tokens))), True


def class_word_counts(train):
    counts = defaultdict(Counter)
    for text, label in zip(train["text"], train["label"]):
        for word in set(tokenize_words(text)):
            counts[word][int(label)] += 1
    return counts


def strongest_word_index(text, label, counts, n_classes, alpha=1.0):
    words = tokenize_words(text)
    if len(words) < 2:
        return None
    best_idx, best_score = None, -1.0
    for i, word in enumerate(words):
        c = counts.get(word)
        if not c:
            continue
        score = (c[label] + alpha) / (sum(c.values()) + alpha * n_classes)
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx


def drop_word(text, index):
    return " ".join(w for j, w in enumerate(tokenize_words(text)) if j != index)


def build_variants(val, train, seed=0):
    val = val.reset_index(drop=True)
    n_classes = int(max(train["label"].max(), val["label"].max())) + 1
    variants = {"original": val}

    for n_words in (1, 2):
        rng = np.random.default_rng([seed, n_words])
        texts, keep = zip(*(add_typos(t, n_words, rng) for t in val["text"]))
        df = val.assign(text=list(texts))
        variants[f"typo_{n_words}"] = df[list(keep)]

    rng = np.random.default_rng([seed, 99])
    texts, keep = zip(*(shuffle_words(t, rng) for t in val["text"]))
    variants["shuffled_words"] = val.assign(text=list(texts))[list(keep)]

    counts = class_word_counts(train)
    idx = [strongest_word_index(t, int(l), counts, n_classes) for t, l in zip(val["text"], val["label"])]
    keep = [i is not None for i in idx]
    texts = [drop_word(t, i) if i is not None else t for t, i in zip(val["text"], idx)]
    variants["drop_keyword"] = val.assign(text=texts)[keep]

    tokenizer = Tokenizer.build(train["text"])
    has_unk = np.array([UNK_ID in tokenizer.encode_words(t) for t in val["text"]])
    variants["has_unknown_word"] = val[has_unk]
    variants["no_unknown_word"] = val[~has_unk]
    return variants


def evaluate_robustness(models, variants, device="cpu", models_dir=MODELS_DIR):
    rows = []
    for model in models:
        original = variants["original"]
        clean_correct = predict_any(models_dir / model, original, device).argmax(axis=1) == original["label"].to_numpy()
        for name, df in variants.items():
            if len(df) == 0:
                continue
            if name == "original":
                correct = clean_correct
            else:
                correct = predict_any(models_dir / model, df, device).argmax(axis=1) == df["label"].to_numpy()
            clean_same_rows = clean_correct[df.index.to_numpy()]
            rows.append({
                "model": model,
                "variant": name,
                "n": len(df),
                "accuracy": float(correct.mean()),
                "errors": int((~correct).sum()),
                "clean_accuracy_same_rows": float(clean_same_rows.mean()),
                "drop_pp": float((clean_same_rows.mean() - correct.mean()) * 100),
            })
    return pd.DataFrame(rows)


def plot_results(results, path):
    pivot = results.pivot(index="variant", columns="model", values="accuracy")
    order = [v for v in ["original", "typo_1", "typo_2", "shuffled_words", "drop_keyword",
                         "has_unknown_word", "no_unknown_word"] if v in pivot.index]
    pivot = pivot.loc[order] * 100
    ax = pivot.plot(kind="bar", figsize=(12, 5), width=0.8)
    ax.set_ylabel("Tačnost (%)")
    ax.set_xlabel("")
    ax.set_ylim(max(pivot.min().min() - 3, 0), 100)
    ax.set_title("Robusnost modela na izmenjenim naslovima (val)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    train = pd.read_parquet(args.data_dir / "train.parquet")
    split = pd.read_parquet(args.data_dir / f"{args.split}.parquet")
    variants = build_variants(split, train, seed=args.seed)

    results = evaluate_robustness(args.models, variants, resolve_device(args.device))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_dir / f"{args.split}_robustness.csv", index=False)
    plot_results(results, args.out_dir / f"{args.split}_robustness.png")

    examples = pd.DataFrame({"original": variants["original"]["text"].head(30)})
    for name in ["typo_1", "typo_2", "shuffled_words", "drop_keyword"]:
        examples[name] = variants[name]["text"].reindex(examples.index)
    examples.to_csv(args.out_dir / f"{args.split}_examples.csv", index=False)

    pd.set_option("display.width", 200)
    print("Veličine skupova:")
    for name, df in variants.items():
        print(f"  {name:<18} {len(df)}")
    print("\nTačnost (%):")
    print((results.pivot(index="variant", columns="model", values="accuracy") * 100).round(2).to_string())
    print("\nPad tačnosti u odnosu na iste originalne naslove (procentni poeni):")
    print(results.pivot(index="variant", columns="model", values="drop_pp").round(2).to_string())
    print("\nGreške:")
    print(results.pivot(index="variant", columns="model", values="errors").to_string())
    print("\nPrimeri izmena:")
    for name in ["typo_1", "typo_2", "drop_keyword"]:
        sample = variants[name].head(3)
        for i, text in zip(sample.index, sample["text"]):
            print(f"  [{name}] {variants['original'].loc[i, 'text']}  ->  {text}")
    print(f"\nSačuvano u: {args.out_dir}")


if __name__ == "__main__":
    main()
