import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

PAD = "<pad>"
UNK = "<unk>"
PAD_ID = 0
UNK_ID = 1
NUM_TOKENS = ["<num1>", "<num2>", "<num3>", "<num4+>"]
TEXT_SPECIAL_TOKENS = ["<phone>", "<id>"]
CHAR_SPECIALS = {"<phone>": "\u0001", "<id>": "\u0002"}

WORD_RE = re.compile(r"<\w+>|\w+")
CHAR_SPECIAL_RE = re.compile("|".join(re.escape(t) for t in CHAR_SPECIALS))


def tokenize_words(text):
    return WORD_RE.findall(text)


def char_sequence(text):
    return CHAR_SPECIAL_RE.sub(lambda m: CHAR_SPECIALS[m.group(0)], text)


def number_shape(token):
    n = len(token)
    return NUM_TOKENS[min(n, 4) - 1]


class Tokenizer:
    def __init__(self, word_itos, char_itos, max_words=20, max_chars=60, min_word_freq=2, min_char_freq=5):
        self.word_itos = list(word_itos)
        self.char_itos = list(char_itos)
        self.word_stoi = {t: i for i, t in enumerate(self.word_itos)}
        self.char_stoi = {c: i for i, c in enumerate(self.char_itos)}
        self.max_words = max_words
        self.max_chars = max_chars
        self.min_word_freq = min_word_freq
        self.min_char_freq = min_char_freq

    @classmethod
    def build(cls, texts, min_word_freq=2, min_char_freq=5, max_words=20, max_chars=60):
        word_counts = Counter()
        char_counts = Counter()
        for text in texts:
            word_counts.update(tokenize_words(text))
            char_counts.update(char_sequence(text))

        reserved_words = [PAD, UNK] + NUM_TOKENS + TEXT_SPECIAL_TOKENS
        words = sorted(
            (t for t, c in word_counts.items() if c >= min_word_freq and t not in reserved_words),
            key=lambda t: (-word_counts[t], t),
        )
        reserved_chars = [PAD, UNK] + list(CHAR_SPECIALS.values())
        chars = sorted(
            (c for c, n in char_counts.items() if n >= min_char_freq and c not in reserved_chars),
            key=lambda c: (-char_counts[c], c),
        )
        return cls(
            reserved_words + words,
            reserved_chars + chars,
            max_words=max_words,
            max_chars=max_chars,
            min_word_freq=min_word_freq,
            min_char_freq=min_char_freq,
        )

    @property
    def word_vocab_size(self):
        return len(self.word_itos)

    @property
    def char_vocab_size(self):
        return len(self.char_itos)

    def word_id(self, token):
        if token in self.word_stoi:
            return self.word_stoi[token]
        if token.isdigit():
            return self.word_stoi[number_shape(token)]
        return UNK_ID

    def encode_words(self, text):
        ids = [self.word_id(t) for t in tokenize_words(text)][: self.max_words]
        return ids + [PAD_ID] * (self.max_words - len(ids))

    def encode_chars(self, text):
        ids = [self.char_stoi.get(c, UNK_ID) for c in char_sequence(text)][: self.max_chars]
        return ids + [PAD_ID] * (self.max_chars - len(ids))

    def encode(self, texts):
        texts = list(texts)
        words = np.array([self.encode_words(t) for t in texts], dtype=np.int64).reshape(len(texts), self.max_words)
        chars = np.array([self.encode_chars(t) for t in texts], dtype=np.int64).reshape(len(texts), self.max_chars)
        return words, chars

    def decode_words(self, ids):
        return [self.word_itos[i] for i in ids if i != PAD_ID]

    def decode_chars(self, ids):
        inverse = {v: k for k, v in CHAR_SPECIALS.items()}
        out = []
        for i in ids:
            if i == PAD_ID:
                continue
            c = self.char_itos[i]
            out.append(inverse.get(c, "\u00bf" if c == UNK else c))
        return "".join(out)

    def to_dict(self):
        return {
            "max_words": self.max_words,
            "max_chars": self.max_chars,
            "min_word_freq": self.min_word_freq,
            "min_char_freq": self.min_char_freq,
            "word_itos": self.word_itos,
            "char_itos": self.char_itos,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["word_itos"],
            data["char_itos"],
            max_words=data["max_words"],
            max_chars=data["max_chars"],
            min_word_freq=data["min_word_freq"],
            min_char_freq=data["min_char_freq"],
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def coverage_stats(tokenizer, texts):
    texts = list(texts)
    num_ids = {tokenizer.word_stoi[t] for t in NUM_TOKENS}
    total_tokens = unk_tokens = num_tokens = 0
    titles_with_unk = truncated_words = truncated_chars = 0
    for text in texts:
        tokens = tokenize_words(text)
        ids = [tokenizer.word_id(t) for t in tokens]
        total_tokens += len(ids)
        unk = sum(i == UNK_ID for i in ids)
        unk_tokens += unk
        num_tokens += sum(i in num_ids for i in ids)
        titles_with_unk += unk > 0
        truncated_words += len(tokens) > tokenizer.max_words
        truncated_chars += len(char_sequence(text)) > tokenizer.max_chars
    n = max(len(texts), 1)
    return {
        "titles": len(texts),
        "tokens": total_tokens,
        "unk_token_rate": unk_tokens / max(total_tokens, 1),
        "num_shape_token_rate": num_tokens / max(total_tokens, 1),
        "titles_with_unk_rate": titles_with_unk / n,
        "truncated_words_rate": truncated_words / n,
        "truncated_chars_rate": truncated_chars / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "vocab.json")
    parser.add_argument("--min-word-freq", type=int, default=2)
    parser.add_argument("--min-char-freq", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=60)
    args = parser.parse_args()

    train = pd.read_parquet(args.data_dir / "train.parquet")
    val = pd.read_parquet(args.data_dir / "val.parquet")

    tokenizer = Tokenizer.build(
        train["text"],
        min_word_freq=args.min_word_freq,
        min_char_freq=args.min_char_freq,
        max_words=args.max_words,
        max_chars=args.max_chars,
    )
    tokenizer.save(args.output)

    print(f"Vokabular reči:     {tokenizer.word_vocab_size}")
    print(f"Vokabular karaktera: {tokenizer.char_vocab_size}")
    printable = [c for c in tokenizer.char_itos if len(c) == 1 and c.isprintable()]
    print(f"Karakteri: {''.join(printable)}")

    for name, df in [("train", train), ("val", val)]:
        stats = coverage_stats(tokenizer, df["text"])
        print(f"\n{name}:")
        print(f"  udeo <unk> tokena:            {stats['unk_token_rate']:.2%}")
        print(f"  udeo naslova sa bar jednim <unk>: {stats['titles_with_unk_rate']:.2%}")
        print(f"  udeo tokena <numN>:            {stats['num_shape_token_rate']:.2%}")
        print(f"  odsečeno (reči / karakteri):   {stats['truncated_words_rate']:.2%} / {stats['truncated_chars_rate']:.2%}")

    print("\nPrimeri kodiranja (val):")
    for text in val["text"].sample(6, random_state=0):
        words, chars = tokenizer.encode([text])
        print(f"  {text}")
        print(f"    reči:      {tokenizer.decode_words(words[0])}")
        print(f"    karakteri: {tokenizer.decode_chars(chars[0])}")

    print(f"\nSačuvano: {args.output}")


if __name__ == "__main__":
    main()
