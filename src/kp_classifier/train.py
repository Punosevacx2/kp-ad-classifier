import argparse
import copy
import json
import random
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score

from kp_classifier.data.dataset import encode_split, make_loader
from kp_classifier.data.tokenizer import Tokenizer
from kp_classifier.evaluate import build_predictions_frame, evaluate_predictions, load_label_map
from kp_classifier.models.textcnn import build_model, parameter_table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "textcnn.yaml"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "reports" / "runs"


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(config, overrides):
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not key or not _:
            raise ValueError(f"Neispravan --set argument: {item} (očekivano kljuc=vrednost)")
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise KeyError(f"Nepoznat ključ u configu: {key}")
            target = target[part]
        if parts[-1] not in target:
            raise KeyError(f"Nepoznat ključ u configu: {key}")
        target[parts[-1]] = yaml.safe_load(raw)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    outputs = []
    for words, chars, *_ in loader:
        logits = model(words.to(device), chars.to(device))
        outputs.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(outputs).numpy()


def run_epoch(model, loader, criterion, device, optimizer=None, grad_clip=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_correct, total = 0.0, 0, 0
    preds, labels = [], []
    with torch.set_grad_enabled(training):
        for words, chars, y in loader:
            words, chars, y = words.to(device), chars.to(device), y.to(device)
            logits = model(words, chars)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            pred = logits.argmax(dim=1)
            total_loss += loss.item() * y.size(0)
            total_correct += (pred == y).sum().item()
            total += y.size(0)
            preds.append(pred.cpu())
            labels.append(y.cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def train_model(config, train_dataset, val_dataset, num_classes, word_vocab_size, char_vocab_size, verbose=True):
    training = config["training"]
    set_seed(config["seed"])
    device = resolve_device(training.get("device", "auto"))

    model = build_model(config["model"], word_vocab_size, char_vocab_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=training.get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=training["lr_factor"], patience=training["lr_patience"]
    )
    train_loader = make_loader(train_dataset, training["batch_size"], shuffle=True, seed=config["seed"])
    val_loader = make_loader(val_dataset, training["batch_size"] * 4, shuffle=False)

    history = []
    best_loss, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
    if verbose:
        print(f"Uređaj: {device}")
        print(f"{'ep':>3} {'train loss':>10} {'train acc':>9} {'val loss':>9} {'val acc':>8} {'val F1':>7} {'lr':>8} {'s':>5}")

    for epoch in range(1, training["max_epochs"] + 1):
        start = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, training.get("grad_clip"))
        val_metrics = run_epoch(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["loss"])
        elapsed = time.time() - start

        history.append({
            "epoch": epoch,
            "lr": lr,
            "seconds": round(elapsed, 1),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })
        improved = val_metrics["loss"] < best_loss
        if improved:
            best_loss, best_epoch, bad_epochs = val_metrics["loss"], epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad_epochs += 1
        if verbose:
            print(
                f"{epoch:>3} {train_metrics['loss']:>10.4f} {train_metrics['accuracy']:>9.4f} "
                f"{val_metrics['loss']:>9.4f} {val_metrics['accuracy']:>8.4f} {val_metrics['macro_f1']:>7.4f} "
                f"{lr:>8.1e} {elapsed:>5.1f}{'  *' if improved else ''}"
            )
        if bad_epochs >= training["early_stopping_patience"]:
            if verbose:
                print(f"Rano zaustavljanje: val loss se nije poboljšao {bad_epochs} epohe zaredom")
            break

    model.load_state_dict(best_state)
    if verbose:
        print(f"Najbolja epoha: {best_epoch} (val loss {best_loss:.4f})")
    return model, pd.DataFrame(history), best_epoch, device


def plot_history(history, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, label in zip(axes, ["loss", "accuracy", "macro_f1"], ["Loss", "Accuracy", "Macro F1"]):
        ax.plot(history["epoch"], history[f"train_{metric}"], "o-", label="train")
        ax.plot(history["epoch"], history[f"val_{metric}"], "o-", label="val")
        ax.set_xlabel("Epoha")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_model(model_dir, model, tokenizer, label_map_path, config, extra):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pt")
    tokenizer.save(model_dir / "vocab.json")
    shutil.copy(label_map_path, model_dir / "label_map.json")
    with open(model_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({**config, **extra}, f, allow_unicode=True, sort_keys=False)


def load_trained(model_dir, device="cpu"):
    model_dir = Path(model_dir)
    config = load_config(model_dir / "config.yaml")
    tokenizer = Tokenizer.load(model_dir / "vocab.json")
    label_map = load_label_map(model_dir / "label_map.json")
    model = build_model(config["model"], tokenizer.word_vocab_size, tokenizer.char_vocab_size, len(label_map))
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device, weights_only=True))
    model.to(device).eval()
    return model, tokenizer, label_map, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--name", default=None)
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KLJUC=VREDNOST")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.set)
    if args.name:
        config["name"] = args.name
    name = config["name"]

    splits = {s: pd.read_parquet(args.data_dir / f"{s}.parquet") for s in ["train", "val", "test"]}
    label_map_path = args.data_dir / "label_map.json"
    label_map = load_label_map(label_map_path)

    tok_cfg = config["tokenizer"]
    tokenizer = Tokenizer.build(
        splits["train"]["text"],
        min_word_freq=tok_cfg["min_word_freq"],
        min_char_freq=tok_cfg["min_char_freq"],
        max_words=tok_cfg["max_words"],
        max_chars=tok_cfg["max_chars"],
    )
    datasets = {s: encode_split(tokenizer, df) for s, df in splits.items()}
    print(f"Vokabular: {tokenizer.word_vocab_size} reči, {tokenizer.char_vocab_size} karaktera")

    model, history, best_epoch, device = train_model(
        config, datasets["train"], datasets["val"], len(label_map),
        tokenizer.word_vocab_size, tokenizer.char_vocab_size,
    )

    params = parameter_table(model)
    print("\nParametri:")
    for layer, n in params:
        print(f"  {layer:<35} {n:>10,}")

    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(run_dir / "history.csv", index=False)
    plot_history(history, run_dir / "learning_curves.png", f"{name}: krive učenja")

    eval_splits = ["val", "test"] if args.eval_test else ["val"]
    batch_size = config["training"]["batch_size"] * 4
    for split in eval_splits:
        print()
        proba = predict_proba(model, make_loader(datasets[split], batch_size, shuffle=False), device)
        evaluate_predictions(build_predictions_frame(splits[split], proba), label_map, run_dir, name, split)

    extra = {
        "best_epoch": int(best_epoch),
        "epochs_trained": int(len(history)),
        "word_vocab_size": tokenizer.word_vocab_size,
        "char_vocab_size": tokenizer.char_vocab_size,
        "n_parameters": int(params[-1][1]),
        "device": str(device),
        "train_seconds": float(history["seconds"].sum()),
    }
    save_model(MODELS_DIR / name, model.cpu(), tokenizer, label_map_path, config, extra)
    with open(run_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(extra, f, indent=2)
    print(f"\nModel sačuvan u: {MODELS_DIR / name}")


if __name__ == "__main__":
    main()
