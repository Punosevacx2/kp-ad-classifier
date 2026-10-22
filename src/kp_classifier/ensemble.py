import argparse
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar

from kp_classifier.data.dataset import encode_split, make_loader
from kp_classifier.evaluate import build_predictions_frame, evaluate_predictions, load_label_map
from kp_classifier.train import load_trained, predict_proba, resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "reports" / "runs"


def member_proba(model_dir, df, device="cpu"):
    model_dir = Path(model_dir)
    if (model_dir / "model.joblib").exists():
        pipeline = joblib.load(model_dir / "model.joblib")
        return pipeline.predict_proba(df["text"])
    model, tokenizer, _, _ = load_trained(model_dir, device)
    loader = make_loader(encode_split(tokenizer, df), 512, shuffle=False)
    return predict_proba(model, loader, device)


def predict_any(model_dir, df, device="cpu"):
    model_dir = Path(model_dir)
    spec_path = model_dir / "ensemble.yaml"
    if spec_path.exists():
        with open(spec_path, encoding="utf-8") as f:
            spec = yaml.safe_load(f)
        members = [model_dir.parent / m for m in spec["members"]]
        proba, _ = ensemble_proba(members, df, device)
        return apply_temperature(proba, spec.get("temperature", 1.0))
    return member_proba(model_dir, df, device)


def ensemble_proba(model_dirs, df, device="cpu"):
    probas = [member_proba(d, df, device) for d in model_dirs]
    return np.mean(probas, axis=0), probas


def apply_temperature(proba, temperature):
    logits = np.log(np.clip(proba, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(proba, labels):
    labels = np.asarray(labels)
    idx = np.arange(len(labels))

    def nll(log_t):
        p = apply_temperature(proba, np.exp(log_t))
        return -np.mean(np.log(np.clip(p[idx, labels], 1e-12, 1.0)))

    result = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(result.x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model_dirs = [MODELS_DIR / m for m in args.models]
    for d in model_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Ne postoji model: {d}")

    label_map_path = args.data_dir / "label_map.json"
    label_map = load_label_map(label_map_path)
    splits = ["val", "test"] if args.eval_test else ["val"]
    data = {s: pd.read_parquet(args.data_dir / f"{s}.parquet") for s in splits}

    val_proba, member_probas = ensemble_proba(model_dirs, data["val"], device)
    labels = data["val"]["label"].to_numpy()
    print("Članovi ansambla (val tačnost):")
    for name, p in zip(args.models, member_probas):
        print(f"  {name:<30} {(p.argmax(axis=1) == labels).mean():.4f}  greške: {(p.argmax(axis=1) != labels).sum()}")
    print(f"  {'ANSAMBL':<30} {(val_proba.argmax(axis=1) == labels).mean():.4f}  greške: {(val_proba.argmax(axis=1) != labels).sum()}")

    temperature = 1.0
    if args.calibrate:
        temperature = fit_temperature(val_proba, labels)
        print(f"\nTemperatura (fitovana na val): {temperature:.4f}")

    run_dir = RUNS_DIR / args.name
    for split in splits:
        proba = val_proba if split == "val" else ensemble_proba(model_dirs, data[split], device)[0]
        proba = apply_temperature(proba, temperature)
        print()
        evaluate_predictions(build_predictions_frame(data[split], proba), label_map, run_dir, args.name, split)

    model_dir = MODELS_DIR / args.name
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(label_map_path, model_dir / "label_map.json")
    with open(model_dir / "ensemble.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"members": list(args.models), "temperature": temperature}, f, allow_unicode=True)
    print(f"\nAnsambl sačuvan u: {model_dir}")


if __name__ == "__main__":
    main()
