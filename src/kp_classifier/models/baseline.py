import argparse
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import f1_score, log_loss

from kp_classifier.evaluate import (
    build_predictions_frame,
    evaluate_predictions,
    label_names,
    load_label_map,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "baseline_tfidf.yaml"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "reports" / "runs"
TOKEN_PATTERN = r"<\w+>|\w+"


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_splits(data_dir=DATA_DIR):
    return {name: pd.read_parquet(Path(data_dir) / f"{name}.parquet") for name in ["train", "val", "test"]}


def build_pipeline(config, C):
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            token_pattern=TOKEN_PATTERN,
            lowercase=False,
            ngram_range=tuple(config["word_ngram_range"]),
            min_df=config["min_df"],
            sublinear_tf=config["sublinear_tf"],
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            lowercase=False,
            ngram_range=tuple(config["char_ngram_range"]),
            min_df=config["min_df"],
            sublinear_tf=config["sublinear_tf"],
        )),
    ])
    classifier = LogisticRegression(C=C, max_iter=config["max_iter"])
    return Pipeline([("features", features), ("clf", classifier)])


def select_C(train, val, config, verbose=True):
    tolerance = config.get("C_tolerance", 0.0)
    y = val["label"].to_numpy()
    results, pipelines = [], {}
    for C in config["C_grid"]:
        start = time.time()
        pipeline = build_pipeline(config, C)
        pipeline.fit(train["text"], train["label"])
        proba = pipeline.predict_proba(val["text"])
        pred = pipeline.classes_[proba.argmax(axis=1)]
        macro_f1 = f1_score(y, pred, average="macro")
        accuracy = float((pred == y).mean())
        val_log_loss = log_loss(y, proba, labels=pipeline.classes_)
        elapsed = time.time() - start
        results.append({
            "C": C,
            "val_macro_f1": macro_f1,
            "val_accuracy": accuracy,
            "val_log_loss": val_log_loss,
            "seconds": round(elapsed, 1),
        })
        pipelines[C] = pipeline
        if verbose:
            print(f"C={C:<6} val macro F1={macro_f1:.4f}  accuracy={accuracy:.4f}  "
                  f"log loss={val_log_loss:.4f}  ({elapsed:.1f}s)")
    search = pd.DataFrame(results)
    best_f1 = search["val_macro_f1"].max()
    candidates = search[search["val_macro_f1"] >= best_f1 - tolerance]
    best_C = float(candidates.sort_values("val_log_loss").iloc[0]["C"])
    return best_C, pipelines[best_C], search


def feature_names(pipeline):
    union = pipeline.named_steps["features"]
    names = []
    for prefix, vectorizer in union.transformer_list:
        names.extend(f"{prefix}:{f}" for f in vectorizer.get_feature_names_out())
    return np.array(names)


def top_features(pipeline, names, k=20):
    coef = pipeline.named_steps["clf"].coef_
    all_names = feature_names(pipeline)
    rows = []
    for class_index, class_name in enumerate(names):
        order = np.argsort(-coef[class_index])[:k]
        for rank, idx in enumerate(order, start=1):
            rows.append({
                "grupa": class_name,
                "rang": rank,
                "osobina": all_names[idx],
                "tezina": float(coef[class_index, idx]),
            })
    return pd.DataFrame(rows)


def feature_weight(pipeline, names, feature):
    all_names = list(feature_names(pipeline))
    if feature not in all_names:
        return None
    idx = all_names.index(feature)
    coef = pipeline.named_steps["clf"].coef_[:, idx]
    return pd.Series(coef, index=names).sort_values(ascending=False)


def prior_predictions(train, df, n_classes):
    priors = np.bincount(train["label"], minlength=n_classes) / len(train)
    proba = np.tile(priors, (len(df), 1))
    return build_predictions_frame(df, proba)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--eval-test", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    name = config["name"]
    splits = load_splits(args.data_dir)
    label_map = load_label_map(args.data_dir / "label_map.json")
    names = label_names(label_map)
    eval_splits = ["val", "test"] if args.eval_test else ["val"]

    print("=== Trivijalni baseline (uvek najčešća klasa iz trening skupa) ===")
    for split in eval_splits:
        evaluate_predictions(
            prior_predictions(splits["train"], splits[split], len(names)),
            label_map, RUNS_DIR / "majority", "majority", split, verbose=False,
        )
    with open(RUNS_DIR / "majority" / "val_metrics.json", encoding="utf-8") as f:
        majority = json.load(f)
    print(f"val accuracy={majority['accuracy']:.4f}  macro F1={majority['macro_f1']:.4f}\n")

    print(f"=== {name}: izbor C na validacionom skupu ===")
    best_C, pipeline, search = select_C(splits["train"], splits["val"], config)
    print(f"\nNajbolji C: {best_C}")

    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    search.to_csv(run_dir / "C_search.csv", index=False)

    n_features = len(feature_names(pipeline))
    print(f"Broj osobina: {n_features}\n")

    for split in eval_splits:
        proba = pipeline.predict_proba(splits[split]["text"])
        predictions = build_predictions_frame(splits[split], proba)
        evaluate_predictions(predictions, label_map, run_dir, name, split)
        print()

    features = top_features(pipeline, names, config["top_features"])
    features.to_csv(run_dir / "top_features.csv", index=False)
    print("Najjače osobine po grupi (top 8):")
    for group, sub in features.groupby("grupa", sort=False):
        print(f"  {group}: " + ", ".join(sub["osobina"].head(8)))

    for token in ["word:<phone>", "word:<id>"]:
        weights = feature_weight(pipeline, names, token)
        if weights is not None:
            print(f"\nTežine za {token}:")
            print(weights.round(3).to_string())

    model_dir = MODELS_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "model.joblib")
    shutil.copy(args.data_dir / "label_map.json", model_dir / "label_map.json")
    with open(model_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({**config, "best_C": best_C, "n_features": n_features}, f, allow_unicode=True)
    print(f"\nModel sačuvan u: {model_dir}")


if __name__ == "__main__":
    main()
    