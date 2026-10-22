import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABEL_MAP = PROJECT_ROOT / "data" / "processed" / "label_map.json"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "reports" / "runs"
COVERAGE_LEVELS = (1.0, 0.99, 0.98, 0.95, 0.9, 0.8)
ACCURACY_TARGETS = (0.99, 0.995, 0.999)


def load_label_map(path=DEFAULT_LABEL_MAP):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def label_names(label_map):
    return [item["group_name"] for item in sorted(label_map, key=lambda x: x["index"])]


def proba_columns(n_classes):
    return [f"proba_{i}" for i in range(n_classes)]


def build_predictions_frame(df, proba):
    proba = np.asarray(proba, dtype=float)
    out = df[["name", "text", "label"]].reset_index(drop=True).copy()
    out["pred"] = proba.argmax(axis=1)
    out["confidence"] = proba.max(axis=1)
    out[proba_columns(proba.shape[1])] = proba
    return out


def extract_proba(predictions, n_classes):
    cols = proba_columns(n_classes)
    if all(c in predictions.columns for c in cols):
        return predictions[cols].to_numpy()
    return None


def top_k_accuracy(y_true, proba, k):
    top_k = np.argsort(-proba, axis=1)[:, :k]
    return float(np.mean([y in row for y, row in zip(y_true, top_k)]))


def expected_calibration_error(y_true, proba, n_bins=15):
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_true
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def macro_f1(y_true, y_pred, n_classes):
    return float(f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0))


def bootstrap_ci(y_true, y_pred, n_classes, n_boot=1000, alpha=0.05, seed=0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    correct = (y_true == y_pred).astype(float)
    accs, f1s = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        accs[i] = correct[idx].mean()
        f1s[i] = macro_f1(y_true[idx], y_pred[idx], n_classes)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "accuracy": [float(np.percentile(accs, lo)), float(np.percentile(accs, hi))],
        "macro_f1": [float(np.percentile(f1s, lo)), float(np.percentile(f1s, hi))],
        "n_boot": n_boot,
        "level": 1 - alpha,
    }


def compute_metrics(y_true, y_pred, names, proba=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(names)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    metrics = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": macro_f1(y_true, y_pred, len(names)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "errors": int((y_true != y_pred).sum()),
        "per_class": {
            name: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for name, p, r, f, s in zip(names, precision, recall, f1, support)
        },
    }
    if proba is not None:
        metrics["log_loss"] = float(log_loss(y_true, np.clip(proba, 1e-12, 1), labels=labels))
        metrics["top2_accuracy"] = top_k_accuracy(y_true, proba, 2)
        metrics["ece"] = expected_calibration_error(y_true, proba)
    return metrics


def risk_coverage_curve(y_true, proba):
    y_true = np.asarray(y_true)
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_true
    order = np.argsort(-confidence, kind="stable")
    k = np.arange(1, len(y_true) + 1)
    return pd.DataFrame({
        "coverage": k / len(y_true),
        "accuracy": np.cumsum(correct[order]) / k,
        "threshold": confidence[order],
    })


def coverage_summary(curve, coverages=COVERAGE_LEVELS, targets=ACCURACY_TARGETS):
    n = len(curve)
    at_coverage = []
    for c in coverages:
        row = curve.iloc[max(int(np.ceil(c * n)) - 1, 0)]
        at_coverage.append({
            "coverage": float(c),
            "accuracy": float(row["accuracy"]),
            "threshold": float(row["threshold"]),
        })
    for_target = []
    for t in targets:
        ok = curve[curve["accuracy"] >= t]
        if len(ok):
            row = ok.iloc[-1]
            for_target.append({
                "target_accuracy": float(t),
                "coverage": float(row["coverage"]),
                "threshold": float(row["threshold"]),
            })
        else:
            for_target.append({"target_accuracy": float(t), "coverage": 0.0, "threshold": None})
    return {"at_coverage": at_coverage, "for_target_accuracy": for_target}


def mcnemar_test(y_true, pred_a, pred_b):
    y_true = np.asarray(y_true)
    correct_a = np.asarray(pred_a) == y_true
    correct_b = np.asarray(pred_b) == y_true
    only_a = int((correct_a & ~correct_b).sum())
    only_b = int((~correct_a & correct_b).sum())
    discordant = only_a + only_b
    p_value = 1.0 if discordant == 0 else float(binomtest(min(only_a, only_b), discordant, 0.5).pvalue)
    return {
        "n": int(len(y_true)),
        "both_correct": int((correct_a & correct_b).sum()),
        "both_wrong": int((~correct_a & ~correct_b).sum()),
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "p_value": p_value,
    }


def compare_predictions(pred_a, pred_b):
    merged = pred_a[["text", "label", "pred"]].merge(
        pred_b[["text", "label", "pred"]], on=["text", "label"], suffixes=("_a", "_b"), validate="one_to_one"
    )
    if len(merged) != len(pred_a) or len(merged) != len(pred_b):
        raise ValueError("Predikcije nisu napravljene na istom skupu primera")
    return mcnemar_test(merged["label"], merged["pred_a"], merged["pred_b"])


def per_class_table(metrics):
    table = pd.DataFrame(metrics["per_class"]).T
    table["support"] = table["support"].astype(int)
    return table.sort_values("f1")


def top_confusions(y_true, y_pred, names, k=10):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    rows = [
        {"stvarna": names[i], "predvidjena": names[j], "broj": int(cm[i, j])}
        for i in range(len(names))
        for j in range(len(names))
        if i != j and cm[i, j] > 0
    ]
    if not rows:
        return pd.DataFrame(columns=["stvarna", "predvidjena", "broj"])
    return pd.DataFrame(rows).sort_values("broj", ascending=False).head(k).reset_index(drop=True)


def error_table(predictions, names):
    errors = predictions[predictions["label"] != predictions["pred"]].copy()
    errors["stvarna"] = errors["label"].map(lambda i: names[i])
    errors["predvidjena"] = errors["pred"].map(lambda i: names[i])
    columns = ["name", "text", "stvarna", "predvidjena"]
    if "confidence" in errors.columns:
        columns.append("confidence")
        errors = errors.sort_values("confidence", ascending=False)
    return errors[columns].reset_index(drop=True)


def plot_confusion_matrix(y_true, y_pred, names, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0) * 100

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        vmin=0,
        vmax=100,
        xticklabels=names,
        yticklabels=names,
        cbar_kws={"label": "% stvarne klase"},
        ax=ax,
    )
    ax.set_xlabel("Predviđena grupa")
    ax.set_ylabel("Stvarna grupa")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability_diagram(y_true, proba, path, title, n_bins=15):
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_true
    edges = np.linspace(0, 1, n_bins + 1)
    centers, accs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            centers.append(confidence[mask].mean())
            accs.append(correct[mask].mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot([0, 1], [0, 1], "--", color="gray", label="savršena kalibracija")
    ax1.plot(centers, accs, "o-", label="model")
    ax1.set_xlabel("Prosečna sigurnost modela")
    ax1.set_ylabel("Tačnost")
    ax1.set_title(title)
    ax1.legend()
    ax2.hist(confidence, bins=edges)
    ax2.set_yscale("log")
    ax2.set_xlabel("Sigurnost modela (max verovatnoća)")
    ax2.set_ylabel("Broj primera (log)")
    ax2.set_title("Raspodela sigurnosti")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_risk_coverage(curve, path, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve["coverage"] * 100, curve["accuracy"] * 100)
    ax.set_xlabel("Pokrivenost: % oglasa koje model klasifikuje automatski")
    ax.set_ylabel("Tačnost na pokrivenim oglasima (%)")
    ax.set_xlim(50, 100)
    ax.set_ylim(max(curve["accuracy"].min() * 100 - 0.2, 95), 100.05)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def format_summary(metrics, model_name, split):
    ci = metrics.get("ci", {})

    def with_ci(key):
        value = f"{metrics[key]:.4f}"
        if key in ci:
            value += f"  [{ci[key][0]:.4f}, {ci[key][1]:.4f}]"
        return value

    lines = [
        f"=== {model_name} | {split} | n={metrics['n']} ===",
        f"accuracy:        {with_ci('accuracy')}",
        f"macro F1:        {with_ci('macro_f1')}",
        f"macro precision: {metrics['macro_precision']:.4f}",
        f"macro recall:    {metrics['macro_recall']:.4f}",
        f"greške:          {metrics['errors']}",
    ]
    if "log_loss" in metrics:
        lines += [
            f"log loss:        {metrics['log_loss']:.4f}",
            f"top-2 accuracy:  {metrics['top2_accuracy']:.4f}",
            f"ECE:             {metrics['ece']:.4f}",
        ]
    if ci:
        lines.append(f"(u zagradama: {ci['level']:.0%} bootstrap interval poverenja, {ci['n_boot']} uzoraka)")
    return "\n".join(lines)


def format_coverage(summary):
    lines = ["Pokrivenost i tačnost (oglasi sortirani po sigurnosti modela):"]
    for row in summary["at_coverage"]:
        lines.append(
            f"  pokrivenost {row['coverage']:>6.1%}: tačnost {row['accuracy']:.4f}  (prag sigurnosti {row['threshold']:.3f})"
        )
    for row in summary["for_target_accuracy"]:
        threshold = "-" if row["threshold"] is None else f"{row['threshold']:.3f}"
        lines.append(
            f"  za tačnost ≥ {row['target_accuracy']:.1%}: pokrivenost {row['coverage']:.1%}  (prag {threshold})"
        )
    return "\n".join(lines)


def evaluate_predictions(predictions, label_map, out_dir, model_name, split, verbose=True, n_boot=1000):
    names = label_names(label_map)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = predictions["label"].to_numpy()
    y_pred = predictions["pred"].to_numpy()
    proba = extract_proba(predictions, len(names))

    metrics = compute_metrics(y_true, y_pred, names, proba)
    metrics["model"] = model_name
    metrics["split"] = split
    if n_boot:
        metrics["ci"] = bootstrap_ci(y_true, y_pred, len(names), n_boot=n_boot)

    curve = None
    if proba is not None:
        curve = risk_coverage_curve(y_true, proba)
        metrics["risk_coverage"] = coverage_summary(curve)

    with open(out_dir / f"{split}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    per_class_table(metrics).to_csv(out_dir / f"{split}_per_class.csv")
    confusions = top_confusions(y_true, y_pred, names)
    confusions.to_csv(out_dir / f"{split}_top_confusions.csv", index=False)
    error_table(predictions, names).to_csv(out_dir / f"{split}_errors.csv", index=False)
    predictions.to_parquet(out_dir / f"{split}_predictions.parquet", index=False)

    plot_confusion_matrix(
        y_true, y_pred, names, out_dir / f"{split}_confusion_matrix.png",
        f"{model_name}: konfuziona matrica ({split})",
    )
    if proba is not None:
        plot_reliability_diagram(
            y_true, proba, out_dir / f"{split}_reliability.png",
            f"{model_name}: kalibracija ({split})",
        )
        plot_risk_coverage(
            curve, out_dir / f"{split}_risk_coverage.png",
            f"{model_name}: tačnost i pokrivenost ({split})",
        )

    if verbose:
        print(format_summary(metrics, model_name, split))
        if "risk_coverage" in metrics:
            print("\n" + format_coverage(metrics["risk_coverage"]))
        print("\nPo klasama (sortirano po F1):")
        print(per_class_table(metrics).round(4).to_string())
        print("\nNajčešće zamene:")
        print(confusions.to_string(index=False) if len(confusions) else "nema grešaka")
        print(f"\nRezultati sačuvani u: {out_dir}")
    return metrics


def compare_runs(runs_dir=DEFAULT_RUNS_DIR, split="val"):
    rows = []
    for path in sorted(Path(runs_dir).glob(f"*/{split}_metrics.json")):
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        ci = m.get("ci", {})
        rows.append({
            "run": path.parent.name,
            "accuracy": m["accuracy"],
            "acc_ci_low": ci.get("accuracy", [None, None])[0],
            "acc_ci_high": ci.get("accuracy", [None, None])[1],
            "macro_f1": m["macro_f1"],
            "errors": m["errors"],
            "log_loss": m.get("log_loss"),
            "ece": m.get("ece"),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def format_mcnemar(result, name_a, name_b):
    return "\n".join([
        f"=== McNemar: {name_a} (A) vs {name_b} (B) | n={result['n']} ===",
        f"oba tačna:          {result['both_correct']}",
        f"oba pogrešna:       {result['both_wrong']}",
        f"samo A tačan:       {result['only_a_correct']}",
        f"samo B tačan:       {result['only_b_correct']}",
        f"p-vrednost (egzaktni test): {result['p_value']:.4f}",
        "razlika je statistički značajna (p < 0.05)" if result["p_value"] < 0.05
        else "razlika NIJE statistički značajna (p ≥ 0.05)",
    ])


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--predictions", type=Path, required=True)
    ev.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    ev.add_argument("--out-dir", type=Path, required=True)
    ev.add_argument("--model-name", default="model")
    ev.add_argument("--split", default="val")

    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    cmp_.add_argument("--split", default="val")

    mc = sub.add_parser("mcnemar")
    mc.add_argument("run_a")
    mc.add_argument("run_b")
    mc.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    mc.add_argument("--split", default="val")

    args = parser.parse_args()
    if args.command == "evaluate":
        predictions = pd.read_parquet(args.predictions)
        evaluate_predictions(predictions, load_label_map(args.label_map), args.out_dir, args.model_name, args.split)
    elif args.command == "compare":
        table = compare_runs(args.runs_dir, args.split)
        print(table.round(4).to_string(index=False) if len(table) else "Nema rezultata.")
    else:
        pred_a = pd.read_parquet(args.runs_dir / args.run_a / f"{args.split}_predictions.parquet")
        pred_b = pd.read_parquet(args.runs_dir / args.run_b / f"{args.split}_predictions.parquet")
        print(format_mcnemar(compare_predictions(pred_a, pred_b), args.run_a, args.run_b))


if __name__ == "__main__":
    main()
