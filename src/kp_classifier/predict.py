import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from kp_classifier.data.preprocess import normalize_text
from kp_classifier.ensemble import apply_temperature
from kp_classifier.evaluate import load_label_map
from kp_classifier.train import load_trained, resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL = "ens_mm_ls_cal"
DEFAULT_THRESHOLD = 0.9


class _SklearnMember:
    def __init__(self, model_dir):
        self.pipeline = joblib.load(Path(model_dir) / "model.joblib")

    def __call__(self, texts):
        return self.pipeline.predict_proba(list(texts))


class _TorchMember:
    def __init__(self, model_dir, device):
        self.device = device
        self.model, self.tokenizer, _, _ = load_trained(model_dir, device)

    @torch.no_grad()
    def __call__(self, texts, batch_size=512):
        words, chars = self.tokenizer.encode(texts)
        loader = DataLoader(TensorDataset(torch.tensor(words), torch.tensor(chars)), batch_size=batch_size)
        outputs = []
        for w, c in loader:
            logits = self.model(w.to(self.device), c.to(self.device))
            outputs.append(torch.softmax(logits, dim=1).cpu())
        return torch.cat(outputs).numpy()


def _load_member(model_dir, device):
    model_dir = Path(model_dir)
    if (model_dir / "model.joblib").exists():
        return _SklearnMember(model_dir)
    if (model_dir / "model.pt").exists():
        return _TorchMember(model_dir, device)
    raise FileNotFoundError(f"U {model_dir} nema model.joblib ni model.pt")


class Predictor:
    def __init__(self, model_dir, device="cpu"):
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Ne postoji model: {model_dir}")
        self.device = resolve_device(device)
        self.label_map = sorted(load_label_map(model_dir / "label_map.json"), key=lambda x: x["index"])
        spec_path = model_dir / "ensemble.yaml"
        if spec_path.exists():
            with open(spec_path, encoding="utf-8") as f:
                spec = yaml.safe_load(f)
            self.members = [_load_member(model_dir.parent / m, self.device) for m in spec["members"]]
            self.temperature = float(spec.get("temperature", 1.0))
        else:
            self.members = [_load_member(model_dir, self.device)]
            self.temperature = 1.0
        self.model_name = model_dir.name

    def predict_proba(self, titles):
        texts = [normalize_text(t) for t in titles]
        if not texts:
            return texts, np.zeros((0, len(self.label_map)))
        proba = np.mean([member(texts) for member in self.members], axis=0)
        return texts, apply_temperature(proba, self.temperature)

    def predict(self, titles, top_k=3, threshold=DEFAULT_THRESHOLD):
        titles = list(titles)
        texts, proba = self.predict_proba(titles)
        results = []
        for title, text, p in zip(titles, texts, proba):
            order = np.argsort(-p)[:top_k]
            top = [
                {
                    "group_id": self.label_map[i]["group_id"],
                    "group_name": self.label_map[i]["group_name"],
                    "probability": round(float(p[i]), 4),
                }
                for i in order
            ]
            results.append({
                "name": title,
                "normalized": text,
                "group_id": top[0]["group_id"],
                "group_name": top[0]["group_name"],
                "confidence": top[0]["probability"],
                "needs_review": bool(text == "" or top[0]["probability"] < threshold),
                "top_k": top,
            })
        return results


def read_titles(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        column = "name" if "name" in df.columns else df.columns[0]
        return df[column].fillna("").astype(str).tolist()
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def format_result(result):
    flag = "  [PROVERITI]" if result["needs_review"] else ""
    lines = [f"{result['name']}", f"  -> {result['group_name']} ({result['group_id']})  {result['confidence']:.3f}{flag}"]
    for rank, item in enumerate(result["top_k"][1:], start=2):
        lines.append(f"     {rank}. {item['group_name']} ({item['group_id']})  {item['probability']:.3f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Klasifikacija naslova oglasa u grupe")
    parser.add_argument("titles", nargs="*", help="jedan ili više naslova oglasa")
    parser.add_argument("--file", type=Path, help="txt (jedan naslov po redu) ili csv sa kolonom 'name'")
    parser.add_argument("--output", type=Path, help="csv fajl za rezultate (uz --file)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="ime modela u folderu models/")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="ispod ove sigurnosti oglas se označava za ručnu proveru")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", action="store_true", help="ispis u JSON formatu")
    args = parser.parse_args()

    titles = list(args.titles)
    if args.file:
        titles += read_titles(args.file)
    if not titles:
        parser.error("navedi bar jedan naslov ili --file")

    predictor = Predictor(args.models_dir / args.model, device=args.device)
    results = predictor.predict(titles, top_k=args.top_k, threshold=args.threshold)

    if args.output:
        rows = [{k: v for k, v in r.items() if k != "top_k"} for r in results]
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"Sačuvano {len(rows)} predikcija u {args.output}", file=sys.stderr)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif not args.output:
        for result in results:
            print(format_result(result))


if __name__ == "__main__":
    main()
