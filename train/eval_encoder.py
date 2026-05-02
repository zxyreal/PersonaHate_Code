#!/usr/bin/env python3
"""
Evaluate a trained encoder model on multiple hate speech benchmarks.
"""

import argparse
import json
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from datasets import load_dataset

load_dotenv()

ROOT_DIR = Path(__file__).parent

# Load config
CONFIG_PATH = ROOT_DIR / "config.json"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f).get("eval", {})
else:
    CONFIG = {}

DATA_DIR = ROOT_DIR / CONFIG.get("data_dir", "data/benchmarks")


def predict_batch(model, tokenizer, texts, device, batch_size=64, max_length=512):
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
        preds.extend(batch_preds)
    return preds


def load_hatexplain():
    df = pd.read_csv(DATA_DIR / CONFIG.get("hatexplain", "all_7_models_voting.csv"))
    df = df[df["label"] != "offensive"]
    df["binary_label"] = df["ground_truth"].astype(int)
    return df["text"].tolist(), df["binary_label"].tolist(), "HateXplain (no-off)"


def load_hatebench():
    ds = load_dataset("TrustAIRLab/HateBenchSet",
                      cache_dir=str(DATA_DIR / "hatebench_cache"))
    df = pd.DataFrame(ds["train"])
    texts = df["text"].tolist()
    labels = [int(x) for x in df["hate_label"].tolist()]
    return texts, labels, "HateBenchSet"


def load_davidson():
    df = pd.read_csv(DATA_DIR / CONFIG.get("davidson", "davidson_labeled_data.csv"))
    df = df[df["class"].astype(int) != 1]  # remove offensive (class=1)
    df["binary_label"] = (df["class"].astype(int) == 0).astype(int)  # 0=hate
    return df["tweet"].tolist(), df["binary_label"].tolist(), "Davidson (no-off)"


def load_mhs():
    df = pd.read_csv(DATA_DIR / CONFIG.get("mhs", "mhs_aggregated.csv"))
    df = df[df["label"].notna()]
    texts = df["text"].tolist()
    labels = [int(x) for x in df["label"].tolist()]
    return texts, labels, "MHS"


def load_newwave():
    df = pd.read_csv(DATA_DIR / CONFIG.get("newwave", "new_wave_hate/data_ground_truth.csv"))
    texts = df["text"].tolist()
    labels = [int(x) for x in df["ground_truth"].tolist()]
    return texts, labels, "NewWave"


def evaluate_on_dataset(model, tokenizer, texts, labels, dataset_name, device):
    valid = [(t, l) for t, l in zip(texts, labels) if isinstance(t, str) and len(t.strip()) > 0]
    texts_clean = [t for t, l in valid]
    labels_clean = [l for t, l in valid]

    preds = predict_batch(model, tokenizer, texts_clean, device)

    acc = accuracy_score(labels_clean, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels_clean, preds, average="binary", zero_division=0)

    n_hate = sum(labels_clean)
    n_total = len(labels_clean)

    print(f"\n{'='*60}")
    print(f"  {dataset_name}: {n_total:,} samples ({n_hate:,} hate, {n_total-n_hate:,} non-hate)")
    print(f"  Acc: {acc*100:.1f}%  F1: {f1*100:.1f}%  Prec: {prec*100:.1f}%  Rec: {rec*100:.1f}%")
    print(f"{'='*60}")

    return {
        "dataset": dataset_name,
        "samples": n_total,
        "accuracy": round(acc * 100, 1),
        "f1": round(f1 * 100, 1),
        "precision": round(prec * 100, 1),
        "recall": round(rec * 100, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--datasets", type=str, default="all",
                        help="Comma-separated: hatexplain,hatebench,davidson,mhs,newwave,all")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    loaders = {
        "hatexplain": load_hatexplain,
        "hatebench": load_hatebench,
        "davidson": load_davidson,
        "mhs": load_mhs,
        "newwave": load_newwave,
    }

    if args.datasets == "all":
        datasets_to_eval = list(loaders.keys())
    else:
        datasets_to_eval = [d.strip() for d in args.datasets.split(",")]

    results = []
    for ds_key in datasets_to_eval:
        if ds_key not in loaders:
            print(f"Unknown dataset: {ds_key}")
            continue
        try:
            texts, labels, name = loaders[ds_key]()
            r = evaluate_on_dataset(model, tokenizer, texts, labels, name, device)
            results.append(r)
        except Exception as e:
            print(f"Error on {ds_key}: {e}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Dataset':<15} {'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6}")
    print(f"  {'-'*45}")
    for r in results:
        print(f"  {r['dataset']:<15} {r['accuracy']:>5.1f}% {r['f1']:>5.1f}% {r['precision']:>5.1f}% {r['recall']:>5.1f}%")

    output_path = Path(args.model_path).parent.parent / "eval_all_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
