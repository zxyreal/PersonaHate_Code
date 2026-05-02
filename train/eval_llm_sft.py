#!/usr/bin/env python3
"""Evaluate LLM SFT model on hate speech benchmarks via vLLM."""

import sys
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
sys.stdout.reconfigure(line_buffering=True)

ROOT_DIR = Path(__file__).parent

# Load config
CONFIG_PATH = ROOT_DIR / "config.json"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f).get("eval", {})
else:
    CONFIG = {}

DATA_DIR = ROOT_DIR / CONFIG.get("data_dir", "data/benchmarks")

SYSTEM_PROMPT = "You are a hate speech classifier. Analyze the given text and respond with only 'yes' if it is hate speech, or 'no' if it is not."
USER_TEMPLATE = """Is the following text hate speech? Hate speech is content that expresses hatred or promotes violence against people based on race, ethnicity, religion, gender, sexual orientation, disability, or origin.

Text: "{text}"

Answer with only one word: yes or no"""


def parse_yes_no(text):
    if not text:
        return None
    text = text.strip().lower()
    first = text.split()[0].strip(".,!?:;\"'") if text.split() else ""
    if first in ["yes", "true", "1"]:
        return 1
    elif first in ["no", "false", "0"]:
        return 0
    if "yes" in text[:20]:
        return 1
    elif "no" in text[:20]:
        return 0
    return None


def predict_batch(client, model_name, texts, workers=16):
    preds = [None] * len(texts)

    def predict_one(idx):
        text = texts[idx][:1500]
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(text=text)},
                ],
                temperature=0, max_tokens=10,
            )
            raw = resp.choices[0].message.content.strip()
            return idx, parse_yes_no(raw)
        except Exception as e:
            return idx, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(predict_one, i) for i in range(len(texts))]
        done = 0
        for f in as_completed(futures):
            idx, pred = f.result()
            preds[idx] = pred
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(texts)}")

    return preds


def load_hatexplain():
    df = pd.read_csv(DATA_DIR / CONFIG.get("hatexplain", "all_7_models_voting.csv"))
    df = df[df["label"] != "offensive"]
    return df["text"].tolist(), df["ground_truth"].astype(int).tolist(), "HateXplain (no-off)"


def load_hatebench():
    from datasets import load_dataset
    ds = load_dataset("TrustAIRLab/HateBenchSet", cache_dir=str(DATA_DIR / "hatebench_cache"))
    df = pd.DataFrame(ds["train"])
    return df["text"].tolist(), [int(x) for x in df["hate_label"].tolist()], "HateBenchSet"


def load_davidson():
    df = pd.read_csv(DATA_DIR / CONFIG.get("davidson", "davidson_labeled_data.csv"))
    df = df[df["class"].astype(int) != 1]
    df["binary_label"] = (df["class"].astype(int) == 0).astype(int)
    return df["tweet"].tolist(), df["binary_label"].tolist(), "Davidson (no-off)"


def load_mhs():
    df = pd.read_csv(DATA_DIR / CONFIG.get("mhs", "mhs_aggregated.csv"))
    df = df[df["label"].notna()]
    return df["text"].tolist(), [int(x) for x in df["label"].tolist()], "MHS"


def load_newwave():
    df = pd.read_csv(DATA_DIR / CONFIG.get("newwave", "new_wave_hate/data_ground_truth.csv"))
    return df["text"].tolist(), [int(x) for x in df["ground_truth"].tolist()], "NewWave"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--model", default=None, help="Model name (auto-detect if not set)")
    parser.add_argument("--output", default="llm_sft_eval_results.json")
    args = parser.parse_args()

    client = OpenAI(api_key="EMPTY", base_url=f"{args.host}/v1")

    if args.model:
        model_name = args.model
    else:
        models = client.models.list()
        model_name = models.data[0].id
    print(f"Model: {model_name}")

    loaders = [load_hatexplain, load_hatebench, load_davidson, load_mhs, load_newwave]
    results = []

    for loader in loaders:
        try:
            texts, labels, name = loader()
        except Exception as e:
            print(f"Error loading {loader.__name__}: {e}")
            continue

        valid = [(t, l) for t, l in zip(texts, labels) if isinstance(t, str) and len(t.strip()) > 0]
        texts_c = [t for t, l in valid]
        labels_c = [l for t, l in valid]

        print(f"\n{'='*60}")
        print(f"  {name}: {len(texts_c)} samples")

        preds = predict_batch(client, model_name, texts_c)

        valid_mask = [p is not None for p in preds]
        y_true = [l for l, v in zip(labels_c, valid_mask) if v]
        y_pred = [p for p, v in zip(preds, valid_mask) if v]

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

        print(f"  Valid: {len(y_true)}/{len(texts_c)}")
        print(f"  Acc: {acc*100:.1f}%  F1: {f1*100:.1f}%  Prec: {prec*100:.1f}%  Rec: {rec*100:.1f}%")
        print(f"{'='*60}")

        results.append({
            "dataset": name, "samples": len(y_true),
            "accuracy": round(acc*100, 1), "f1": round(f1*100, 1),
            "precision": round(prec*100, 1), "recall": round(rec*100, 1),
        })

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Dataset':<20} {'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6}")
    print(f"  {'-'*50}")
    for r in results:
        print(f"  {r['dataset']:<20} {r['accuracy']:>5.1f}% {r['f1']:>5.1f}% {r['precision']:>5.1f}% {r['recall']:>5.1f}%")

    out_path = Path(__file__).parent / args.output
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
