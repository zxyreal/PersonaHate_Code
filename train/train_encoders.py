#!/usr/bin/env python3
"""
Train encoder models on PersonaHate balanced dataset for hate speech classification.
Models: BERT-base, RoBERTa-base, DeBERTa-v3-base, XLM-RoBERTa-base
"""

import argparse
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT_DIR = Path(__file__).parent

# Load config
CONFIG_PATH = ROOT_DIR / "config.json"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f).get("train_encoders", {})
else:
    CONFIG = {}

DATA_PATH = ROOT_DIR / CONFIG.get("data_path", "data/personahate_train_balanced_6judge.jsonl")
OUTPUT_DIR = ROOT_DIR / CONFIG.get("output_dir", "models")

MODELS = {
    "bert-base": {"name": "bert-base-uncased", "max_length": 512},
    "roberta-base": {"name": "roberta-base", "max_length": 512},
    "deberta-v3-base": {"name": "microsoft/deberta-v3-base", "max_length": 512},
    "xlm-roberta-base": {"name": "xlm-roberta-base", "max_length": 512},
}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def load_data(max_samples=None):
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_json(DATA_PATH, lines=True)
    df = df[df["text"].notna() & (df["text"].str.len() > 0)]
    df["text"] = df["text"].astype(str)
    if "split_label" in df.columns and "label" not in df.columns:
        df["label"] = df["split_label"]
    print(f"Total: {len(df):,}, Hate: {(df['label']==1).sum():,}, Non-hate: {(df['label']==0).sum():,}")

    if max_samples and len(df) > max_samples:
        hate_df = df[df["label"] == 1].sample(n=min(max_samples // 2, (df["label"] == 1).sum()), random_state=42)
        nonhate_df = df[df["label"] == 0].sample(n=min(max_samples // 2, (df["label"] == 0).sum()), random_state=42)
        df = pd.concat([hate_df, nonhate_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Sampled to: {len(df):,}")

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])
    print(f"Train: {len(train_df):,}, Test: {len(test_df):,}")
    return train_df, test_df


def train_model(model_key, max_samples=None, epochs=3, batch_size=16):
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
        return

    config = MODELS[model_key]
    model_name = config["name"]
    max_length = config["max_length"]

    print("=" * 70)
    print(f"Training: {model_key} ({model_name})")
    print(f"Dataset: PersonaHate balanced ({DATA_PATH.name})")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_df, test_df = load_data(max_samples)

    def make_dataset(df):
        ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
        ds = ds.map(
            lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=max_length),
            batched=True,
        )
        return ds

    train_dataset = make_dataset(train_df)
    test_dataset = make_dataset(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    output_dir = OUTPUT_DIR / f"{model_key}_personahate" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\nStarting training...")
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    results = trainer.evaluate()
    print(f"\nFinal: Acc {results['eval_accuracy']*100:.1f}%  F1 {results['eval_f1']*100:.1f}%  "
          f"Prec {results['eval_precision']*100:.1f}%  Rec {results['eval_recall']*100:.1f}%")

    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)

    print(f"Model saved to: {final_dir}")
    return str(final_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    train_model(args.model, args.max_samples, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
