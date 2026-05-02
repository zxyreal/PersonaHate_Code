#!/usr/bin/env python3
"""
Build group-balanced training set from per-model 6-judge labeled data.

Reads all 8 model directories, merges, and samples up to N hate + N non-hate
per identity group (1:1 balance). Groups with fewer than N hate samples
contribute all available, matched by equal non-hate.

Usage:
    python build_training_set.py
    python build_training_set.py --n-per-group 500 --output custom_train.jsonl
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

# Load config
SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "config.json"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f).get("build_training_set", {})
else:
    CONFIG = {}

SPEECH_DIR = SCRIPT_DIR.parent / CONFIG.get("speech_dir", "persona_2_speech/hatespeech")

MODEL_DIRS = CONFIG.get("model_dirs", [
    "gpt4o-mini",
    "claude-3-haiku",
    "gemini-2.5-flash",
    "deepseek-r1",
    "llama-3.1-8b",
    "mistral-7b",
    "qwen2.5-7b",
    "gemma-2-9b",
])

LABELED_FILENAME = CONFIG.get("labeled_file", "personas_merged_speech_labeled_6judge.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-group", type=int, default=CONFIG.get("n_per_group", 1000),
                        help="Max hate (and non-hate) samples per group")
    parser.add_argument("--output", type=str,
                        default=CONFIG.get("output", "personahate_train_balanced_6judge.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    group_data = defaultdict(lambda: {"hate": [], "non_hate": []})
    total_loaded = 0

    for dirname in MODEL_DIRS:
        path = SPEECH_DIR / dirname / LABELED_FILENAME
        if not path.exists():
            print(f"  Skipping {dirname}: {LABELED_FILENAME} not found")
            continue
        count = 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                group = r.get("group", "unknown")
                if r.get("vote_label") is True:
                    group_data[group]["hate"].append(r)
                else:
                    group_data[group]["non_hate"].append(r)
                count += 1
        total_loaded += count
        print(f"  {dirname}: {count}")

    print(f"\nTotal loaded: {total_loaded} from {len(MODEL_DIRS)} models")
    print(f"Groups: {len(group_data)}\n")

    train = []
    total_hate = 0
    total_nonhate = 0

    print(f"{'Group':<50} {'Hate':>6} {'NonH':>6} {'Total':>7}")
    print("-" * 72)

    for group in sorted(group_data.keys()):
        hate_pool = group_data[group]["hate"]
        nonhate_pool = group_data[group]["non_hate"]

        n_hate = min(len(hate_pool), args.n_per_group)
        n_nonhate = n_hate

        sampled_hate = random.sample(hate_pool, n_hate)
        sampled_nonhate = random.sample(nonhate_pool, n_nonhate)

        for r in sampled_hate:
            r["split_label"] = 1
            train.append(r)
        for r in sampled_nonhate:
            r["split_label"] = 0
            train.append(r)

        total_hate += n_hate
        total_nonhate += n_nonhate
        marker = " *" if n_hate < args.n_per_group else ""
        print(f"{group:<50} {n_hate:>6} {n_nonhate:>6} {n_hate+n_nonhate:>7}{marker}")

    print("-" * 72)
    print(f"{'TOTAL':<50} {total_hate:>6} {total_nonhate:>6} {total_hate+total_nonhate:>7}")
    print(f"\n* = group had fewer than {args.n_per_group} hate samples")

    random.shuffle(train)

    out_dir = SCRIPT_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output
    with open(out_path, 'w') as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\nSaved: {out_path}")
    print(f"Total: {len(train)} ({total_hate} hate + {total_nonhate} non-hate)")


if __name__ == "__main__":
    main()
