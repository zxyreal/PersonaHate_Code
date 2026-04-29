#!/usr/bin/env python3
"""
Run 6-judge voting on NewWave generated speech.
Reuses vote_label_6judge.py's judge functions.
"""

import json, sys, os
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vote_label_6judge import (
    JUDGES_API, JUDGES_VLLM, majority_vote, process_record
)

INPUT_DIR = Path(__file__).parent / "hatespeech_newwave"
INPUT_FILE = INPUT_DIR / "all_models_merged.jsonl"
OUTPUT_FILE = INPUT_DIR / "all_models_voted.jsonl"


def main():
    print("Loading generated NewWave speech...")
    records = []
    with open(INPUT_FILE) as f:
        for line in f:
            row = json.loads(line)
            if row.get("is_refused") or len(row.get("text", "")) < 20:
                continue
            records.append(row)

    print(f"  Records to label: {len(records)}")

    # Use the existing vote pipeline
    # It expects records with 'text' field
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("Running 6-judge voting...")
    done = 0
    total = len(records)

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(process_record, (r, list(JUDGES_API.keys()), list(JUDGES_VLLM.keys()))): r
                   for r in records}
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 200 == 0:
                print(f"  Progress: {done}/{total}")

    # Save
    with open(OUTPUT_FILE, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_hate = sum(1 for r in records if r.get("vote_label") == 1)
    n_nonhate = sum(1 for r in records if r.get("vote_label") == 0)
    n_none = sum(1 for r in records if r.get("vote_label") is None)
    print(f"\n  Results: {n_hate} hate, {n_nonhate} non-hate, {n_none} unlabeled")
    print(f"  Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
