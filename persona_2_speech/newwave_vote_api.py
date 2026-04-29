#!/usr/bin/env python3
"""Vote on NewWave data using 4 API judges (no vLLM needed)."""

import json, sys, os, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vote_label_6judge import (
    judge_gpt, judge_claude, judge_gemini, judge_moderation
)

INPUT_DIR = Path(__file__).parent / "hatespeech_newwave"
INPUT_FILE = INPUT_DIR / "all_models_merged.jsonl"
OUTPUT_FILE = INPUT_DIR / "all_models_voted.jsonl"

JUDGES = {
    "gpt-4o-mini": judge_gpt,
    "claude-3-haiku": judge_claude,
    "gemini-2.5-flash": judge_gemini,
    "moderation": judge_moderation,
}


def process_one(record):
    text = record.get("text", "")
    if not text or len(text.strip()) < 10:
        record["vote_label"] = None
        return record

    results = {}
    def call_judge(name):
        return name, JUDGES[name](text)

    with ThreadPoolExecutor(max_workers=len(JUDGES)) as ex:
        for future in as_completed([ex.submit(call_judge, n) for n in JUDGES]):
            name, res = future.result()
            results[name] = res

    votes = []
    for name in JUDGES:
        res = results[name]
        record[f"judge_{name}"] = res["is_hate"]
        record[f"judge_{name}_raw"] = res.get("raw", "")
        votes.append(res["is_hate"])

    valid = [v for v in votes if v is not None]
    hate_count = sum(1 for v in valid if v)
    record["hate_votes"] = hate_count
    record["total_votes"] = len(valid)
    # With 4 judges, majority = 3+
    record["vote_label"] = 1 if hate_count >= 3 else 0 if valid else None

    return record


def main():
    print("Loading NewWave data...")
    records = []
    with open(INPUT_FILE) as f:
        for line in f:
            row = json.loads(line)
            if row.get("is_refused") or row.get("error") or len(row.get("text", "")) < 20:
                continue
            records.append(row)

    print(f"  Records to label: {len(records)}")

    # Resume support
    done_texts = set()
    done_records = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                r = json.loads(line)
                if r.get("vote_label") is not None:
                    done_texts.add(r.get("text", "")[:100])
                    done_records.append(r)
        print(f"  Already labeled: {len(done_records)}")

    todo = [r for r in records if r.get("text", "")[:100] not in done_texts]
    print(f"  To label: {len(todo)}")

    if not todo:
        print("  All done!")
        records = done_records
    else:
        completed = 0
        total = len(todo)
        results = list(done_records)

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = {pool.submit(process_one, r): r for r in todo}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                if completed % 500 == 0:
                    print(f"  Progress: {completed}/{total}")
                    # Intermediate save
                    with open(OUTPUT_FILE, "w") as f:
                        for r in results:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")

        records = results

    # Final save
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
