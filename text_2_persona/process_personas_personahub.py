"""
Persona Processing Pipeline

1. Compute embeddings using text-embedding-3-small
2. Remove semantically similar personas (cosine similarity > 0.9)
3. Select 1000 most diverse personas using FPS

Usage:
    python process_personas.py --api_key YOUR_OPENAI_API_KEY

Or set environment variable:
    export OPENAI_API_KEY=your_key
    python process_personas.py
"""

import json
import numpy as np
import os
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm
import time
from pathlib import Path

# Check for dependencies
try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file
except ImportError:
    pass  # dotenv is optional

from da_fps import FPS


def load_personas(filepath: str) -> List[Dict]:
    """Load personas from JSONL file."""
    personas = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading personas"):
            personas.append(json.loads(line.strip()))
    return personas


def compute_embeddings_batch(
    client: OpenAI,
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 500  # Reduced to stay under 300k token limit
) -> np.ndarray:
    """
    Compute embeddings in batches.

    text-embedding-3-small has max 8191 tokens per text.
    API allows up to 2048 texts per batch.
    """
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch = texts[i:i + batch_size]

        # Retry logic for API errors
        max_retries = 5
        for retry in range(max_retries):
            try:
                response = client.embeddings.create(
                    model=model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                if retry < max_retries - 1:
                    wait_time = 2 ** retry
                    print(f"\nAPI error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e

        # Rate limiting: be nice to the API
        if i + batch_size < len(texts):
            time.sleep(0.1)

    return np.array(all_embeddings)


def deduplicate_by_similarity(
    embeddings: np.ndarray,
    threshold: float = 0.9,
    batch_size: int = 5000
) -> np.ndarray:
    """
    Remove items with cosine similarity > threshold.

    Uses a greedy approach: keep first occurrence, remove similar ones.
    Processes in batches to handle large datasets.

    Returns indices of items to keep.
    """
    n = len(embeddings)

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings / norms

    # Track which items to keep
    keep_mask = np.ones(n, dtype=bool)

    print(f"Deduplicating {n} items with threshold {threshold}...")

    # Process in batches to avoid memory issues
    for i in tqdm(range(n), desc="Deduplicating"):
        if not keep_mask[i]:
            continue

        # Compare with remaining items
        # Only check items after current one (items before are already processed)
        remaining_indices = np.where(keep_mask)[0]
        remaining_indices = remaining_indices[remaining_indices > i]

        if len(remaining_indices) == 0:
            continue

        # Compute similarities with remaining items
        similarities = normalized[remaining_indices] @ normalized[i]

        # Mark similar items for removal
        too_similar = similarities > threshold
        keep_mask[remaining_indices[too_similar]] = False

    kept_indices = np.where(keep_mask)[0]
    print(f"Kept {len(kept_indices)} items after deduplication ({n - len(kept_indices)} removed)")

    return kept_indices


def deduplicate_fast(
    embeddings: np.ndarray,
    threshold: float = 0.9,
    sample_size: int = 10000
) -> np.ndarray:
    """
    Faster deduplication using random sampling for initial filtering.

    For very large datasets (>100k), this is more practical.
    """
    n = len(embeddings)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms

    keep_mask = np.ones(n, dtype=bool)

    print(f"Fast deduplicating {n} items with threshold {threshold}...")

    # Process each item
    processed = 0
    for i in tqdm(range(n), desc="Deduplicating"):
        if not keep_mask[i]:
            continue

        processed += 1

        # Only compare with items after current one
        future_mask = keep_mask.copy()
        future_mask[:i+1] = False
        future_indices = np.where(future_mask)[0]

        if len(future_indices) == 0:
            continue

        # Compute similarities
        similarities = normalized[future_indices] @ normalized[i]

        # Mark similar items for removal
        too_similar = similarities > threshold
        keep_mask[future_indices[too_similar]] = False

    kept_indices = np.where(keep_mask)[0]
    removed = n - len(kept_indices)
    print(f"Kept {len(kept_indices)} items ({removed} removed, {removed/n*100:.1f}%)")

    return kept_indices


def save_personas(personas: List[Dict], indices: np.ndarray, filepath: str):
    """Save selected personas to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for idx in tqdm(indices, desc=f"Saving to {filepath}"):
            f.write(json.dumps(personas[idx], ensure_ascii=False) + '\n')
    print(f"Saved {len(indices)} personas to {filepath}")


def main():
    # Load config
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f).get("process_personas_personahub", {})
    else:
        config = {}

    parser = argparse.ArgumentParser(description="Process personas: embed, deduplicate, select")
    parser.add_argument("--input", default=config.get("input", "data/persona_personahub_200k.jsonl"), help="Input JSONL file")
    parser.add_argument("--api_key", default=None, help="OpenAI API key")
    parser.add_argument("--threshold", type=float, default=config.get("threshold", 0.9), help="Similarity threshold")
    parser.add_argument("--n_select", type=int, default=config.get("n_select", 1000), help="Number of personas to select")
    parser.add_argument("--embeddings_cache", default="embeddings_cache.npy", help="Cache file for embeddings")
    parser.add_argument("--skip_embedding", action="store_true", help="Skip embedding computation (use cache)")
    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.skip_embedding:
        print("Error: Please provide OpenAI API key via --api_key or OPENAI_API_KEY env var")
        return

    # Paths
    input_path = Path(args.input)
    if not input_path.exists():
        input_path = Path(__file__).parent / args.input

    output_dir = Path(__file__).parent / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    dedup_output = output_dir / "personahub_deduplicated.jsonl"
    final_output = output_dir / f"personahub_selected_{args.n_select}.jsonl"
    embeddings_cache = output_dir / "personahub_embeddings_cache.npy"

    print("=" * 60)
    print("Persona Processing Pipeline")
    print("=" * 60)

    # Step 1: Load personas
    print("\n[Step 1] Loading personas...")
    personas = load_personas(str(input_path))
    print(f"Loaded {len(personas)} personas")

    # Extract persona texts
    texts = [p["persona"] for p in personas]

    # Step 2: Compute or load embeddings
    if args.skip_embedding and embeddings_cache.exists():
        print(f"\n[Step 2] Loading cached embeddings from {embeddings_cache}...")
        embeddings = np.load(str(embeddings_cache))
    else:
        print("\n[Step 2] Computing embeddings with text-embedding-3-small...")
        client = OpenAI(api_key=api_key)
        embeddings = compute_embeddings_batch(client, texts)

        # Cache embeddings
        np.save(str(embeddings_cache), embeddings)
        print(f"Cached embeddings to {embeddings_cache}")

    print(f"Embeddings shape: {embeddings.shape}")

    # Step 3: Deduplicate
    print(f"\n[Step 3] Removing personas with cosine similarity > {args.threshold}...")
    kept_indices = deduplicate_fast(embeddings, threshold=args.threshold)

    # Save deduplicated personas
    save_personas(personas, kept_indices, str(dedup_output))

    # Step 4: Select using FPS
    print(f"\n[Step 4] Selecting {args.n_select} most diverse personas using FPS...")
    dedup_embeddings = embeddings[kept_indices]

    fps = FPS(random_state=42)
    fps.fit(dedup_embeddings, budget=args.n_select, verbose=True)

    # Map back to original indices
    fps_indices_in_dedup = fps.get_selected_indices()
    final_indices = kept_indices[fps_indices_in_dedup]

    # Save final selection
    save_personas(personas, final_indices, str(final_output))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Original personas:      {len(personas)}")
    print(f"After deduplication:    {len(kept_indices)}")
    print(f"Final selection:        {args.n_select}")
    print(f"\nOutput files:")
    print(f"  - Deduplicated: {dedup_output}")
    print(f"  - Final 1000:   {final_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
