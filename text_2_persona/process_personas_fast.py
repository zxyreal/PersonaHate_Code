"""
Fast Persona Processing Pipeline using Faiss

1. Load pre-computed embeddings
2. Fast deduplication using Faiss index
3. Select 1000 most diverse personas using FPS
"""

import json
import numpy as np
import os
from typing import List, Dict
from tqdm import tqdm
from pathlib import Path
import faiss


def load_personas(filepath: str) -> List[Dict]:
    """Load personas from JSONL file."""
    personas = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading personas"):
            personas.append(json.loads(line.strip()))
    return personas


def deduplicate_with_faiss(
    embeddings: np.ndarray,
    threshold: float = 0.9,
    batch_size: int = 10000
) -> np.ndarray:
    """
    Fast deduplication using Faiss.

    Strategy:
    1. Normalize embeddings for cosine similarity
    2. Build Faiss index for inner product (= cosine sim for normalized vectors)
    3. For each point, find neighbors with similarity > threshold
    4. Keep only the first occurrence in each cluster

    Returns indices of items to keep.
    """
    n, d = embeddings.shape
    print(f"Deduplicating {n} items with threshold {threshold} using Faiss...")

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    normalized = normalized.astype(np.float32)

    # Build Faiss index for inner product search
    print("Building Faiss index...")
    index = faiss.IndexFlatIP(d)
    index.add(normalized)

    # Track which items to keep
    keep_mask = np.ones(n, dtype=bool)

    # Process in batches
    print("Finding similar pairs...")
    for start in tqdm(range(0, n, batch_size), desc="Processing batches"):
        end = min(start + batch_size, n)
        batch = normalized[start:end]

        # Find k nearest neighbors (we need enough to catch all similar items)
        # For threshold 0.9, most items won't have many similar neighbors
        k = min(100, n)  # Search top 100 neighbors
        similarities, indices = index.search(batch, k)

        # Mark duplicates
        for i, (sims, idxs) in enumerate(zip(similarities, indices)):
            global_idx = start + i
            if not keep_mask[global_idx]:
                continue

            # Find items similar to this one (excluding self)
            for sim, idx in zip(sims[1:], idxs[1:]):  # Skip first (self)
                if sim > threshold and idx > global_idx and keep_mask[idx]:
                    keep_mask[idx] = False

    kept_indices = np.where(keep_mask)[0]
    removed = n - len(kept_indices)
    print(f"Kept {len(kept_indices)} items ({removed} removed, {removed/n*100:.1f}%)")

    return kept_indices


def fps_large_scale(
    embeddings: np.ndarray,
    budget: int,
    random_state: int = 42,
    batch_size: int = 50000
) -> np.ndarray:
    """
    Memory-efficient FPS for large datasets.

    Instead of computing O(n²) distance matrix, we incrementally
    update minimum distances as we select new points.

    Uses L2 distance on normalized vectors (equivalent to cosine distance).

    Returns indices of selected points.
    """
    np.random.seed(random_state)
    n, d = embeddings.shape

    print(f"FPS: Selecting {budget} points from {n} points...")

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    normalized = normalized.astype(np.float32)

    # Track minimum distance to selected set for each point
    min_distances = np.full(n, np.inf, dtype=np.float32)

    # Start with random point
    selected = [np.random.randint(n)]
    min_distances[selected[0]] = 0

    # Update min distances from first point
    first_point = normalized[selected[0]:selected[0]+1]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = normalized[start:end]
        # L2 distance squared (for normalized vectors: ||a-b||² = 2 - 2*a·b)
        similarities = batch @ first_point.T
        distances = np.sqrt(np.maximum(0, 2 - 2 * similarities.flatten()))
        min_distances[start:end] = np.minimum(min_distances[start:end], distances)

    # Iteratively select farthest point
    for i in tqdm(range(1, budget), desc="FPS selection"):
        # Select point with maximum minimum distance
        new_idx = np.argmax(min_distances)
        selected.append(new_idx)
        min_distances[new_idx] = 0

        # Update min distances from new point
        new_point = normalized[new_idx:new_idx+1]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = normalized[start:end]
            similarities = batch @ new_point.T
            distances = np.sqrt(np.maximum(0, 2 - 2 * similarities.flatten()))
            min_distances[start:end] = np.minimum(min_distances[start:end], distances)

    return np.array(selected)


def save_personas(personas: List[Dict], indices: np.ndarray, filepath: str):
    """Save selected personas to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for idx in tqdm(indices, desc=f"Saving to {Path(filepath).name}"):
            f.write(json.dumps(personas[idx], ensure_ascii=False) + '\n')
    print(f"Saved {len(indices)} personas to {filepath}")


def main():
    # Paths
    base_dir = Path("/Users/xinyu/Downloads/DA-FPS")
    input_path = base_dir / "persona_1000000.jsonl"
    embeddings_cache = base_dir / "embeddings_cache.npy"
    dedup_output = base_dir / "persona_deduplicated.jsonl"
    final_output = base_dir / "persona_selected_1000.jsonl"

    print("=" * 60)
    print("Fast Persona Processing Pipeline")
    print("=" * 60)

    # Check if embeddings exist
    if not embeddings_cache.exists():
        print(f"Error: Embeddings cache not found at {embeddings_cache}")
        print("Please run process_personas.py first to compute embeddings.")
        return

    # Step 1: Load personas
    print("\n[Step 1] Loading personas...")
    personas = load_personas(str(input_path))
    print(f"Loaded {len(personas)} personas")

    # Step 2: Load embeddings
    print("\n[Step 2] Loading cached embeddings...")
    embeddings = np.load(str(embeddings_cache))
    print(f"Embeddings shape: {embeddings.shape}")

    # Step 3: Fast deduplication
    print(f"\n[Step 3] Fast deduplication with Faiss (threshold=0.9)...")
    kept_indices = deduplicate_with_faiss(embeddings, threshold=0.9)

    # Save deduplicated personas
    save_personas(personas, kept_indices, str(dedup_output))

    # Step 4: Select using FPS (optimized for large scale)
    print(f"\n[Step 4] Selecting 1000 most diverse personas using FPS...")
    dedup_embeddings = embeddings[kept_indices]

    # Use memory-efficient FPS implementation
    fps_indices_in_dedup = fps_large_scale(dedup_embeddings, budget=1000, random_state=42)

    # Map back to original indices
    final_indices = kept_indices[fps_indices_in_dedup]

    # Save final selection
    save_personas(personas, final_indices, str(final_output))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Original personas:      {len(personas)}")
    print(f"After deduplication:    {len(kept_indices)}")
    print(f"Final selection:        1000")
    print(f"\nOutput files:")
    print(f"  - Deduplicated: {dedup_output}")
    print(f"  - Final 1000:   {final_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
