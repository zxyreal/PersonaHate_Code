"""
Apply DA-FPS to PersonaHub dataset using Faiss for fast processing.
"""

import numpy as np
import json
import faiss
from tqdm import tqdm

def deduplicate_with_faiss(embeddings, threshold=0.9):
    """
    Fast deduplication using Faiss index.
    Remove items with cosine similarity > threshold.
    """
    print(f"Deduplicating {len(embeddings)} embeddings with threshold {threshold}...")

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = (embeddings / norms).astype('float32')

    # Build Faiss index
    d = normalized.shape[1]
    index = faiss.IndexFlatIP(d)  # Inner product = cosine for normalized vectors

    keep_indices = []

    for i in tqdm(range(len(normalized)), desc="Deduplicating"):
        if i == 0:
            keep_indices.append(i)
            index.add(normalized[i:i+1])
            continue

        # Search for similar items in already-kept set
        D, I = index.search(normalized[i:i+1], 1)

        if D[0][0] < threshold:  # Not similar to any kept item
            keep_indices.append(i)
            index.add(normalized[i:i+1])

    print(f"Kept {len(keep_indices)} / {len(embeddings)} after deduplication")
    return np.array(keep_indices)

def fps_selection(embeddings, budget, seed=42):
    """
    Farthest Point Sampling for diverse selection.
    """
    print(f"Running FPS to select {budget} diverse points...")
    np.random.seed(seed)

    n = len(embeddings)
    if budget >= n:
        return np.arange(n)

    # Normalize for cosine distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms

    # Initialize with random point
    selected = [np.random.randint(n)]

    # Track minimum distance to selected set for each point
    min_distances = np.ones(n) * np.inf

    for _ in tqdm(range(budget - 1), desc="FPS selection"):
        # Update distances based on last selected point
        last_selected = selected[-1]
        # Cosine distance = 1 - cosine_similarity
        similarities = normalized @ normalized[last_selected]
        distances = 1 - similarities
        min_distances = np.minimum(min_distances, distances)

        # Mark already selected as -inf
        min_distances[selected] = -np.inf

        # Select farthest point
        next_idx = np.argmax(min_distances)
        selected.append(next_idx)

    return np.array(selected)

def main():
    # Load embeddings
    print("Loading embeddings...")
    embeddings = np.load("personahub_embeddings.npy")
    print(f"Loaded embeddings shape: {embeddings.shape}")

    # Load personas
    print("Loading personas...")
    personas = []
    with open("personahub_200k.jsonl", "r") as f:
        for line in f:
            personas.append(json.loads(line))
    print(f"Loaded {len(personas)} personas")

    # Step 1: Deduplicate
    keep_indices = deduplicate_with_faiss(embeddings, threshold=0.9)
    deduplicated_embeddings = embeddings[keep_indices]
    deduplicated_personas = [personas[i] for i in keep_indices]
    print(f"After deduplication: {len(deduplicated_embeddings)} personas")

    # Step 2: FPS selection
    budget = 1000
    if len(deduplicated_embeddings) <= budget:
        selected_indices = np.arange(len(deduplicated_embeddings))
    else:
        selected_indices = fps_selection(deduplicated_embeddings, budget)

    # Get final personas
    final_personas = [deduplicated_personas[i] for i in selected_indices]
    print(f"Selected {len(final_personas)} diverse personas")

    # Save results
    output_file = "personahub_selected_1000.jsonl"
    with open(output_file, "w") as f:
        for i, p in enumerate(final_personas):
            record = {"id": i, "persona": p["persona"]}
            f.write(json.dumps(record) + "\n")
    print(f"Saved to {output_file}")

    # Also save the indices for reference
    np.save("personahub_selected_indices.npy", keep_indices[selected_indices])
    print("Saved indices to personahub_selected_indices.npy")

if __name__ == "__main__":
    main()
