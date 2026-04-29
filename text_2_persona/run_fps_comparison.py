#!/usr/bin/env python3
"""
Compare FPS vs DA-FPS vs Random on merged persona pool.
Scalable version: uses batch-incremental distance updates (O(n*budget) time,
O(n) space) instead of O(n²) distance matrix precomputation.

For DA-FPS, KNN distances are precomputed once with Faiss and density weights
are approximated from the KNN graph.
"""

import json
import numpy as np
from pathlib import Path
import faiss
import time

MERGED_PATH = Path(__file__).parent / "persona_merged.jsonl"
OUTPUT_DIR = Path(__file__).parent
BUDGET = 1000
RANDOM_STATE = 42


def load_personas(path):
    personas = []
    with open(path) as f:
        for line in f:
            personas.append(json.loads(line.strip()))
    return personas


def fps_fast(normalized, budget, random_state=42, batch_size=50000):
    """
    Memory-efficient FPS using batch-incremental distance updates.
    O(n * budget) time, O(n) space.
    Uses cosine distance (1 - dot product) on pre-normalized vectors.
    """
    np.random.seed(random_state)
    n, d = normalized.shape
    X = normalized.astype(np.float32)

    min_distances = np.full(n, np.inf, dtype=np.float32)
    selected = [np.random.randint(n)]
    min_distances[selected[0]] = 0

    # Update from first selected point
    first = X[selected[0]:selected[0] + 1]
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        sims = X[s:e] @ first.T
        dists = 1.0 - sims.flatten()
        min_distances[s:e] = np.minimum(min_distances[s:e], dists)

    for i in range(1, budget):
        if i % 100 == 0:
            print(f"    FPS: {i}/{budget}")
        new_idx = int(np.argmax(min_distances))
        selected.append(new_idx)
        min_distances[new_idx] = 0

        new_point = X[new_idx:new_idx + 1]
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            sims = X[s:e] @ new_point.T
            dists = 1.0 - sims.flatten()
            min_distances[s:e] = np.minimum(min_distances[s:e], dists)

    return np.array(selected)


def dafps_fast(normalized, budget, k=10, warmup_ratio=0.1, random_state=42,
               k_large=100, batch_size=50000):
    """
    Scalable DA-FPS using Faiss for KNN precomputation.

    1. Precompute KNN graph with Faiss (one-time O(n*k_large) search)
    2. Warmup: standard FPS for first u points
    3. DA-FPS phase: approximate density weight from KNN graph
       omega(i) = count of precomputed k_large-NN within adaptive radius r_k(i)
    """
    np.random.seed(random_state)
    n, d = normalized.shape
    X = normalized.astype(np.float32)
    epsilon_x = 1e-6
    u = max(1, int(warmup_ratio * budget))
    k_eff = min(k, n - 1)
    k_search = min(k_large, n - 1)

    # Precompute KNN with Faiss
    print(f"    DA-FPS: Precomputing {k_search}-NN with Faiss...")
    t_knn = time.time()
    index = faiss.IndexFlatIP(d)
    index.add(X)
    # Search k_search+1 to skip self-match
    knn_sims, _ = index.search(X, k_search + 1)
    # Convert to cosine distances, skip self (column 0)
    knn_dists = 1.0 - knn_sims[:, 1:]  # shape (n, k_search)
    rho_k = knn_dists[:, k_eff - 1].copy()  # k-th NN cosine distance
    print(f"    DA-FPS: KNN done in {time.time() - t_knn:.1f}s")

    # Initialize
    min_distances = np.full(n, np.inf, dtype=np.float32)
    selected = [np.random.randint(n)]
    min_distances[selected[0]] = 0

    # Update from first selected point
    first = X[selected[0]:selected[0] + 1]
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        sims = X[s:e] @ first.T
        dists = 1.0 - sims.flatten()
        min_distances[s:e] = np.minimum(min_distances[s:e], dists)

    for i in range(1, budget):
        if i % 100 == 0:
            print(f"    DA-FPS: {i}/{budget}")

        if i < u:
            # Warmup phase: standard FPS
            new_idx = int(np.argmax(min_distances))
        else:
            # DA-FPS phase: weighted selection
            t = len(selected)
            # Adaptive radius: r_k(x) = min(min_dist(x) + eps/|L|, rho_k(x))
            r_k = np.minimum(min_distances + epsilon_x / t, rho_k)
            # Approximate weight: count of KNN within adaptive radius
            # knn_dists[:, j] is distance to (j+1)-th nearest neighbor
            weights = np.sum(knn_dists <= r_k[:, None], axis=1).astype(np.float32)
            weights += 1.0  # include self
            scores = weights * min_distances
            new_idx = int(np.argmax(scores))

        selected.append(new_idx)
        min_distances[new_idx] = 0

        new_point = X[new_idx:new_idx + 1]
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            sims = X[s:e] @ new_point.T
            dists = 1.0 - sims.flatten()
            min_distances[s:e] = np.minimum(min_distances[s:e], dists)

    return np.array(selected)


def compute_coverage_metrics(embeddings, selected_indices, all_embeddings):
    """Compute fill distance and mean nearest distance (cosine)."""
    selected_emb = embeddings[selected_indices]

    sel_norm = selected_emb / np.linalg.norm(selected_emb, axis=1, keepdims=True)
    all_norm = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)

    batch_size = 10000
    max_min_dist = 0
    sum_min_dist = 0
    n = len(all_norm)

    for i in range(0, n, batch_size):
        batch = all_norm[i:i + batch_size]
        sims = batch @ sel_norm.T  # (batch, budget)
        min_dists = 1 - sims.max(axis=1)  # cosine distance to nearest selected
        max_min_dist = max(max_min_dist, min_dists.max())
        sum_min_dist += min_dists.sum()

    return {
        "fill_distance": float(max_min_dist),
        "mean_nearest_distance": float(sum_min_dist / n),
    }


def main():
    print("[1] Loading personas...")
    personas = load_personas(MERGED_PATH)
    print(f"  Total: {len(personas)}")

    sources = {}
    for p in personas:
        s = p.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1
    print(f"  Sources: {sources}")

    # Load embeddings (cached from previous run or encode now)
    cache_path = OUTPUT_DIR / "merged_embeddings.npy"
    if cache_path.exists():
        print(f"\n[2] Loading cached embeddings from {cache_path}...")
        embeddings = np.load(str(cache_path))
    else:
        print("\n[2] Encoding with sentence-transformers...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode([p["persona"] for p in personas],
                                  show_progress_bar=True, batch_size=256,
                                  convert_to_numpy=True)
        np.save(str(cache_path), embeddings)
        print(f"  Cached to {cache_path}")

    print(f"  Embeddings shape: {embeddings.shape}")

    # Normalize for cosine distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms

    # ── FPS ──
    print(f"\n[3] Running FPS (budget={BUDGET})...")
    t0 = time.time()
    fps_indices = fps_fast(normalized, BUDGET, RANDOM_STATE)
    fps_time = time.time() - t0
    print(f"  FPS done in {fps_time:.1f}s")

    fps_sources = {}
    for idx in fps_indices:
        s = personas[idx].get("source", "unknown")
        fps_sources[s] = fps_sources.get(s, 0) + 1
    print(f"  FPS source balance: {fps_sources}")

    fps_metrics = compute_coverage_metrics(embeddings, fps_indices, embeddings)
    print(f"  FPS fill distance: {fps_metrics['fill_distance']:.4f}")
    print(f"  FPS mean nearest dist: {fps_metrics['mean_nearest_distance']:.4f}")

    # ── DA-FPS ──
    print(f"\n[4] Running DA-FPS (budget={BUDGET})...")
    t0 = time.time()
    dafps_indices = dafps_fast(normalized, BUDGET, k=10, warmup_ratio=0.1,
                               random_state=RANDOM_STATE)
    dafps_time = time.time() - t0
    print(f"  DA-FPS done in {dafps_time:.1f}s")

    dafps_sources = {}
    for idx in dafps_indices:
        s = personas[idx].get("source", "unknown")
        dafps_sources[s] = dafps_sources.get(s, 0) + 1
    print(f"  DA-FPS source balance: {dafps_sources}")

    dafps_metrics = compute_coverage_metrics(embeddings, dafps_indices, embeddings)
    print(f"  DA-FPS fill distance: {dafps_metrics['fill_distance']:.4f}")
    print(f"  DA-FPS mean nearest dist: {dafps_metrics['mean_nearest_distance']:.4f}")

    # ── Random baseline ──
    print(f"\n[5] Running Random sampling (budget={BUDGET})...")
    rng = np.random.RandomState(RANDOM_STATE)
    random_indices = rng.choice(len(personas), BUDGET, replace=False)

    random_sources = {}
    for idx in random_indices:
        s = personas[idx].get("source", "unknown")
        random_sources[s] = random_sources.get(s, 0) + 1
    print(f"  Random source balance: {random_sources}")

    random_metrics = compute_coverage_metrics(embeddings, random_indices, embeddings)
    print(f"  Random fill distance: {random_metrics['fill_distance']:.4f}")
    print(f"  Random mean nearest dist: {random_metrics['mean_nearest_distance']:.4f}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Method':<12} {'Fill Dist':>10} {'Mean NN Dist':>13} {'Time':>8} {'4chan':>6} {'hub':>6}")
    print("-" * 60)
    print(f"{'Random':<12} {random_metrics['fill_distance']:>10.4f} "
          f"{random_metrics['mean_nearest_distance']:>13.4f} {'--':>8} "
          f"{random_sources.get('4chan', 0):>6} {random_sources.get('personahub', 0):>6}")
    print(f"{'FPS':<12} {fps_metrics['fill_distance']:>10.4f} "
          f"{fps_metrics['mean_nearest_distance']:>13.4f} {fps_time:>7.1f}s "
          f"{fps_sources.get('4chan', 0):>6} {fps_sources.get('personahub', 0):>6}")
    print(f"{'DA-FPS':<12} {dafps_metrics['fill_distance']:>10.4f} "
          f"{dafps_metrics['mean_nearest_distance']:>13.4f} {dafps_time:>7.1f}s "
          f"{dafps_sources.get('4chan', 0):>6} {dafps_sources.get('personahub', 0):>6}")

    # Save selected personas
    for name, indices in [("fps", fps_indices), ("dafps", dafps_indices)]:
        out_path = OUTPUT_DIR / f"persona_selected_1000_{name}.jsonl"
        with open(out_path, 'w') as f:
            for i, idx in enumerate(indices):
                record = {"id": i, "persona": personas[idx]["persona"],
                          "source": personas[idx]["source"]}
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"\nSaved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
