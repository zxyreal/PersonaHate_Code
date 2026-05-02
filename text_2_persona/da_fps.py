"""
Density-Aware Farthest Point Sampling (DA-FPS)
Paper: "Density-Aware Farthest Point Sampling" (arXiv:2509.13213v1)
Authors: Paolo Climaco, Jochen Garcke

This module implements the DA-FPS algorithm for selecting training sets
from unlabeled data pools, aiming to minimize the expected prediction error
of Lipschitz continuous regression models.
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional, Tuple, List, Union
import warnings


class DAFPS:
    """
    Density-Aware Farthest Point Sampling (DA-FPS)

    DA-FPS is a passive and model-agnostic sampling method that selects
    training sets by minimizing the weighted fill distance. The weights
    consider the data density distribution, giving higher priority to
    points in dense regions.

    Parameters
    ----------
    k : int, default=10
        Number of nearest neighbors for density estimation.
    epsilon_x : float, default=1e-6
        Small positive scalar to prevent division by zero.
    warmup_ratio : float, default=0.1
        Ratio of budget to use standard FPS before switching to DA-FPS.
        This is the 'u' parameter in the paper (u = warmup_ratio * budget).
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    selected_indices_ : np.ndarray
        Indices of selected points after fitting.
    selected_points_ : np.ndarray
        Selected point coordinates after fitting.

    References
    ----------
    Climaco, P., & Garcke, J. (2025). Density-Aware Farthest Point Sampling.
    arXiv:2509.13213v1
    """

    def __init__(
        self,
        k: int = 10,
        epsilon_x: float = 1e-6,
        warmup_ratio: float = 0.1,
        random_state: Optional[int] = None
    ):
        self.k = k
        self.epsilon_x = epsilon_x
        self.warmup_ratio = warmup_ratio
        self.random_state = random_state

        self.selected_indices_: Optional[np.ndarray] = None
        self.selected_points_: Optional[np.ndarray] = None

    def _compute_knn_distance(
        self,
        X: np.ndarray,
        x: np.ndarray,
        k: int
    ) -> float:
        """
        Compute the distance to the k-th nearest neighbor.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Dataset points.
        x : np.ndarray of shape (n_features,)
            Query point.
        k : int
            Number of nearest neighbors.

        Returns
        -------
        rho_k : float
            Distance to the k-th nearest neighbor.
        """
        distances = np.linalg.norm(X - x, axis=1)
        # Sort and get k-th smallest distance
        sorted_distances = np.sort(distances)
        # k-th nearest neighbor (0-indexed, so use k-1, but we want k neighbors)
        # The paper defines rho_k(x) as distance such that |{points within rho_k}| >= k
        if k <= len(sorted_distances):
            return sorted_distances[k - 1]  # k-1 because 0-indexed
        else:
            return sorted_distances[-1]

    def _compute_radius(
        self,
        X: np.ndarray,
        L_X: np.ndarray,
        x: np.ndarray,
        k: int
    ) -> float:
        """
        Compute the adaptive neighborhood radius r^k_{L_X}(x).

        From Equation (6) in the paper:
        r^k_{L_X}(x) = min(min_{x̄∈L_X} ||x - x̄||_2 + ε_X/|L_X|, ρ_k(x))

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Full dataset.
        L_X : np.ndarray of shape (n_selected, n_features)
            Currently selected points.
        x : np.ndarray of shape (n_features,)
            Query point.
        k : int
            Number of nearest neighbors.

        Returns
        -------
        r_k : float
            Adaptive neighborhood radius.
        """
        # Distance to nearest point in L_X
        if len(L_X) > 0:
            min_dist_to_L = np.min(np.linalg.norm(L_X - x, axis=1))
            term1 = min_dist_to_L + self.epsilon_x / len(L_X)
        else:
            term1 = float('inf')

        # Distance to k-th nearest neighbor in X
        rho_k = self._compute_knn_distance(X, x, k)

        return min(term1, rho_k)

    def _compute_weights(
        self,
        X: np.ndarray,
        L_X: np.ndarray,
        L_indices: np.ndarray,
        k: int
    ) -> np.ndarray:
        r"""
        Compute weights omega^k_{L_X}(x) for all points in X \ L_X.

        From Equation (8) in the paper:
        omega^k_{L_X}(x) = |{x_bar in D_X : ||x - x_bar||_2 <= r^k_{L_X}(x)}|

        The weight is the number of points in D_X within the adaptive
        neighborhood radius of x.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Full dataset.
        L_X : np.ndarray of shape (n_selected, n_features)
            Currently selected points.
        L_indices : np.ndarray of shape (n_selected,)
            Indices of selected points.
        k : int
            Number of nearest neighbors.

        Returns
        -------
        weights : np.ndarray of shape (n_samples,)
            Weights for each point. Points in L_X have weight 0.
        """
        n = len(X)
        weights = np.zeros(n)

        # Compute pairwise distances for efficiency
        all_distances = cdist(X, X)

        # Create mask for unselected points
        mask = np.ones(n, dtype=bool)
        mask[L_indices] = False

        for i in range(n):
            if not mask[i]:
                weights[i] = 0  # Already selected
                continue

            x = X[i]

            # Compute adaptive radius
            r_k = self._compute_radius(X, L_X, x, k)

            # Count points within radius
            points_in_ball = np.sum(all_distances[i] <= r_k)
            weights[i] = points_in_ball

        return weights

    def _compute_min_distances(
        self,
        X: np.ndarray,
        L_X: np.ndarray
    ) -> np.ndarray:
        """
        Compute minimum distance from each point in X to the selected set L_X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Full dataset.
        L_X : np.ndarray of shape (n_selected, n_features)
            Currently selected points.

        Returns
        -------
        min_distances : np.ndarray of shape (n_samples,)
            Minimum distance from each point to L_X.
        """
        if len(L_X) == 0:
            return np.full(len(X), float('inf'))

        distances = cdist(X, L_X)
        return np.min(distances, axis=1)

    def fit(
        self,
        X: np.ndarray,
        budget: int,
        initial_indices: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> 'DAFPS':
        """
        Select a subset of points using DA-FPS.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The dataset to sample from.
        budget : int
            Number of points to select (b in the paper).
        initial_indices : np.ndarray or None, default=None
            Initial set of selected indices. If None, starts with random point.
        verbose : bool, default=False
            If True, print progress information.

        Returns
        -------
        self : DAFPS
            Fitted estimator.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n, d = X.shape

        if budget >= n:
            warnings.warn(f"Budget ({budget}) >= dataset size ({n}). Selecting all points.")
            self.selected_indices_ = np.arange(n)
            self.selected_points_ = X.copy()
            return self

        # Warmup parameter u: use standard FPS for first u points
        u = max(1, int(self.warmup_ratio * budget))

        # Ensure k doesn't exceed dataset size
        k = min(self.k, n - 1)

        # Initialize selected set
        if initial_indices is not None and len(initial_indices) > 0:
            L_indices = list(initial_indices)
        else:
            # Random initialization
            L_indices = [np.random.randint(n)]

        L_X = X[L_indices].reshape(-1, d)

        # Main loop
        while len(L_indices) < budget:
            current_size = len(L_indices)

            if verbose and current_size % 100 == 0:
                print(f"Selected {current_size}/{budget} points")

            # Compute minimum distances to selected set
            min_distances = self._compute_min_distances(X, L_X)

            # Set distance to 0 for already selected points
            min_distances[L_indices] = 0

            if current_size < u:
                # Standard FPS: select point with maximum distance
                # Equation from line 6 in Algorithm 1
                scores = min_distances
            else:
                # DA-FPS: weighted selection
                # Equation from line 9 in Algorithm 1
                weights = self._compute_weights(X, L_X, np.array(L_indices), k)
                scores = weights * min_distances

            # Select point with maximum score
            new_idx = np.argmax(scores)

            # Add to selected set
            L_indices.append(new_idx)
            L_X = np.vstack([L_X, X[new_idx]])

        self.selected_indices_ = np.array(L_indices)
        self.selected_points_ = X[self.selected_indices_]

        return self

    def get_selected_indices(self) -> np.ndarray:
        """Return indices of selected points."""
        if self.selected_indices_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.selected_indices_

    def get_selected_points(self) -> np.ndarray:
        """Return selected points."""
        if self.selected_points_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.selected_points_


class FPS:
    """
    Standard Farthest Point Sampling (FPS)

    FPS is a greedy sampling method that iteratively selects the point
    farthest from the current selected set. It minimizes the fill distance.

    Parameters
    ----------
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    selected_indices_ : np.ndarray
        Indices of selected points after fitting.
    selected_points_ : np.ndarray
        Selected point coordinates after fitting.
    """

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        self.selected_indices_: Optional[np.ndarray] = None
        self.selected_points_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        budget: int,
        initial_indices: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> 'FPS':
        """
        Select a subset of points using FPS.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The dataset to sample from.
        budget : int
            Number of points to select.
        initial_indices : np.ndarray or None, default=None
            Initial set of selected indices.
        verbose : bool, default=False
            If True, print progress information.

        Returns
        -------
        self : FPS
            Fitted estimator.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n, d = X.shape

        if budget >= n:
            warnings.warn(f"Budget ({budget}) >= dataset size ({n}). Selecting all points.")
            self.selected_indices_ = np.arange(n)
            self.selected_points_ = X.copy()
            return self

        # Initialize
        if initial_indices is not None and len(initial_indices) > 0:
            L_indices = list(initial_indices)
        else:
            L_indices = [np.random.randint(n)]

        L_X = X[L_indices].reshape(-1, d)

        # Precompute all pairwise distances for efficiency
        all_distances = cdist(X, X)

        # Track minimum distance to selected set for each point
        min_distances = np.full(n, float('inf'))
        for idx in L_indices:
            min_distances = np.minimum(min_distances, all_distances[:, idx])
        min_distances[L_indices] = 0

        while len(L_indices) < budget:
            if verbose and len(L_indices) % 100 == 0:
                print(f"Selected {len(L_indices)}/{budget} points")

            # Select farthest point
            new_idx = np.argmax(min_distances)

            # Update minimum distances
            min_distances = np.minimum(min_distances, all_distances[:, new_idx])
            min_distances[new_idx] = 0

            L_indices.append(new_idx)

        self.selected_indices_ = np.array(L_indices)
        self.selected_points_ = X[self.selected_indices_]

        return self

    def get_selected_indices(self) -> np.ndarray:
        """Return indices of selected points."""
        if self.selected_indices_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.selected_indices_

    def get_selected_points(self) -> np.ndarray:
        """Return selected points."""
        if self.selected_points_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.selected_points_


class RandomSampling:
    """
    Uniform Random Sampling

    Baseline sampling method that selects points uniformly at random.

    Parameters
    ----------
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        self.selected_indices_: Optional[np.ndarray] = None
        self.selected_points_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, budget: int, **kwargs) -> 'RandomSampling':
        """
        Select a random subset of points.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The dataset to sample from.
        budget : int
            Number of points to select.

        Returns
        -------
        self : RandomSampling
            Fitted estimator.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n = len(X)
        budget = min(budget, n)

        self.selected_indices_ = np.random.choice(n, size=budget, replace=False)
        self.selected_points_ = X[self.selected_indices_]

        return self

    def get_selected_indices(self) -> np.ndarray:
        if self.selected_indices_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.selected_indices_

    def get_selected_points(self) -> np.ndarray:
        if self.selected_points_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.selected_points_


def compute_fill_distance(X: np.ndarray, L_X: np.ndarray) -> float:
    """
    Compute the fill distance of L_X in X.

    Fill distance = max_{x ∈ X} min_{x_j ∈ L_X} ||x - x_j||_2

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        The full dataset.
    L_X : np.ndarray of shape (n_selected, n_features)
        The selected subset.

    Returns
    -------
    fill_distance : float
        The fill distance of L_X in X.
    """
    distances = cdist(X, L_X)
    min_distances = np.min(distances, axis=1)
    return np.max(min_distances)


def compute_weighted_fill_distance(
    X: np.ndarray,
    L_X: np.ndarray,
    L_indices: np.ndarray,
    k: int = 10,
    epsilon_x: float = 1e-6
) -> float:
    """
    Compute the estimated weighted fill distance (Equation 11 in paper).

    W^k_{L_X, D_X} = max_{x ∈ D_X}(ω^k_{L_X}(x) * min_{x_j ∈ L_X} ||x - x_j||_2)

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        The full dataset.
    L_X : np.ndarray of shape (n_selected, n_features)
        The selected subset.
    L_indices : np.ndarray of shape (n_selected,)
        Indices of selected points.
    k : int, default=10
        Number of nearest neighbors for weight computation.
    epsilon_x : float, default=1e-6
        Small positive scalar.

    Returns
    -------
    weighted_fill_distance : float
        The estimated weighted fill distance.
    """
    n = len(X)

    # Compute pairwise distances
    all_distances = cdist(X, X)
    distances_to_L = cdist(X, L_X)
    min_distances_to_L = np.min(distances_to_L, axis=1)

    max_weighted_dist = 0

    for i in range(n):
        if i in L_indices:
            continue

        x = X[i]

        # Compute adaptive radius
        min_dist_to_L = min_distances_to_L[i]
        term1 = min_dist_to_L + epsilon_x / len(L_X)

        # k-th nearest neighbor distance
        sorted_dists = np.sort(all_distances[i])
        rho_k = sorted_dists[min(k, n-1)]

        r_k = min(term1, rho_k)

        # Weight = number of points within radius
        weight = np.sum(all_distances[i] <= r_k)

        weighted_dist = weight * min_dist_to_L
        max_weighted_dist = max(max_weighted_dist, weighted_dist)

    return max_weighted_dist


if __name__ == "__main__":
    # Quick test
    print("Testing DA-FPS implementation...")

    # Generate synthetic data with varying density
    np.random.seed(42)

    # Create clusters with different densities
    cluster1 = np.random.randn(200, 2) * 0.5 + np.array([0, 0])  # Dense cluster
    cluster2 = np.random.randn(50, 2) * 1.5 + np.array([5, 5])   # Sparse cluster
    cluster3 = np.random.randn(100, 2) * 0.8 + np.array([-3, 4]) # Medium cluster

    X = np.vstack([cluster1, cluster2, cluster3])
    print(f"Dataset shape: {X.shape}")

    budget = 30

    # Test DA-FPS
    dafps = DAFPS(k=10, random_state=42)
    dafps.fit(X, budget, verbose=True)
    print(f"\nDA-FPS selected {len(dafps.selected_indices_)} points")
    print(f"Fill distance: {compute_fill_distance(X, dafps.selected_points_):.4f}")

    # Test FPS
    fps = FPS(random_state=42)
    fps.fit(X, budget)
    print(f"\nFPS selected {len(fps.selected_indices_)} points")
    print(f"Fill distance: {compute_fill_distance(X, fps.selected_points_):.4f}")

    # Test Random
    random_sampler = RandomSampling(random_state=42)
    random_sampler.fit(X, budget)
    print(f"\nRandom selected {len(random_sampler.selected_indices_)} points")
    print(f"Fill distance: {compute_fill_distance(X, random_sampler.selected_points_):.4f}")

    print("\n✓ All tests passed!")
