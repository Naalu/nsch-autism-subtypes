"""k-means clustering: an alternative second method.

Why this file exists
--------------------
The proposal named hierarchical clustering as the cross-check for the latent
classes. On the real data, average linkage turned out to peel off a handful
of unusual children at each merge and leave one cluster holding almost
everyone, at every K we tried. That makes it useless as a check. k-means on
the 0/1 flags is what Vargason et al. (2019) used for the same kind of data,
so it was added as a second option. Which one the paper reports is an open
decision (see ``docs/validation-checklist.md``).

This file implements Lloyd's algorithm using NumPy only, selected with
``--backend ours``; ``reference.cluster_kmeans_sklearn`` is the library version.

How k-means works (Lloyd's algorithm)
-------------------------------------
1. Pick ``n_clusters`` children at random as the first "centers".
2. Assign every child to the nearest center. Distance is squared Euclidean on
   the 0/1 vector, which for a binary center is just the number of conditions
   on which the child and the center disagree.
3. Move each center to the mean of the children assigned to it. The center is
   now a vector of condition rates, like a latent class profile.
4. Repeat 2 and 3 until no child changes cluster.

Different starting centers can give different answers, so run it
``n_starts`` times and keep the run with the smallest total distance from
children to their centers (the "inertia").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PartitionResult:
    """Cluster labels plus the final centers.

    Attributes
    ----------
    labels : ndarray, shape (n,)
        Cluster index 0..K-1 for each child.
    centers : ndarray, shape (K, J)
        Mean flag vector of each cluster, comparable to latent class condition rates.
    inertia : float
        Sum of squared distances from each child to its center. Lower is a tighter clustering.
    """

    labels: np.ndarray
    centers: np.ndarray
    inertia: float


def _squared_distances(flag_matrix: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return the squared distance from every child to every center.

    Broadcasting creates a (children, clusters, conditions) table of
    differences. Summing over conditions leaves (children, clusters).
    """
    differences = flag_matrix[:, None, :] - centers[None, :, :]
    return (differences**2).sum(axis=2)


def cluster_kmeans(
    X: np.ndarray,
    n_clusters: int,
    n_starts: int = 20,
    seed: int = 0,
    max_iter: int = 300,
) -> PartitionResult:
    """Run k-means on a 0/1 matrix.

    Parameters
    ----------
    X : ndarray, shape (n, J)
        The flag matrix, values 0 or 1, no missing values.
    n_clusters : int
        K.
    n_starts : int
        Number of random initializations; the best by inertia is returned.
    seed : int
        Seed for choosing the starting centers.
    max_iter : int
        Cap on the assign/update rounds within one start.

    Returns
    -------
    PartitionResult
    """
    flag_matrix = np.asarray(X, dtype=float)
    if flag_matrix.ndim != 2 or min(flag_matrix.shape) == 0 or not np.isin(flag_matrix, [0, 1]).all():
        raise ValueError("X must be a nonempty binary matrix")
    n_children = len(flag_matrix)
    if not isinstance(n_clusters, (int, np.integer)) or not 1 <= n_clusters <= n_children:
        raise ValueError("n_clusters must be between 1 and the number of rows")
    if n_starts < 1 or max_iter < 1:
        raise ValueError("n_starts and max_iter must be positive")
    if any(not isinstance(value, (int, np.integer)) for value in (n_starts, max_iter)):
        raise ValueError("n_starts and max_iter must be integers")

    random_generator = np.random.default_rng(seed)
    best_partition = None
    for _start in range(n_starts):
        # Choose distinct row positions. Identical profiles can still occur
        # in different rows, so empty clusters are handled below.
        starting_rows = random_generator.choice(n_children, n_clusters, replace=False)
        centers = flag_matrix[starting_rows].copy()
        previous_labels = None

        for _iteration in range(max_iter):
            # Assignment step: choose the nearest center for each child.
            squared_distances = _squared_distances(flag_matrix, centers)
            labels = squared_distances.argmin(axis=1)
            if previous_labels is not None and np.array_equal(labels, previous_labels):
                break

            # Update step: each center becomes its members' mean profile.
            # Keep the previous center when a cluster has no members.
            for cluster_index in range(n_clusters):
                cluster_members = flag_matrix[labels == cluster_index]
                if len(cluster_members) > 0:
                    centers[cluster_index] = cluster_members.mean(axis=0)
            previous_labels = labels.copy()

        # Reassign against the final centers, including when the iteration
        # cap stopped us immediately after a center update.
        squared_distances = _squared_distances(flag_matrix, centers)
        labels = squared_distances.argmin(axis=1)
        assigned_distances = squared_distances[np.arange(n_children), labels]
        inertia = float(assigned_distances.sum())

        # Lower inertia means a tighter partition; keep the best start.
        if best_partition is None or inertia < best_partition.inertia:
            best_partition = PartitionResult(labels=labels, centers=centers, inertia=inertia)

    return best_partition
