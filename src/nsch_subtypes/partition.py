"""k-means on the 0/1 flags: an alternative cross-check method.

Added after the first runs showed that average-linkage hierarchical
clustering peels off outliers on this data (one cluster holds nearly every
child at any K). k-means on binary indicators is what Vargason et al. (2019)
used for the same problem, so it is a reasonable second method. Written from
scratch for the course if adopted; ``reference.cluster_kmeans_sklearn`` is the
library version.

Method (Lloyd's algorithm)
--------------------------
Start from ``n_clusters`` children chosen at random as centers. Repeat: assign
each child to the nearest center (squared Euclidean distance on the 0/1
vector, which equals the number of flags that differ from the center when the
center is binary); move each center to the mean of its members. Stop when no
assignment changes. Keep the best of ``n_starts`` runs by total within-cluster
distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PartitionResult:
    """Labels plus the final centers.

    Attributes
    ----------
    labels : ndarray, shape (n,)
    centers : ndarray, shape (K, J)
        Mean flag vector of each cluster.
    inertia : float
        Total squared distance of children to their centers (lower is better).
    """

    labels: np.ndarray
    centers: np.ndarray
    inertia: float


def cluster_kmeans(X: np.ndarray, n_clusters: int, n_starts: int = 20, seed: int = 0, max_iter: int = 300) -> PartitionResult:
    """k-means on a 0/1 matrix. See the module docstring and docs/implementation-spec.md."""
    raise NotImplementedError("cluster_kmeans is implemented by the project team; see docs/implementation-spec.md")
