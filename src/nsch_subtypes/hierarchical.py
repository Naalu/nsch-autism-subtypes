"""Agglomerative hierarchical clustering with Jaccard distance and average linkage.

This is the project's cross-check method and is written from scratch (NumPy
only) as part of the course requirement. ``docs/implementation-spec.md``
spells out the contract; ``reference.cluster_scipy`` is the library
implementation the pipeline checks it against.

Method
------
Every child starts as its own group. At each step the two groups with the
smallest average pairwise distance are merged, until one group remains. The
merge history is a tree; cutting it to ``n_clusters`` groups gives the labels.

Jaccard distance between two children is ``1 - |A & B| / |A | B|`` where A and
B are their sets of flags equal to 1. Two children with no flags at all have
distance 0 by convention.

Average linkage (UPGMA): the distance between groups is the mean of the
pairwise distances between their members. With ``weights`` (how many children
each row stands for), the update after merging A and B is

    d(A+B, C) = (w_A d(A, C) + w_B d(B, C)) / (w_A + w_B)

which is why the pipeline can collapse identical rows first and pass counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HierarchicalResult:
    """Labels plus the full merge history.

    Attributes
    ----------
    labels : ndarray, shape (n,)
        Cluster index 0..n_clusters-1 for each input row.
    merges : ndarray, shape (n - 1, 4)
        One row per merge in SciPy linkage format: ``[i, j, distance, size]``
        where ``i`` and ``j`` are the merged groups (original rows are
        0..n-1, the group formed by merge ``m`` is ``n + m``) and ``size`` is
        the number of input rows in the new group. Distances are
        non-decreasing down the rows. This format lets
        ``scipy.cluster.hierarchy.dendrogram`` draw the tree directly.
    """

    labels: np.ndarray
    merges: np.ndarray


def jaccard_distances(X: np.ndarray) -> np.ndarray:
    """Pairwise Jaccard distance matrix, shape (n, n), for a 0/1 matrix."""
    raise NotImplementedError("jaccard_distances is implemented by the project team; see docs/implementation-spec.md")


def cluster_hierarchical(
    X: np.ndarray,
    n_clusters: int,
    weights: np.ndarray | None = None,
) -> HierarchicalResult:
    """Cluster a 0/1 matrix into ``n_clusters`` groups.

    Parameters
    ----------
    X : ndarray, shape (n, J), values 0 or 1, no missing values
    n_clusters : number of groups to cut the tree into (1 <= n_clusters <= n)
    weights : optional ndarray, shape (n,), positive counts; row ``i`` stands
        for ``weights[i]`` identical children. ``None`` means all ones.

    Returns
    -------
    HierarchicalResult
    """
    raise NotImplementedError("cluster_hierarchical is implemented by the project team; see docs/implementation-spec.md")
