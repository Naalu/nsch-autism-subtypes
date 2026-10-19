"""Hierarchical clustering: the proposal's second method, written from scratch.

Like ``lca.py``, this file holds the result type and the function signatures;
the algorithm is implemented by the project team (see
``docs/implementation-spec.md``). ``reference.cluster_scipy`` is the library
version the pipeline compares against.

Why a second method
-------------------
If two methods that work on different principles find the same groups, the
groups are probably a feature of the data and not an artifact of one method.
Latent class analysis assumes a probability model; hierarchical clustering
assumes nothing beyond a distance between children. Agreement between them
is one of the project's checks (step 4).

How agglomerative hierarchical clustering works
-----------------------------------------------
"Agglomerative" means bottom-up. Every child starts as its own group. At each
step, find the two groups that are closest together and merge them. Repeat
until one group is left. Recording the merges gives a tree (a "dendrogram");
cutting the tree at the right height gives K groups.

Two choices define the method:

Distance between two children. We use the Jaccard distance on the sets of
conditions each child has. If child A has {ADHD, anxiety} and child B has
{ADHD, epilepsy}, they share 1 condition out of 3 distinct ones, so the
Jaccard similarity is 1/3 and the distance is 1 - 1/3 = 2/3. In general

    d(A, B) = 1 - |A and B| / |A or B|

Two children with no conditions at all have distance 0 by convention.
Jaccard ignores conditions that both children lack, which matters here
because most children lack most of the 18 conditions.

Distance between two groups ("linkage"). We use average linkage, also called
UPGMA: the distance between groups is the mean of all pairwise distances
between their members. When each row stands for ``weights[i]`` identical
children, the exact update after merging groups A and B is

    d(A+B, C) = (w_A * d(A, C) + w_B * d(B, C)) / (w_A + w_B)

which is why the pipeline can collapse the 8,891 children into about 1,350
distinct condition profiles, pass the counts as weights, and get the same
answer much faster.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HierarchicalResult:
    """Cluster labels plus the full merge history (the tree).

    Attributes
    ----------
    labels : ndarray, shape (n,)
        Cluster index 0..n_clusters-1 for each input row.
    merges : ndarray, shape (n - 1, 4)
        One row per merge, in the format SciPy uses so that
        ``scipy.cluster.hierarchy.dendrogram`` can draw it directly:
        ``[group_a, group_b, distance, size]``. Original rows are numbered
        0..n-1 and the group created by merge number ``m`` is numbered
        ``n + m``. ``size`` is how many original rows the new group holds.
        Distances never decrease from one row to the next.
    """

    labels: np.ndarray
    merges: np.ndarray


def jaccard_distances(X: np.ndarray) -> np.ndarray:
    """Pairwise Jaccard distances between the rows of a 0/1 matrix.

    Parameters
    ----------
    X : ndarray, shape (n, J), values 0 or 1

    Returns
    -------
    ndarray, shape (n, n)
        ``out[i, j]`` is the Jaccard distance between rows i and j. The diagonal
        is 0 and the matrix is symmetric. Two all-zero rows have distance 0.
    """
    raise NotImplementedError("jaccard_distances is implemented by the project team; see docs/implementation-spec.md")


def cluster_hierarchical(
    X: np.ndarray,
    n_clusters: int,
    weights: np.ndarray | None = None,
) -> HierarchicalResult:
    """Group the rows of a 0/1 matrix into ``n_clusters`` clusters by average-linkage hierarchical clustering.

    Parameters
    ----------
    X : ndarray, shape (n, J)
        The flag matrix, values 0 or 1, no missing values.
    n_clusters : int
        How many groups to cut the tree into (1 <= n_clusters <= n).
    weights : ndarray, shape (n,), optional
        Positive counts. Row ``i`` stands for ``weights[i]`` identical children.
        ``None`` means every row is one child.

    Returns
    -------
    HierarchicalResult
    """
    raise NotImplementedError("cluster_hierarchical is implemented by the project team; see docs/implementation-spec.md")
