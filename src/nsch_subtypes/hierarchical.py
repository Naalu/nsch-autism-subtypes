"""Hierarchical clustering: the proposal's second method, implemented from scratch.

This file implements Jaccard distances and weighted average linkage using
NumPy only. Select it with ``--backend ours``; ``reference.cluster_scipy``
is the library counterpart (see docs/implementation.md).

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
        One row per merge, using SciPy group ids and merge distances:
        ``[group_a, group_b, distance, size]``. Original rows are numbered
        0..n-1 and the group created by merge number ``m`` is numbered
        ``n + m``. ``size`` is the total input weight held by the group (row count
        when weights are omitted).
        Weighted sizes must be converted to leaf counts before passing the
        tree to SciPy; ``plots.dendrogram_plot`` handles this conversion.
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
    # Use numeric arrays so matrix multiplication counts shared conditions
    # correctly even when the input uses booleans or small integer types.
    flag_matrix = np.asarray(X, dtype=float)
    if flag_matrix.ndim != 2 or min(flag_matrix.shape) == 0 or not np.isin(flag_matrix, [0, 1]).all():
        raise ValueError("X must be a nonempty binary matrix")

    # For each pair of children, count conditions shared by both (intersection)
    # and conditions present in either child (union).
    shared_conditions = flag_matrix @ flag_matrix.T
    conditions_per_child = flag_matrix.sum(axis=1)
    combined_conditions = conditions_per_child[:, None] + conditions_per_child[None, :] - shared_conditions

    # Similarity is shared / combined. Two all-zero rows get similarity 1,
    # and therefore distance 0, rather than an undefined division by zero.
    similarities = np.divide(
        shared_conditions,
        combined_conditions,
        out=np.ones_like(shared_conditions),
        where=combined_conditions > 0,
    )
    distances = 1 - similarities
    np.fill_diagonal(distances, 0)
    return distances


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
    distance_matrix = jaccard_distances(X)
    n_rows = len(distance_matrix)
    if not isinstance(n_clusters, (int, np.integer)) or not 1 <= n_clusters <= n_rows:
        raise ValueError("n_clusters must be between 1 and the number of rows")

    # A row can represent one child or many children with the same profile.
    cluster_weights = np.ones(n_rows) if weights is None else np.asarray(weights, dtype=float).copy()
    if cluster_weights.shape != (n_rows,) or not np.isfinite(cluster_weights).all() or np.any(cluster_weights <= 0):
        raise ValueError("weights must contain one finite positive count per row")

    # Matrix slots stay fixed while group ids change after each merge.
    # Reusing one of the merged slots keeps the distance matrix at n x n.
    row_positions = np.arange(n_rows)
    active_slots = np.ones(n_rows, dtype=bool)
    group_ids = np.arange(n_rows)
    members_by_slot = {row: [row] for row in range(n_rows)}
    merges = np.empty((n_rows - 1, 4))
    labels = np.arange(n_rows)

    # Infinite distances prevent choosing a group itself or a retired slot.
    np.fill_diagonal(distance_matrix, np.inf)
    for merge_index in range(n_rows - 1):
        # Select the closest active pair. NumPy's first minimum gives a
        # deterministic tie rule when several pairs have equal distance.
        slot_a, slot_b = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
        merge_distance = distance_matrix[slot_a, slot_b]
        merged_weight = cluster_weights[slot_a] + cluster_weights[slot_b]
        group_a = group_ids[slot_a]
        group_b = group_ids[slot_b]
        merges[merge_index] = [min(group_a, group_b), max(group_a, group_b), merge_distance, merged_weight]

        # Average linkage: the new group's distance to another group is
        # the weighted average of the two previous distances.
        other_groups = active_slots & (row_positions != slot_a) & (row_positions != slot_b)
        remaining_slots = np.flatnonzero(other_groups)
        updated_distances = (
            cluster_weights[slot_a] * distance_matrix[slot_a, remaining_slots] + cluster_weights[slot_b] * distance_matrix[slot_b, remaining_slots]
        ) / merged_weight
        distance_matrix[slot_a, remaining_slots] = updated_distances
        distance_matrix[remaining_slots, slot_a] = updated_distances

        # Keep the merged group in slot A and retire slot B. Public group
        # ids follow SciPy's convention: original rows, then new merges.
        distance_matrix[slot_b, :] = np.inf
        distance_matrix[:, slot_b] = np.inf
        active_slots[slot_b] = False
        cluster_weights[slot_a] = merged_weight
        group_ids[slot_a] = n_rows + merge_index
        members_by_slot[slot_a].extend(members_by_slot.pop(slot_b))

        # Save labels when the requested number of groups remains, then
        # continue merging to retain the full tree for the dendrogram.
        groups_remaining = n_rows - merge_index - 1
        if groups_remaining == n_clusters:
            groups_in_row_order = sorted(members_by_slot.values(), key=min)
            for cluster_label, member_rows in enumerate(groups_in_row_order):
                labels[member_rows] = cluster_label

    return HierarchicalResult(labels=labels, merges=merges)
