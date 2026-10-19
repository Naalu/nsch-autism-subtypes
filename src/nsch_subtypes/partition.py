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

As with the other methods, the algorithm is implemented by the team if we
adopt it; ``reference.cluster_kmeans_sklearn`` is the library version.

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
    raise NotImplementedError("cluster_kmeans is implemented by the project team; see docs/implementation-spec.md")
