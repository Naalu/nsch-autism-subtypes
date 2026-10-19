"""Library versions of the three clustering methods, used to check the from-scratch code.

Each function here wraps a well-tested library (StepMix, SciPy, scikit-learn)
and repackages its output into the same result types the from-scratch
modules return. That lets the pipeline switch between the two with a single
flag (``--backend ours`` or ``--backend reference``) and lets the tests
compare them number for number.

If you are reading the code to learn the methods, read ``lca.py``,
``hierarchical.py`` and ``partition.py`` first; this file is plumbing.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from .hierarchical import HierarchicalResult
from .lca import LCAResult
from .partition import PartitionResult


def fit_lca_stepmix(
    X: np.ndarray,
    n_classes: int,
    n_starts: int = 20,
    seed: int = 0,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> LCAResult:
    """Fit a latent class model with the StepMix package. Same arguments as :func:`nsch_subtypes.lca.fit_lca`."""
    from stepmix import StepMix

    X = np.asarray(X, dtype=float)
    model = StepMix(
        n_components=n_classes,  # StepMix's word for K
        measurement="binary",  # each item is a yes/no flag
        n_init=n_starts,
        random_state=seed,
        abs_tol=tol,
        max_iter=max_iter,
        verbose=0,
        progress_bar=0,
    )
    model.fit(X)

    # Pull the fitted numbers out of StepMix's nested dictionary.
    parameters = model.get_parameters()
    condition_rates = np.asarray(parameters["measurement"]["pis"], dtype=float)  # shape (K, J)
    class_sizes = np.asarray(parameters["weights"], dtype=float)  # shape (K,)
    responsibilities = model.predict_proba(X)  # shape (n, K)

    # StepMix's score() is the average log-likelihood per child; we report the total.
    log_likelihood = float(model.score(X) * X.shape[0])

    return LCAResult(
        n_classes=n_classes,
        class_sizes=class_sizes,
        condition_rates=condition_rates,
        responsibilities=responsibilities,
        log_likelihood=log_likelihood,
        n_iter=int(getattr(model, "n_iter_", -1)),
        converged=bool(getattr(model, "converged_", True)),
    )


def cluster_scipy(X: np.ndarray, n_clusters: int, weights: np.ndarray | None = None) -> HierarchicalResult:
    """Average-linkage clustering on Jaccard distances with SciPy. Same arguments as :func:`nsch_subtypes.hierarchical.cluster_hierarchical`.

    SciPy has no weighted average linkage, so ``weights`` are honored by
    literally repeating each row ``weights[i]`` times. That gives the exact
    answer but is slow for thousands of children; the from-scratch version
    handles weights directly.
    """
    X = np.asarray(X, dtype=bool)
    if weights is not None:
        X = np.repeat(X, np.asarray(weights, dtype=int), axis=0)

    # pdist returns the distances between every pair of rows as one long vector.
    pairwise = pdist(X, metric="jaccard")
    # SciPy gives NaN for two all-zero rows (0/0); we define that distance as 0.
    pairwise = np.nan_to_num(pairwise, nan=0.0)

    merges = linkage(pairwise, method="average")
    # fcluster numbers clusters from 1; we use 0-based labels everywhere.
    labels = fcluster(merges, t=n_clusters, criterion="maxclust") - 1
    return HierarchicalResult(labels=labels, merges=merges)


def cluster_kmeans_sklearn(
    X: np.ndarray,
    n_clusters: int,
    n_starts: int = 20,
    seed: int = 0,
    max_iter: int = 300,
) -> PartitionResult:
    """k-means with scikit-learn. Same arguments as :func:`nsch_subtypes.partition.cluster_kmeans`."""
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=n_clusters, n_init=n_starts, random_state=seed, max_iter=max_iter)
    model.fit(np.asarray(X, dtype=float))
    return PartitionResult(labels=model.labels_, centers=model.cluster_centers_, inertia=float(model.inertia_))
