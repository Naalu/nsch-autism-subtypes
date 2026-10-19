"""Library implementations used to check the from-scratch code.

``fit_lca_stepmix`` wraps StepMix (a latent class package), ``cluster_scipy``
wraps SciPy's average-linkage routine and ``cluster_kmeans_sklearn`` wraps
scikit-learn. All return the same result types as the from-scratch modules
so the pipeline can switch between backends with one flag.
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
    """Fit a binary latent class model with StepMix and repackage the result."""
    from stepmix import StepMix

    X = np.asarray(X, dtype=float)
    model = StepMix(
        n_components=n_classes,
        measurement="binary",
        n_init=n_starts,
        random_state=seed,
        abs_tol=tol,
        max_iter=max_iter,
        verbose=0,
        progress_bar=0,
    )
    model.fit(X)
    params = model.get_parameters()
    rates = np.asarray(params["measurement"]["pis"], dtype=float)
    sizes = np.asarray(params["weights"], dtype=float)
    resp = model.predict_proba(X)
    ll = float(model.score(X) * X.shape[0])  # StepMix reports the mean log-likelihood
    return LCAResult(
        n_classes=n_classes,
        class_sizes=sizes,
        condition_rates=rates,
        responsibilities=resp,
        log_likelihood=ll,
        n_iter=int(getattr(model, "n_iter_", -1)),
        converged=bool(getattr(model, "converged_", True)),
    )


def cluster_scipy(X: np.ndarray, n_clusters: int, weights: np.ndarray | None = None) -> HierarchicalResult:
    """Average-linkage clustering on Jaccard distances with SciPy.

    SciPy has no weighted average linkage, so ``weights`` are honored by
    repeating rows, which is exact but only practical for modest counts.
    """
    X = np.asarray(X, dtype=bool)
    if weights is not None:
        X = np.repeat(X, np.asarray(weights, dtype=int), axis=0)
    d = pdist(X, metric="jaccard")
    d = np.nan_to_num(d, nan=0.0)  # two all-zero rows: SciPy returns nan, we say 0
    merges = linkage(d, method="average")
    labels = fcluster(merges, t=n_clusters, criterion="maxclust") - 1
    return HierarchicalResult(labels=labels, merges=merges)


def cluster_kmeans_sklearn(X: np.ndarray, n_clusters: int, n_starts: int = 20, seed: int = 0, max_iter: int = 300) -> PartitionResult:
    """k-means on the 0/1 flags with scikit-learn."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, n_init=n_starts, random_state=seed, max_iter=max_iter).fit(np.asarray(X, dtype=float))
    return PartitionResult(labels=km.labels_, centers=km.cluster_centers_, inertia=float(km.inertia_))
