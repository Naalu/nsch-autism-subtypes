"""Pick the clustering implementations by name: ``ours`` or ``reference``."""

from __future__ import annotations

from collections.abc import Callable

from . import hierarchical, lca, partition, reference

BACKENDS: dict[str, dict[str, Callable]] = {
    "ours": {
        "lca": lca.fit_lca,
        "hierarchical": hierarchical.cluster_hierarchical,
        "kmeans": partition.cluster_kmeans,
    },
    "reference": {
        "lca": reference.fit_lca_stepmix,
        "hierarchical": reference.cluster_scipy,
        "kmeans": reference.cluster_kmeans_sklearn,
    },
}


def get_backend(name: str) -> dict[str, Callable]:
    """Return ``{"lca": ..., "hierarchical": ..., "kmeans": ...}`` for the named backend."""
    try:
        return BACKENDS[name]
    except KeyError:
        raise ValueError(f"Unknown backend {name!r}; choose from {sorted(BACKENDS)}") from None
