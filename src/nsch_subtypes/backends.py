"""Choose between the from-scratch clustering code ("ours") and the library versions ("reference").

Every pipeline script takes ``--backend ours`` or ``--backend reference`` and
asks this module for the matching functions. The two backends have identical
signatures and return identical result types, so the rest of the code never
needs to know which one it is using.

The ``reference`` backend exists for two reasons: it lets the pipeline run
end to end before the from-scratch code is finished, and it gives the tests
something trustworthy to compare the from-scratch code against.
"""

from __future__ import annotations

from collections.abc import Callable

from . import hierarchical, lca, partition, reference

# Method name -> function, for each backend.
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
    """Return the dictionary ``{"lca": ..., "hierarchical": ..., "kmeans": ...}`` for ``name``."""
    try:
        return BACKENDS[name]
    except KeyError:
        raise ValueError(f"Unknown backend {name!r}; choose from {sorted(BACKENDS)}") from None
