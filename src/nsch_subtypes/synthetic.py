"""Synthetic binary data with known classes, for tests and for checking implementations."""

from __future__ import annotations

import numpy as np


def make_synthetic(
    n: int = 600,
    n_items: int = 8,
    n_classes: int = 3,
    seed: int = 0,
    separation: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``n`` children from ``n_classes`` latent classes.

    Each class gets its own rate per item: a random subset of items is
    "characteristic" of the class (rate ``0.2 + separation``) and the rest are
    rare (rate 0.1). Classes are equal-sized.

    Returns
    -------
    X : ndarray (n, n_items) of 0/1
    labels : ndarray (n,) true class of each row
    rates : ndarray (n_classes, n_items) the generating rates
    """
    rng = np.random.default_rng(seed)
    rates = np.full((n_classes, n_items), 0.1)
    for k in range(n_classes):
        chosen = rng.choice(n_items, size=max(1, n_items // n_classes), replace=False)
        rates[k, chosen] = 0.2 + separation
    labels = np.repeat(np.arange(n_classes), n // n_classes)
    labels = np.concatenate([labels, rng.integers(0, n_classes, n - len(labels))])
    rng.shuffle(labels)
    X = (rng.random((n, n_items)) < rates[labels]).astype(np.uint8)
    return X, labels, rates
