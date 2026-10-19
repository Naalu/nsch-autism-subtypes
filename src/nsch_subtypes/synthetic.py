"""Make fake 0/1 data with known classes, so the clustering code can be tested against the truth.

With real survey data nobody knows the "right" clusters. With synthetic data
we do, because we generated it from chosen classes. The tests use this to
check that an implementation recovers the planted classes.
"""

from __future__ import annotations

import numpy as np


def make_synthetic(
    n: int = 600,
    n_items: int = 8,
    n_classes: int = 3,
    seed: int = 0,
    separation: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``n`` fake children from ``n_classes`` planted latent classes.

    Each class gets a few "characteristic" items that its members have with
    probability ``0.2 + separation`` (0.8 by default); every other item is
    rare, with probability 0.1. Larger ``separation`` makes the classes
    easier to tell apart. Classes are as close to equal-sized as ``n`` allows.

    Returns
    -------
    X : ndarray, shape (n, n_items), values 0 or 1
        The flag matrix.
    labels : ndarray, shape (n,)
        The true class of each row.
    rates : ndarray, shape (n_classes, n_items)
        The condition rates the data were generated from.
    """
    rng = np.random.default_rng(seed)

    # Start every rate at "rare", then raise a distinct subset of items for each class.
    rates = np.full((n_classes, n_items), 0.1)
    for k in range(n_classes):
        characteristic_items = rng.choice(n_items, size=max(1, n_items // n_classes), replace=False)
        rates[k, characteristic_items] = 0.2 + separation

    # Equal class sizes, with any remainder assigned at random, then shuffled.
    labels = np.repeat(np.arange(n_classes), n // n_classes)
    labels = np.concatenate([labels, rng.integers(0, n_classes, n - len(labels))])
    rng.shuffle(labels)

    # Each flag is a coin flip with that child's class rate for that item.
    X = (rng.random((n, n_items)) < rates[labels]).astype(np.uint8)
    return X, labels, rates
