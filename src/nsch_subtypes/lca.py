"""Latent class model for binary indicators, fit by EM.

This is the project's main clustering method and is written from scratch
(NumPy only) as part of the course requirement. ``docs/implementation-spec.md``
spells out the contract; ``reference.fit_lca_stepmix`` is the library
implementation the pipeline checks it against.

Model
-----
Each child ``i`` has ``J`` binary flags ``x_ij``. There are ``K`` classes with
sizes ``pi_k`` (summing to 1) and per-class condition rates ``theta_kj``.
Within a class, flags are independent (local independence), so

    P(x_i) = sum_k pi_k * prod_j theta_kj^x_ij * (1 - theta_kj)^(1 - x_ij)

EM alternates an E-step (posterior class probabilities for every child, the
"responsibilities") and an M-step (class sizes and rates as responsibility-
weighted averages). The log-likelihood never decreases between iterations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LCAResult:
    """A fitted latent class model.

    Attributes
    ----------
    n_classes : int
    class_sizes : ndarray, shape (K,)
        Mixing proportions ``pi_k``; sum to 1.
    condition_rates : ndarray, shape (K, J)
        ``theta_kj``: probability that a child in class ``k`` has flag ``j``.
    responsibilities : ndarray, shape (n, K)
        Posterior probability that child ``i`` belongs to class ``k``; rows sum to 1.
    log_likelihood : float
        Total log-likelihood of the data at the returned parameters.
    n_iter : int
        EM iterations used by the best start.
    converged : bool
    """

    n_classes: int
    class_sizes: np.ndarray
    condition_rates: np.ndarray
    responsibilities: np.ndarray
    log_likelihood: float
    n_iter: int
    converged: bool

    @property
    def labels(self) -> np.ndarray:
        """Most likely class for each child (modal assignment)."""
        return self.responsibilities.argmax(axis=1)

    @property
    def n_parameters(self) -> int:
        k, j = self.condition_rates.shape
        return (k - 1) + k * j

    def bic(self, n: int | None = None) -> float:
        """Bayesian information criterion: ``-2 ll + p ln n``. Lower is better."""
        n = self.responsibilities.shape[0] if n is None else n
        return -2.0 * self.log_likelihood + self.n_parameters * np.log(n)

    def entropy(self) -> float:
        """Relative entropy of the classification, in [0, 1]; 1 = certain assignments."""
        r = np.clip(self.responsibilities, 1e-12, 1.0)
        n, k = r.shape
        if k == 1:
            return 1.0
        return float(1.0 - (-(r * np.log(r)).sum() / (n * np.log(k))))


def fit_lca(
    X: np.ndarray,
    n_classes: int,
    n_starts: int = 20,
    seed: int = 0,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> LCAResult:
    """Fit a ``n_classes``-class latent class model to a 0/1 matrix.

    Parameters
    ----------
    X : ndarray, shape (n, J), values 0 or 1, no missing values
    n_classes : K >= 1
    n_starts : number of random initializations; the start with the highest
        final log-likelihood is returned
    seed : seed for the random starts (``numpy.random.default_rng(seed)``)
    tol : stop when the log-likelihood improves by less than this
    max_iter : iteration cap per start

    Returns
    -------
    LCAResult

    Notes for the implementation (see docs/implementation-spec.md)
    ---------------------------------------------------------------
    - Work in log space and use log-sum-exp in the E-step; a product of 20
      probabilities underflows otherwise.
    - Clip rates to [1e-6, 1 - 1e-6] after each M-step so log(0) never occurs.
    - Draw initial rates uniformly in (0.2, 0.8) and start with equal class sizes.
    - With K = 1 the fit is closed form (rates = column means) and must still
      return a valid LCAResult.
    """
    raise NotImplementedError("fit_lca is implemented by the project team; see docs/implementation-spec.md")
