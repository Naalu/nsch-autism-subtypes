"""Latent class analysis: the project's main clustering method, written from scratch.

This file holds the result type and the function signature. The algorithm
itself is implemented by the project team as a course requirement (see
``docs/implementation-spec.md``). Until then ``fit_lca`` raises
``NotImplementedError`` and the pipeline falls back to ``--backend reference``,
which uses the StepMix library (``reference.fit_lca_stepmix``).

What a latent class model is, in plain terms
--------------------------------------------
Suppose there are K hidden "types" of children. Each type has its own
probability of having each condition: type A might have a 90% chance of
ADHD and a 10% chance of epilepsy, type B the reverse. We never see the
type; we only see each child's row of 0s and 1s. The model asks: what K
types, with what sizes and condition probabilities, would most plausibly
have produced the rows we see?

In symbols, child ``i`` has flags ``x_i1 ... x_iJ``. Class ``k`` has size
``pi_k`` (the ``pi_k`` sum to 1) and condition rates ``theta_kj``. Within a
class the flags are treated as independent coin flips ("local independence"),
so the probability of child ``i``'s row is

    P(x_i) = sum over k of  pi_k * product over j of  theta_kj^x_ij * (1 - theta_kj)^(1 - x_ij)

This is a mixture of Bernoulli distributions. "Latent class analysis" is the
name social scientists use for it.

How it is fit: the EM algorithm
-------------------------------
There is no formula that gives the best ``pi`` and ``theta`` directly, so we
use Expectation-Maximization, which alternates two easy steps:

- E-step: given the current ``pi`` and ``theta``, compute for every child the
  probability of belonging to each class (the "responsibilities", an n-by-K
  table whose rows sum to 1).
- M-step: given the responsibilities, update ``pi_k`` to the average
  responsibility for class k, and ``theta_kj`` to the responsibility-weighted
  share of children with flag j.

Each round can only raise the log-likelihood (or leave it unchanged), so the
algorithm settles down. It can settle on a poor local optimum, which is why
we run it from several random starts and keep the best.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LCAResult:
    """Everything a fitted latent class model produces.

    Attributes
    ----------
    n_classes : int
        K, the number of classes.
    class_sizes : ndarray, shape (K,)
        ``pi_k``: the share of children in each class. Sums to 1.
    condition_rates : ndarray, shape (K, J)
        ``theta_kj``: the probability that a child in class k has condition j.
        A row of this array is the "profile" of a class.
    responsibilities : ndarray, shape (n, K)
        For each child, the probability of belonging to each class. Rows sum to 1.
    log_likelihood : float
        The total log-likelihood of the data at the fitted parameters. Higher
        means the model explains the data better (for a fixed K).
    n_iter : int
        Number of EM rounds the best start needed.
    converged : bool
        Whether that start stopped because the improvement fell below ``tol``
        rather than because it hit ``max_iter``.
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
        """The single most likely class for each child (the "modal" assignment), as integers 0..K-1."""
        return self.responsibilities.argmax(axis=1)

    @property
    def n_parameters(self) -> int:
        """How many free numbers the model has: K-1 class sizes (they sum to 1) plus K*J rates."""
        n_classes, n_items = self.condition_rates.shape
        return (n_classes - 1) + n_classes * n_items

    def bic(self, n: int | None = None) -> float:
        """Bayesian information criterion, ``-2 * log_likelihood + n_parameters * ln(n)``. Lower is better.

        BIC rewards fit (the first term) and penalizes complexity (the second).
        We use it to choose K: adding a class always improves the fit, and BIC
        asks whether the improvement is worth the extra parameters.
        """
        n_children = self.responsibilities.shape[0] if n is None else n
        return -2.0 * self.log_likelihood + self.n_parameters * np.log(n_children)

    def entropy(self) -> float:
        """How cleanly children are assigned to classes, on a 0 to 1 scale.

        1 means every child has probability 1 for one class and 0 for the
        others. Values near 0 mean the classes overlap so much that the model
        cannot tell which class most children belong to. Published latent
        class studies usually treat 0.8 as "good separation".
        """
        probabilities = np.clip(self.responsibilities, 1e-12, 1.0)
        n_children, n_classes = probabilities.shape
        if n_classes == 1:
            return 1.0
        total_entropy = -(probabilities * np.log(probabilities)).sum()
        worst_case_entropy = n_children * np.log(n_classes)  # every child spread evenly over K classes
        return float(1.0 - total_entropy / worst_case_entropy)


def fit_lca(
    X: np.ndarray,
    n_classes: int,
    n_starts: int = 20,
    seed: int = 0,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> LCAResult:
    """Fit a latent class model with ``n_classes`` classes to a 0/1 matrix.

    Parameters
    ----------
    X : ndarray, shape (n, J)
        The flag matrix: one row per child, one column per condition, values
        0 or 1, no missing values. (``X`` is the usual name for a data matrix
        in machine learning code.)
    n_classes : int
        K, at least 1.
    n_starts : int
        Number of random initializations. EM can get stuck in a poor local
        optimum, so run it ``n_starts`` times and return the start whose
        final log-likelihood is highest.
    seed : int
        Seed for the random starts, via ``numpy.random.default_rng(seed)``,
        so that a run can be repeated exactly.
    tol : float
        Stop a start when one EM round improves the log-likelihood by less than this.
    max_iter : int
        Give up on a start after this many rounds.

    Returns
    -------
    LCAResult

    Notes for whoever implements this (full contract in docs/implementation-spec.md)
    ---------------------------------------------------------------------------------
    - Work in log space and use the log-sum-exp trick in the E-step. A product
      of 18 probabilities is a number around 1e-10 and gets smaller with each
      class; multiplying those underflows to zero in floating point.
    - After each M-step clip the rates to ``[1e-6, 1 - 1e-6]`` so ``log(0)``
      never happens.
    - Initialize each start with rates drawn uniformly from (0.2, 0.8) and equal class sizes.
    - When ``n_classes == 1`` there is nothing to iterate: the rates are the
      column means. Return a valid LCAResult anyway (the K=1 fit is the
      baseline on the BIC curve).
    """
    raise NotImplementedError("fit_lca is implemented by the project team; see docs/implementation-spec.md")
