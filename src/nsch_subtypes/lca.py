"""Latent class analysis: the project's main clustering method, implemented from scratch.

This file implements Bernoulli-mixture EM using NumPy only. The pipeline
selects it with ``--backend ours``; ``--backend reference`` selects the
StepMix wrapper. Both return LCAResult (see docs/implementation-spec.md).

The model
---------
``docs/background.md`` explains the idea in plain language. In symbols,
child ``i`` has flags ``x_i1 ... x_iJ``. Class ``k`` has size
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
        Adding a class always improves the fit; BIC asks whether the improvement
        is worth the extra parameters. On this data it keeps falling through
        K = 8, so K is chosen by the stability rule in step 13 and BIC is
        reported alongside.
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


def _expectation_step(
    flag_matrix: np.ndarray,
    class_sizes: np.ndarray,
    condition_rates: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return class probabilities for each child and the total log-likelihood.

    The intermediate tables have shape (children, classes). Computing in
    log space avoids multiplying many small condition probabilities.
    """
    # A present condition contributes log(theta); an absent one contributes
    # log(1 - theta). Adding log(class size) gives the joint probability.
    log_joint = flag_matrix @ np.log(condition_rates).T
    log_joint += (1 - flag_matrix) @ np.log1p(-condition_rates).T
    safe_class_sizes = np.maximum(class_sizes, np.finfo(float).tiny)
    log_joint += np.log(safe_class_sizes)

    # Subtract each row's maximum before exponentiating. This rescales the
    # row without changing the relative probabilities of its classes.
    row_maximum = log_joint.max(axis=1, keepdims=True)
    relative_probabilities = np.exp(log_joint - row_maximum)
    row_totals = relative_probabilities.sum(axis=1, keepdims=True)
    responsibilities = relative_probabilities / row_totals

    # Undo that rescaling when calculating the probability of each child.
    log_likelihood = float((row_maximum + np.log(row_totals)).sum())
    return responsibilities, log_likelihood


# Iteration cap shared by every fit in the pipeline (step 3, the refits in
# steps 5 to 14, and the StepMix reference). With 20 starts the K=8 fit on the
# real data needs more than 500 rounds to converge, so one cap lives here and
# the scripts do not set their own.
MAX_ITER = 2000


def fit_lca(
    X: np.ndarray,
    n_classes: int,
    n_starts: int = 20,
    seed: int = 0,
    tol: float = 1e-6,
    max_iter: int = MAX_ITER,
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

    Notes
    -----
    The E-step works in log space with the log-sum-exp trick, since the product
    of 18 small probabilities underflows. Rates are clipped to ``[1e-6, 1 - 1e-6]``
    after each M-step. Each start draws rates uniformly from (0.2, 0.8) with
    equal class sizes. For ``n_classes == 1`` the rates are the column means and
    no iteration is needed. See ``docs/implementation-spec.md`` for the contract
    shared with the StepMix reference.
    """
    # Validate once, before any random starts or numerical work.
    flag_matrix = np.asarray(X, dtype=float)
    if flag_matrix.ndim != 2 or min(flag_matrix.shape) == 0 or not np.isin(flag_matrix, [0, 1]).all():
        raise ValueError("X must be a nonempty binary matrix")
    if not isinstance(n_classes, (int, np.integer)) or n_classes < 1:
        raise ValueError("n_classes must be a positive integer")
    if any(not isinstance(value, (int, np.integer)) for value in (n_starts, max_iter)):
        raise ValueError("n_starts and max_iter must be integers")
    if n_starts < 1 or max_iter < 1 or not np.isfinite(tol) or tol < 0:
        raise ValueError("n_starts and max_iter must be positive and tol finite and nonnegative")
    n_children, n_conditions = flag_matrix.shape

    # With one class, the best condition rates are simply the column means.
    # Clipping also handles conditions that are always absent or present.
    if n_classes == 1:
        condition_rates = np.clip(flag_matrix.mean(axis=0, keepdims=True), 1e-6, 1 - 1e-6)
        class_sizes = np.ones(1)
        responsibilities, log_likelihood = _expectation_step(flag_matrix, class_sizes, condition_rates)
        return LCAResult(
            n_classes=1,
            class_sizes=class_sizes,
            condition_rates=condition_rates,
            responsibilities=responsibilities,
            log_likelihood=log_likelihood,
            n_iter=0,
            converged=True,
        )

    random_generator = np.random.default_rng(seed)
    best_fit = None
    for _start in range(n_starts):
        # Different starting profiles explore different local optima. Using
        # one seeded generator makes the entire sequence reproducible.
        condition_rates = random_generator.uniform(0.2, 0.8, size=(n_classes, n_conditions))
        class_sizes = np.full(n_classes, 1 / n_classes)
        responsibilities, log_likelihood = _expectation_step(flag_matrix, class_sizes, condition_rates)
        converged = False

        for _iteration in range(1, max_iter + 1):
            # M-step: count each child fractionally according to their class
            # probabilities, then calculate the rates within each class.
            class_membership_totals = responsibilities.sum(axis=0)
            class_sizes = class_membership_totals / n_children
            weighted_condition_totals = responsibilities.T @ flag_matrix
            condition_rates = np.divide(
                weighted_condition_totals,
                class_membership_totals[:, None],
                out=condition_rates.copy(),
                where=class_membership_totals[:, None] > 0,
            )
            # The divide above keeps the old rates for an empty class.
            # Clipping prevents log(0) in the next E-step.
            condition_rates = np.clip(condition_rates, 1e-6, 1 - 1e-6)

            # E-step: score the updated parameters. Do this before checking
            # convergence so the returned probabilities and score describe
            # the returned parameters, even at the iteration limit.
            responsibilities, updated_log_likelihood = _expectation_step(flag_matrix, class_sizes, condition_rates)
            improvement = updated_log_likelihood - log_likelihood
            log_likelihood = updated_log_likelihood
            if 0 <= improvement < tol:
                converged = True
                break

        current_fit = LCAResult(
            n_classes=n_classes,
            class_sizes=class_sizes,
            condition_rates=condition_rates,
            responsibilities=responsibilities,
            log_likelihood=log_likelihood,
            n_iter=_iteration,
            converged=converged,
        )
        # Compare starts by fit to the data, not by their class numbering.
        if best_fit is None or current_fit.log_likelihood > best_fit.log_likelihood:
            best_fit = current_fit

    return best_fit
