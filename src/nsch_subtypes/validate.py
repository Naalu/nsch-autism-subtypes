"""The checks the clusters have to pass before we call them subtypes.

Finding clusters is easy; any clustering algorithm will split any data into K
groups if you ask it to. The hard part is showing that the groups mean
something. This module holds the tools for four checks:

1. Agreement between two clusterings (adjusted Rand index), used when a
   second method clusters the same children (step 4), when a model fit on
   early years is compared with one fit on later years (step 5), and when
   the frozen model meets the 2025 data (step 8).
2. Applying a fitted model to new children without refitting (``class_probabilities``).
3. Stability under resampling (bootstrap Jaccard, step 6).
4. Whether the classes tell us anything about an outcome they never saw,
   beyond what a simple count of conditions already tells us (step 7).

Pass lines, fixed in the proposal before any model was fit
----------------------------------------------------------
- ``ARI_PASS = 0.65``: two clusterings of the same children must agree at
  least this well. Steinley (2004) called 0.65 to 0.80 "moderate" agreement,
  0.80 to 0.90 "good" and above 0.90 "excellent". We take the floor of
  moderate because our comparisons are between different children or very
  different methods, where perfect agreement is not expected.
- ``JACCARD_STABLE = 0.75``: a class whose members come back together in
  resampled data at least this often counts as stable (Hennig 2007).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

ARI_PASS = 0.65
JACCARD_STABLE = 0.75
JACCARD_DISSOLVED = 0.50


# --------------------------------------------------------------------------- #
# 1. Agreement between two clusterings
# --------------------------------------------------------------------------- #
def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """How well two clusterings of the same children agree, from about 0 (chance) to 1 (identical).

    The Rand index counts pairs of children: pairs that both clusterings put
    together, plus pairs that both keep apart, divided by all pairs. The
    "adjusted" version (Hubert and Arabie 1985) subtracts what you would get
    by chance, so random labels score about 0 and can even go slightly
    negative. It does not care what the clusters are called, so cluster 2 in
    one labeling can be cluster 0 in the other.
    """
    return float(adjusted_rand_score(labels_a, labels_b))


def match_classes(rates_a: np.ndarray, rates_b: np.ndarray) -> np.ndarray:
    """Work out which class in model B corresponds to which class in model A.

    Two models fit separately will number their classes arbitrarily. We pair
    them up by comparing condition profiles: the distance between class k of
    A and class m of B is the Euclidean distance between their rate vectors,
    and the Hungarian algorithm (``linear_sum_assignment``) finds the
    one-to-one pairing with the smallest total distance.

    Returns ``perm`` such that B's class ``perm[k]`` is the match for A's class ``k``.
    Both models must have the same number of classes.
    """
    # cost[k, m] = distance between A's class k and B's class m, shape (K, K).
    cost = np.linalg.norm(rates_a[:, None, :] - rates_b[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(len(rows), dtype=int)
    perm[rows] = cols
    return perm


def relabel_to_match(labels_b: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Renumber B's labels so they use A's numbering, given ``perm`` from :func:`match_classes`."""
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(len(perm))
    return inverse[labels_b]


def mean_rate_difference(rates_a: np.ndarray, rates_b: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """For each matched pair of classes, the average absolute gap in condition rates (one number per class, 0 = identical profiles)."""
    return np.abs(rates_a - rates_b[perm]).mean(axis=1)


# --------------------------------------------------------------------------- #
# 2. Applying a fitted model to new children
# --------------------------------------------------------------------------- #
def class_probabilities(X: np.ndarray, class_sizes: np.ndarray, condition_rates: np.ndarray) -> np.ndarray:
    """Probability that each child belongs to each class, under a model that is not refit.

    This is one E-step of the EM algorithm, run once with fixed parameters.
    It is how the held-out checks label new children (later survey years,
    the 2025 file) with a model fit on other children.

    Parameters
    ----------
    X : ndarray, shape (n, J), values 0 or 1
    class_sizes : ndarray, shape (K,)
    condition_rates : ndarray, shape (K, J)

    Returns
    -------
    ndarray, shape (n, K), rows sum to 1
    """
    X = np.asarray(X, dtype=float)
    rates = np.clip(np.asarray(condition_rates, dtype=float), 1e-6, 1 - 1e-6)

    # log P(child's row | class k), for every child and class, shape (n, K).
    # Working in logs avoids multiplying 18 small probabilities into zero.
    log_likelihood_per_class = X @ np.log(rates).T + (1 - X) @ np.log(1 - rates).T

    # Bayes' rule: add log prior (class size), then normalize each row.
    log_posterior = log_likelihood_per_class + np.log(np.asarray(class_sizes, dtype=float))[None, :]
    log_posterior -= log_posterior.max(axis=1, keepdims=True)  # subtract the row max before exp, for numerical safety
    posterior = np.exp(log_posterior)
    return posterior / posterior.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# 3. Bootstrap stability (Hennig 2007)
# --------------------------------------------------------------------------- #
@dataclass
class StabilityResult:
    """Bootstrap Jaccard scores for every class and every resample.

    Attributes
    ----------
    jaccard : ndarray, shape (K, n_boot)
        ``jaccard[k, b]`` is how much of original class k came back together
        in resample b (1 = the exact same children, 0 = none of them).
    """

    jaccard: np.ndarray

    @property
    def mean(self) -> np.ndarray:
        """Average Jaccard per class across resamples, shape (K,)."""
        return self.jaccard.mean(axis=1)

    def verdicts(self) -> list[str]:
        """Hennig's reading of the mean Jaccard, one phrase per class."""
        verdicts = []
        for mean_jaccard in self.mean:
            if mean_jaccard >= 0.85:
                verdicts.append("highly stable")
            elif mean_jaccard >= JACCARD_STABLE:
                verdicts.append("stable")
            elif mean_jaccard >= 0.6:
                verdicts.append("pattern, membership doubtful")
            elif mean_jaccard > JACCARD_DISSOLVED:
                verdicts.append("not trustworthy")
            else:
                verdicts.append("dissolved")
        return verdicts


def _jaccard_of_sets(members_a: np.ndarray, members_b: np.ndarray) -> float:
    """Jaccard similarity of two sets of child indices: shared / (either), or 0 if both are empty."""
    n_shared = np.intersect1d(members_a, members_b).size
    n_either = np.union1d(members_a, members_b).size
    return n_shared / n_either if n_either else 0.0


def bootstrap_stability(
    X: np.ndarray,
    labels_ref: np.ndarray,
    fit: Callable[[np.ndarray, int], np.ndarray],
    n_classes: int,
    n_boot: int = 200,
    seed: int = 0,
    progress: Callable[[int], None] | None = None,
) -> StabilityResult:
    """Does each class survive when the sample is perturbed?

    The idea (Hennig 2007): draw a new sample of the same size by picking
    children at random with replacement, so some children appear twice and
    some not at all. Refit the clustering on that sample. For each original
    class, ask which refit cluster overlaps it most, and measure that overlap
    with the Jaccard index. A class that is a real feature of the data will
    reappear nearly intact every time; a class that depends on a few
    particular children will not.

    Parameters
    ----------
    X : ndarray, shape (n, J)
        The flag matrix.
    labels_ref : ndarray, shape (n,)
        The original clustering of these children.
    fit : callable
        ``fit(X_boot, n_classes)`` must refit on a resampled matrix and return
        labels for its rows.
    n_classes : int
        Number of classes in ``labels_ref``.
    n_boot : int
        Number of resamples. 200 is a common choice.
    seed : int
        For the resampling draws.
    progress : callable, optional
        Called with the resample number after each one, for logging.
    """
    rng = np.random.default_rng(seed)
    n_children = X.shape[0]
    jaccard = np.zeros((n_classes, n_boot))

    for b in range(n_boot):
        # Resample with replacement: a list of n row indices, some repeated.
        drawn = rng.integers(0, n_children, n_children)
        labels_boot = fit(X[drawn], n_classes)

        # Which original children ended up in each bootstrap cluster?
        # (A child drawn twice counts once; we compare sets of children.)
        present = np.unique(drawn)
        members_boot = {k: np.unique(drawn[labels_boot == k]) for k in range(labels_boot.max() + 1)}

        for k in range(n_classes):
            members_ref = np.intersect1d(np.flatnonzero(labels_ref == k), present)
            if members_ref.size == 0:
                jaccard[k, b] = 0.0
                continue
            # Best overlap with any bootstrap cluster.
            jaccard[k, b] = max(_jaccard_of_sets(members_ref, members) for members in members_boot.values())

        if progress:
            progress(b + 1)

    return StabilityResult(jaccard=jaccard)


# --------------------------------------------------------------------------- #
# Permutation null: what does "no structure" look like?
# --------------------------------------------------------------------------- #
def shuffle_each_column(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """Shuffle every column independently, which keeps each condition's prevalence but breaks every link between conditions.

    Fitting the latent class model to this shuffled matrix shows how low BIC
    can go when there is nothing to find. If the real data's BIC curve looks
    like the shuffled one, the classes are not telling us much.
    """
    rng = np.random.default_rng(seed)
    shuffled = X.copy()
    for column in range(shuffled.shape[1]):
        rng.shuffle(shuffled[:, column])
    return shuffled


# --------------------------------------------------------------------------- #
# 4. Count-vs-class test on the held-out outcome
# --------------------------------------------------------------------------- #
SEVERITY_ORDER = ["mild", "moderate", "severe"]


def severity_count_vs_class(
    table: pd.DataFrame,
    labels: np.ndarray,
    flags: list[str],
    n_folds: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Do the classes predict parent-rated severity better than counting conditions?

    This is the project's planned null result. A skeptic would say: of course
    children with more conditions are rated more severe, and your "classes"
    are just low, medium and high condition counts with new names. To answer
    that, we fit three models of severity (mild / moderate / severe), each
    adjusted for age and sex:

    - ``count``: severity ~ number of conditions
    - ``class``: severity ~ which class the child is in
    - ``both``: severity ~ count and class together

    If ``class`` and ``both`` are no better than ``count``, the classes carry
    no information beyond the count, and we say so. The outcome is ordered,
    so the models are ordinal logistic regressions (statsmodels
    ``OrderedModel``).

    Two scores are reported for each model, both lower-is-better:

    - BIC, as for the latent class model.
    - Cross-validated log-loss: fit on four fifths of the children, predict the
      severity of the remaining fifth, and record how surprised the model was
      (negative log probability of the true answer); rotate through all five
      fifths and average. This guards against a model looking good only
      because it has more parameters.

    Children whose parent did not rate severity are left out.
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    frame = table.copy()
    frame["cls"] = labels
    frame = frame.dropna(subset=["severity"]).reset_index(drop=True)

    # The outcome, as an ordered categorical so statsmodels knows mild < moderate < severe.
    severity = pd.Series(pd.Categorical(frame["severity"], categories=SEVERITY_ORDER, ordered=True), name="severity")

    # Predictors.
    frame["count"] = frame[flags].sum(axis=1)
    frame["female"] = (frame["sex"] == "female").astype(float)
    frame["age_c"] = frame["age"] - frame["age"].mean()  # centered age, so the intercepts are interpretable
    class_dummies = pd.get_dummies(frame["cls"], prefix="class", drop_first=True).astype(float)
    adjustments = frame[["age_c", "female"]]

    designs = {
        "count": pd.concat([adjustments, frame[["count"]]], axis=1),
        "class": pd.concat([adjustments, class_dummies], axis=1),
        "both": pd.concat([adjustments, frame[["count"]], class_dummies], axis=1),
    }

    # Assign every child to one of n_folds groups at random for cross-validation.
    rng = np.random.default_rng(seed)
    fold_of_child = rng.permutation(len(frame)) % n_folds

    results = []
    for name, design in designs.items():
        # Full fit, for BIC.
        full_fit = OrderedModel(severity, design, distr="logit").fit(method="bfgs", disp=False, maxiter=500)

        # Cross-validated log-loss.
        fold_losses = []
        for fold in range(n_folds):
            train, test = fold_of_child != fold, fold_of_child == fold
            fold_fit = OrderedModel(
                severity[train].reset_index(drop=True),
                design[train].reset_index(drop=True),
                distr="logit",
            ).fit(method="bfgs", disp=False, maxiter=500)
            # Predicted probability of each severity level for the test children, shape (n_test, 3).
            predicted = np.asarray(fold_fit.model.predict(fold_fit.params, exog=design[test].to_numpy()))
            true_level = severity.cat.codes.to_numpy()[test]
            probability_of_truth = predicted[np.arange(len(true_level)), true_level]
            fold_losses.append(-np.log(np.clip(probability_of_truth, 1e-12, 1)).mean())

        results.append(
            {
                "model": name,
                "n_params": len(full_fit.params),
                "log_likelihood": float(full_fit.llf),
                "bic": float(full_fit.bic),
                "cv_log_loss": float(np.mean(fold_losses)),
            }
        )
    return pd.DataFrame(results)
