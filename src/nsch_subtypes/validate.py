"""Checks that the classes have to pass: agreement, replication, stability, and the count test.

Pass lines (fixed before any model was fit; see README):

- ARI >= 0.65 between two labelings of the same children (Steinley 2004's
  floor for "moderate" agreement).
- Mean bootstrap Jaccard >= 0.75 for a class to count as stable (Hennig 2007).
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
# Agreement between two labelings
# --------------------------------------------------------------------------- #
def ari(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted Rand index (Hubert & Arabie 1985); 1 = identical, ~0 = unrelated."""
    return float(adjusted_rand_score(labels_a, labels_b))


def match_classes(rates_a: np.ndarray, rates_b: np.ndarray) -> np.ndarray:
    """Match classes of model B to classes of model A by profile distance.

    Returns ``perm`` such that class ``perm[k]`` of B corresponds to class ``k``
    of A (Hungarian assignment on Euclidean distance between rate vectors).
    Both models must have the same number of classes.
    """
    cost = np.linalg.norm(rates_a[:, None, :] - rates_b[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(len(rows), dtype=int)
    perm[rows] = cols
    return perm


def relabel(labels_b: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Rewrite B's labels in A's numbering, given ``perm`` from :func:`match_classes`."""
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(len(perm))
    return inverse[labels_b]


def profile_distance(rates_a: np.ndarray, rates_b: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Mean absolute difference in condition rates per matched class."""
    return np.abs(rates_a - rates_b[perm]).mean(axis=1)


# --------------------------------------------------------------------------- #
# Applying a fitted model to new children
# --------------------------------------------------------------------------- #
def posterior(X: np.ndarray, class_sizes: np.ndarray, condition_rates: np.ndarray) -> np.ndarray:
    """Class probabilities for new children under a fixed latent class model.

    Used by the held-out checks to label children with a model fit elsewhere.
    Computed in log space; rows sum to 1.
    """
    X = np.asarray(X, dtype=float)
    theta = np.clip(np.asarray(condition_rates, dtype=float), 1e-6, 1 - 1e-6)
    log_lik = X @ np.log(theta).T + (1 - X) @ np.log(1 - theta).T  # (n, K)
    log_post = log_lik + np.log(np.asarray(class_sizes, dtype=float))[None, :]
    log_post -= log_post.max(axis=1, keepdims=True)
    post = np.exp(log_post)
    return post / post.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Bootstrap stability (Hennig 2007)
# --------------------------------------------------------------------------- #
@dataclass
class StabilityResult:
    jaccard: np.ndarray  # (K, n_boot)

    @property
    def mean(self) -> np.ndarray:
        return self.jaccard.mean(axis=1)

    def verdicts(self) -> list[str]:
        out = []
        for m in self.mean:
            if m >= 0.85:
                out.append("highly stable")
            elif m >= JACCARD_STABLE:
                out.append("stable")
            elif m >= 0.6:
                out.append("pattern, membership doubtful")
            elif m > JACCARD_DISSOLVED:
                out.append("not trustworthy")
            else:
                out.append("dissolved")
        return out


def _jaccard_sets(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.intersect1d(a, b).size
    union = np.union1d(a, b).size
    return inter / union if union else 0.0


def bootstrap_stability(
    X: np.ndarray,
    labels_ref: np.ndarray,
    fit: Callable[[np.ndarray, int], np.ndarray],
    n_classes: int,
    n_boot: int = 200,
    seed: int = 0,
    progress: Callable[[int], None] | None = None,
) -> StabilityResult:
    """Cluster-wise bootstrap stability.

    For each resample (drawn with replacement, same size), refit with ``fit``
    and, for every reference class, take the best Jaccard overlap between its
    membership and any bootstrap class, computed over the distinct children
    present in the resample. ``fit(X_boot, n_classes)`` must return labels.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    out = np.zeros((n_classes, n_boot))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        labels_b = fit(X[idx], n_classes)
        present = np.unique(idx)
        in_b = {}
        for k in range(labels_b.max() + 1):
            in_b[k] = np.unique(idx[labels_b == k])
        for k in range(n_classes):
            members = np.intersect1d(np.flatnonzero(labels_ref == k), present)
            if members.size == 0:
                out[k, b] = 0.0
                continue
            out[k, b] = max(_jaccard_sets(members, in_b[j]) for j in in_b)
        if progress:
            progress(b + 1)
    return StabilityResult(jaccard=out)


# --------------------------------------------------------------------------- #
# Permutation null: shuffle each column independently
# --------------------------------------------------------------------------- #
def permute_columns(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """Shuffle every column on its own: same prevalences, no links between flags."""
    rng = np.random.default_rng(seed)
    Xp = X.copy()
    for j in range(Xp.shape[1]):
        rng.shuffle(Xp[:, j])
    return Xp


# --------------------------------------------------------------------------- #
# Count-vs-class test on the held-out outcome
# --------------------------------------------------------------------------- #
SEVERITY_ORDER = ["mild", "moderate", "severe"]


def count_vs_class(
    table: pd.DataFrame,
    labels: np.ndarray,
    flags: list[str],
    n_folds: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Does class membership predict parent-rated severity beyond a count of conditions?

    Fits three ordinal logistic models (severity ~ age + sex + ...):
    ``count`` (number of flags), ``class`` (class indicators), and ``both``.
    Reports BIC and 5-fold cross-validated log-loss per child. Lower is better
    for both. If ``class`` and ``both`` do not beat ``count``, the classes are
    a gradient in disguise, and that is the finding.
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    df = table.copy()
    df["cls"] = labels
    df = df.dropna(subset=["severity"]).reset_index(drop=True)
    y = pd.Series(pd.Categorical(df["severity"], categories=SEVERITY_ORDER, ordered=True), name="severity")
    df["count"] = df[flags].sum(axis=1)
    df["female"] = (df["sex"] == "female").astype(float)
    df["age_c"] = df["age"] - df["age"].mean()
    class_dummies = pd.get_dummies(df["cls"], prefix="class", drop_first=True).astype(float)
    base = df[["age_c", "female"]]
    designs = {
        "count": pd.concat([base, df[["count"]]], axis=1),
        "class": pd.concat([base, class_dummies], axis=1),
        "both": pd.concat([base, df[["count"]], class_dummies], axis=1),
    }
    rng = np.random.default_rng(seed)
    folds = rng.permutation(len(df)) % n_folds
    rows = []
    for name, Xd in designs.items():
        model = OrderedModel(y, Xd, distr="logit")
        res = model.fit(method="bfgs", disp=False, maxiter=500)
        losses = []
        for f in range(n_folds):
            tr, te = folds != f, folds == f
            m = OrderedModel(y[tr].reset_index(drop=True), Xd[tr].reset_index(drop=True), distr="logit")
            r = m.fit(method="bfgs", disp=False, maxiter=500)
            probs = np.asarray(r.model.predict(r.params, exog=Xd[te].to_numpy()))
            codes = y.cat.codes.to_numpy()[te]
            losses.append(-np.log(np.clip(probs[np.arange(len(codes)), codes], 1e-12, 1)).mean())
        rows.append(
            {
                "model": name,
                "n_params": len(res.params),
                "log_likelihood": float(res.llf),
                "bic": float(res.bic),
                "cv_log_loss": float(np.mean(losses)),
            }
        )
    return pd.DataFrame(rows)
