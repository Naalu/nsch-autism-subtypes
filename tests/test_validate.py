"""Class matching, ARI, bootstrap bookkeeping, posterior scoring, weighted shares."""

import numpy as np
import pandas as pd

from nsch_subtypes.describe import class_shares, design_strata, weighted_mean_se
from nsch_subtypes.validate import ari, bootstrap_stability, match_classes, permute_columns, posterior, profile_distance, relabel


def test_match_and_relabel_undo_a_permutation():
    rates = np.array([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]])
    perm_true = np.array([2, 0, 1])
    rates_b = rates[perm_true]  # B's class k is A's class perm_true[k]
    perm = match_classes(rates, rates_b)
    labels_b = np.array([0, 1, 2, 0])
    assert relabel(labels_b, perm).tolist() == [perm_true[0], perm_true[1], perm_true[2], perm_true[0]]
    assert np.allclose(profile_distance(rates, rates_b, perm), 0)


def test_ari_identical_and_random():
    a = np.repeat([0, 1, 2], 50)
    assert ari(a, (a + 1) % 3) == 1.0
    rng = np.random.default_rng(0)
    assert abs(ari(a, rng.integers(0, 3, a.size))) < 0.1


def test_posterior_rows_sum_to_one_and_prefer_matching_class():
    rates = np.array([[0.95, 0.05], [0.05, 0.95]])
    post = posterior(np.array([[1, 0], [0, 1], [1, 1]]), np.array([0.5, 0.5]), rates)
    assert np.allclose(post.sum(axis=1), 1)
    assert post[0, 0] > 0.9 and post[1, 1] > 0.9 and np.isclose(post[2, 0], 0.5)


def test_bootstrap_stability_perfect_method_scores_one():
    X = np.repeat(np.array([[1, 0], [0, 1]], dtype=np.uint8), 40, axis=0)
    labels = X[:, 1].astype(int)
    res = bootstrap_stability(X, labels, fit=lambda Xb, k: Xb[:, 1].astype(int), n_classes=2, n_boot=10, seed=1)
    assert res.jaccard.shape == (2, 10)
    assert np.allclose(res.mean, 1.0)
    assert res.verdicts() == ["highly stable", "highly stable"]


def test_permute_columns_keeps_prevalence():
    rng = np.random.default_rng(0)
    X = (rng.random((200, 5)) < 0.3).astype(np.uint8)
    Xp = permute_columns(X, seed=0)
    assert np.array_equal(X.sum(axis=0), Xp.sum(axis=0)) and not np.array_equal(X, Xp)


def test_weighted_mean_se_matches_hand_calculation():
    y = np.array([1, 0, 1, 1, 0, 0], dtype=float)
    w = np.array([2, 1, 1, 3, 1, 2], dtype=float)
    strata = np.array([0, 0, 0, 1, 1, 1])
    p, se = weighted_mean_se(y, w, strata)
    assert np.isclose(p, (2 + 1 + 3) / 10)
    z = w * (y - p) / w.sum()
    var = sum(len(z[strata == h]) / (len(z[strata == h]) - 1) * ((z[strata == h] - z[strata == h].mean()) ** 2).sum() for h in (0, 1))
    assert np.isclose(se, np.sqrt(var))


def test_class_shares_sum_to_one():
    n = 300
    table = pd.DataFrame({"year": 2020, "state": np.arange(n) % 5, "stratum": "1", "weight": np.linspace(1, 3, n)})
    resp = np.column_stack([np.linspace(0.2, 0.8, n), np.linspace(0.8, 0.2, n)])
    shares = class_shares(table, resp, pooled_years=1)
    assert np.isclose(shares["share"].sum(), 1.0)
    assert (shares["ci_low"] <= shares["share"]).all() and (shares["share"] <= shares["ci_high"]).all()
    assert design_strata(table).max() == 4
