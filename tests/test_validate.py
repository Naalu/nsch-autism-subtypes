"""Tests for the checking tools: class matching, ARI, applying a frozen model, the bootstrap, weighted shares."""

import numpy as np
import pandas as pd
import pytest

from nsch_subtypes.describe import class_profiles, class_shares, design_strata, weighted_mean_with_se
from nsch_subtypes.validate import (
    adjusted_rand_index,
    ari_with_interval,
    bivariate_residuals,
    bootstrap_stability,
    class_model_log_likelihood,
    class_model_pair_probabilities,
    class_probabilities,
    match_classes,
    mean_rate_difference,
    relabel_to_match,
    severity_count_vs_class,
    shuffle_each_column,
)


def test_match_and_relabel_undo_a_shuffled_numbering():
    """If model B is model A with its classes renumbered, matching should recover the renumbering exactly."""
    rates_a = np.array([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]])
    true_perm = np.array([2, 0, 1])
    rates_b = rates_a[true_perm]  # B's class k is A's class true_perm[k]

    perm = match_classes(rates_a, rates_b)
    labels_b = np.array([0, 1, 2, 0])
    assert relabel_to_match(labels_b, perm).tolist() == [true_perm[0], true_perm[1], true_perm[2], true_perm[0]]
    assert np.allclose(mean_rate_difference(rates_a, rates_b, perm), 0)


def test_ari_is_one_for_identical_clusterings_and_near_zero_for_random_ones():
    labels = np.repeat([0, 1, 2], 50)
    assert adjusted_rand_index(labels, (labels + 1) % 3) == 1.0, "renaming the clusters should not matter"
    rng = np.random.default_rng(0)
    assert abs(adjusted_rand_index(labels, rng.integers(0, 3, labels.size))) < 0.1


def test_class_probabilities_sum_to_one_and_favor_the_matching_class():
    rates = np.array([[0.95, 0.05], [0.05, 0.95]])  # class 0 has item 0, class 1 has item 1
    probabilities = class_probabilities(np.array([[1, 0], [0, 1], [1, 1]]), np.array([0.5, 0.5]), rates)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert probabilities[0, 0] > 0.9 and probabilities[1, 1] > 0.9
    assert np.isclose(probabilities[2, 0], 0.5), "a child with both items is a coin flip"


def test_bootstrap_stability_scores_one_when_the_method_always_recovers_the_classes():
    flag_matrix = np.repeat(np.array([[1, 0], [0, 1]], dtype=np.uint8), 40, axis=0)
    labels = flag_matrix[:, 1].astype(int)

    def perfect_method(resampled, n_classes):
        return resampled[:, 1].astype(int)

    result = bootstrap_stability(flag_matrix, labels, fit=perfect_method, n_classes=2, n_boot=10, seed=1)
    assert result.jaccard.shape == (2, 10)
    assert np.allclose(result.mean, 1.0)
    assert result.verdicts() == ["highly stable", "highly stable"]


def test_shuffling_columns_keeps_each_prevalence():
    rng = np.random.default_rng(0)
    flag_matrix = (rng.random((200, 5)) < 0.3).astype(np.uint8)
    shuffled = shuffle_each_column(flag_matrix, seed=0)
    assert np.array_equal(flag_matrix.sum(axis=0), shuffled.sum(axis=0)), "column totals unchanged"
    assert not np.array_equal(flag_matrix, shuffled), "but the rows are different"


def test_weighted_mean_with_se_matches_a_hand_calculation():
    values = np.array([1, 0, 1, 1, 0, 0], dtype=float)
    weights = np.array([2, 1, 1, 3, 1, 2], dtype=float)
    strata = np.array([0, 0, 0, 1, 1, 1])

    mean, se = weighted_mean_with_se(values, weights, strata)
    assert np.isclose(mean, (2 + 1 + 3) / 10)

    scores = weights * (values - mean) / weights.sum()
    variance = 0.0
    for stratum in (0, 1):
        in_stratum = scores[strata == stratum]
        variance += len(in_stratum) / (len(in_stratum) - 1) * ((in_stratum - in_stratum.mean()) ** 2).sum()
    assert np.isclose(se, np.sqrt(variance))


def test_class_shares_sum_to_one_and_intervals_contain_the_estimate():
    n = 300
    table = pd.DataFrame({"year": 2020, "state": np.arange(n) % 5, "stratum": "1", "weight": np.linspace(1, 3, n)})
    responsibilities = np.column_stack([np.linspace(0.2, 0.8, n), np.linspace(0.8, 0.2, n)])

    shares = class_shares(table, responsibilities, pooled_years=1)
    assert np.isclose(shares["share"].sum(), 1.0)
    assert (shares["ci_low"] <= shares["share"]).all() and (shares["share"] <= shares["ci_high"]).all()
    assert design_strata(table).max() == 4, "five states, one stratum each, one year"


def test_match_classes_refuses_models_with_different_numbers_of_classes():
    with pytest.raises(ValueError):
        match_classes(np.zeros((3, 4)), np.zeros((4, 4)))


def test_ari_interval_brackets_the_estimate_and_confident_subset_is_reported():
    rng = np.random.default_rng(0)
    truth = np.repeat([0, 1, 2], 200)
    noisy = np.where(rng.random(truth.size) < 0.2, rng.integers(0, 3, truth.size), truth)
    confidence = np.where(noisy == truth, 0.95, 0.5)  # pretend the disagreements are the unsure children
    out = ari_with_interval(truth, noisy, confidence_a=confidence, n_boot=200)
    assert out["ci_low"] <= out["ari"] <= out["ci_high"]
    assert out["ari_confident"] == 1.0
    assert 0.8 < out["share_confident"] < 0.95
    levels = [row["min_posterior"] for row in out["ari_by_confidence"]]
    assert levels == sorted(levels) and all(row["ari"] == 1.0 for row in out["ari_by_confidence"])


def test_one_class_likelihood_is_the_independent_items_formula():
    """With K = 1 a latent class model is just J independent coin flips, which has a closed form."""
    rng = np.random.default_rng(1)
    X = (rng.random((500, 4)) < [0.1, 0.3, 0.5, 0.7]).astype(np.uint8)
    p = X.mean(axis=0)
    by_hand = (X * np.log(p) + (1 - X) * np.log(1 - p)).sum()
    assert np.isclose(class_model_log_likelihood(X, np.array([1.0]), p[None, :]), by_hand)
    assert class_model_log_likelihood(X, np.array([1.0]), p[None, :], per_child=True).shape == (500,)


def test_bivariate_residuals_flag_a_pair_the_model_cannot_explain():
    """Items 0 and 1 are copies of each other; a one-class model (independence) should fail that pair and only that pair."""
    rng = np.random.default_rng(2)
    X = (rng.random((2000, 4)) < 0.3).astype(np.uint8)
    X[:, 1] = X[:, 0]
    single, joint = class_model_pair_probabilities(np.array([1.0]), X.mean(axis=0)[None, :])
    table = bivariate_residuals(X, single, joint, ["a", "b", "c", "d"])
    assert (table.loc[0, "item_a"], table.loc[0, "item_b"]) == ("a", "b")
    assert table.loc[0, "chi_square"] > 100
    assert (table.loc[1:, "chi_square"] < 15).all()


def test_class_profiles_keep_a_row_for_an_empty_class():
    X = np.array([[1, 0], [0, 1], [1, 1]])
    profiles = class_profiles(X, np.array([0, 0, 2]), ["a", "b"], n_classes=3)
    assert len(profiles) == 3
    assert profiles.loc[1, "n"] == 0 and profiles.loc[1, ["a", "b"]].isna().all()
    assert profiles.loc[0, "a"] == 0.5


def test_severity_test_detects_classes_that_carry_information_beyond_the_count():
    """Plant an effect where class 1 is rated more severe than class 0 at the same condition count."""
    rng = np.random.default_rng(3)
    n = 1500
    labels = rng.integers(0, 2, n)
    flags = [f"f{j}" for j in range(4)]
    X = (rng.random((n, 4)) < 0.4).astype(int)
    latent = 0.3 * X.sum(axis=1) + 1.5 * labels + rng.logistic(size=n)
    severity = np.select([latent < 1.0, latent < 2.5], ["mild", "moderate"], "severe")
    table = pd.DataFrame(X, columns=flags).assign(severity=severity, sex=np.where(rng.random(n) < 0.2, "female", "male"), age=rng.integers(6, 18, n))
    table.loc[:20, "severity"] = None  # a few unrated children are dropped, not miscounted

    out = severity_count_vs_class(table, labels, flags, n_folds=3).set_index("model")
    assert out.loc["both", "bic"] < out.loc["count", "bic"]
    assert out.loc["both", "cv_log_loss"] < out.loc["count", "cv_log_loss"]
    assert out.loc["count", "bic_vs_count"] == 0
