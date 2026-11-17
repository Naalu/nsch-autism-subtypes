"""The factor mixture model: does one group reduce to the latent trait model, and are planted groups found?"""

import numpy as np

from nsch_subtypes.factor_mixture import (
    FactorMixtureResult,
    factor_mixture_group_probabilities,
    factor_mixture_log_likelihood,
    fit_factor_mixture,
    simulate_factor_mixture,
)
from nsch_subtypes.latent_trait import fit_latent_trait, simulate_latent_trait
from nsch_subtypes.validate import adjusted_rand_index
from test_latent_trait import planted_model


def planted_groups(separation: float = 3.0) -> FactorMixtureResult:
    """Two equal groups ``separation`` apart along the first axis, on the planted two-dimensional conditions."""
    conditions = planted_model(n_items=16)
    means = np.array([[-separation / 2, 0.0], [separation / 2, 0.0]])
    covariances = np.repeat((0.5**2 * np.eye(2))[None], 2, axis=0)
    empty = np.zeros((1, 2))
    return FactorMixtureResult(2, 2, True, conditions.slopes, conditions.intercepts, np.array([0.5, 0.5]), means, covariances, empty, empty, 0.0, 0, True)


def test_one_group_is_the_latent_trait_model():
    X = simulate_latent_trait(planted_model(), n=1500, seed=4)
    trait = fit_latent_trait(X, n_dims=2, seed=0)
    one_group = fit_factor_mixture(X, 1, trait=trait)
    assert one_group.n_parameters == trait.n_parameters
    # Same model on a different grid, so the likelihoods agree closely but not exactly.
    assert abs(one_group.log_likelihood - trait.log_likelihood) < 3.0
    assert np.allclose(one_group.group_shares, [1.0]) and np.allclose(one_group.group_means, 0.0, atol=1e-8)
    assert np.allclose(one_group.group_covariances[0], np.eye(2), atol=1e-6)


def test_parameter_counts():
    X = simulate_latent_trait(planted_model(), n=500, seed=4)
    trait = fit_latent_trait(X, n_dims=2, seed=0)
    settings = {"trait": trait, "n_starts": 2, "short_iter": 5, "max_iter": 5}
    n_items = X.shape[1]
    conditions = n_items * 3 - 1
    assert fit_factor_mixture(X, 3, shared_spread=True, **settings).n_parameters == conditions + 2 + 4
    assert fit_factor_mixture(X, 3, shared_spread=False, **settings).n_parameters == conditions + 2 + 4 + 6


def test_planted_groups_are_found_and_no_groups_are_not():
    truth = planted_groups()
    X, groups = simulate_factor_mixture(truth, n=2500, seed=5)
    settings = {"n_starts": 2, "short_iter": 40, "max_iter": 300}
    trait = fit_latent_trait(X, n_dims=2, seed=0)
    one, two = fit_factor_mixture(X, 1, trait=trait), fit_factor_mixture(X, 2, trait=trait, **settings)
    assert two.bic() < one.bic() - 50, "two planted groups beat one clearly"
    assert adjusted_rand_index(two.labels, groups) > 0.6
    assert abs(two.group_shares[0] - 0.5) < 0.1

    # Positions over all children are standardized: mean zero, identity covariance.
    overall_mean = two.group_shares @ two.group_means
    overall_cov = np.einsum("k,kde->de", two.group_shares, two.group_covariances) + (two.group_means * two.group_shares[:, None]).T @ two.group_means
    assert np.allclose(overall_mean, 0.0, atol=1e-6) and np.allclose(overall_cov, np.eye(2), atol=1e-6)

    # The same comparison on data with no groups: the second group buys little.
    smooth = simulate_latent_trait(planted_model(n_items=16), n=2500, seed=6)
    smooth_trait = fit_latent_trait(smooth, n_dims=2, seed=0)
    smooth_one = fit_factor_mixture(smooth, 1, trait=smooth_trait)
    smooth_two = fit_factor_mixture(smooth, 2, trait=smooth_trait, **settings)
    assert smooth_two.bic() > smooth_one.bic() - 10


def test_scoring_new_children_matches_the_fit():
    X, _ = simulate_factor_mixture(planted_groups(), n=1000, seed=7)
    fit = fit_factor_mixture(X, 2, n_starts=2, short_iter=30, max_iter=50)
    assert np.isclose(factor_mixture_log_likelihood(fit, X), fit.log_likelihood)
    per_child = factor_mixture_log_likelihood(fit, X, per_child=True)
    assert per_child.shape == (1000,) and np.isclose(per_child.sum(), fit.log_likelihood)
    probabilities = factor_mixture_group_probabilities(fit, X)
    assert np.allclose(probabilities, fit.group_probabilities) and np.allclose(probabilities.sum(axis=1), 1.0)
    assert fit.scores.shape == (1000, 2) and fit.group_shares[0] >= fit.group_shares[1]
