"""The continuous model: does it recover planted slopes, and is its likelihood on the right scale?"""

import numpy as np

from nsch_subtypes.latent_trait import (
    LatentTraitResult,
    draw_positions,
    fit_latent_density,
    fit_latent_trait,
    latent_trait_log_likelihood,
    latent_trait_pair_probabilities,
    simulate_latent_trait,
)


def planted_model(seed: int = 1, n_items: int = 12) -> LatentTraitResult:
    """Two dimensions, each driving half the items, with varied intercepts."""
    rng = np.random.default_rng(seed)
    slopes = np.zeros((n_items, 2))
    half = n_items // 2
    slopes[:half, 0] = rng.uniform(1.0, 2.0, half)
    slopes[half:, 1] = rng.uniform(1.0, 2.0, n_items - half)
    intercepts = rng.uniform(-1.5, 0.5, n_items)
    return LatentTraitResult(2, slopes, intercepts, np.zeros((1, 2)), 0.0, 0, True)


def test_two_dimensional_fit_recovers_planted_structure():
    truth = planted_model()
    X = simulate_latent_trait(truth, n=4000, seed=2)
    fit = fit_latent_trait(X, n_dims=2, seed=0)
    assert fit.converged and fit.scores.shape == (4000, 2)

    # Dimensions may come back in either order; match by which items load on them.
    half = truth.slopes.shape[0] // 2
    first_dim_is_first_half = np.abs(fit.slopes[:half, 0]).mean() > np.abs(fit.slopes[:half, 1]).mean()
    fitted = fit.slopes if first_dim_is_first_half else fit.slopes[:, ::-1]
    assert np.abs(fitted - truth.slopes).max() < 0.3, "slopes recovered to within 0.3"
    assert np.abs(fit.intercepts - truth.intercepts).max() < 0.25


def test_two_dimensions_beat_one_on_two_dimensional_data():
    X = simulate_latent_trait(planted_model(), n=3000, seed=3)
    one = fit_latent_trait(X, n_dims=1, seed=0)
    two = fit_latent_trait(X, n_dims=2, seed=0)
    assert two.log_likelihood > one.log_likelihood
    assert two.bic() < one.bic()
    assert two.n_parameters == 12 * 3 - 1 and one.n_parameters == 12 * 2


def test_likelihood_is_on_the_same_scale_as_a_latent_class_model():
    """With no latent structure at all (zero slopes) the model reduces to independent items, which has a closed form."""
    rng = np.random.default_rng(0)
    p = np.array([0.2, 0.5, 0.8])
    X = (rng.random((2000, 3)) < p).astype(np.uint8)
    fit = fit_latent_trait(X, n_dims=1, seed=0)
    column_means = X.mean(axis=0)
    independent_ll = (X * np.log(column_means) + (1 - X) * np.log(1 - column_means)).sum()
    # The trait model can only do as well or slightly better than independence here.
    assert fit.log_likelihood >= independent_ll - 1.0
    assert fit.log_likelihood <= independent_ll + 15.0


def test_scoring_function_agrees_with_the_fit_and_rotation_is_orthogonal():
    """The held-out scorer must give the same number as the fit on the fitting data, or step 13 compares apples to oranges."""
    X = simulate_latent_trait(planted_model(seed=4, n_items=8), n=1500, seed=5)
    fit = fit_latent_trait(X, n_dims=2, seed=0)
    assert fit.converged
    assert np.isclose(latent_trait_log_likelihood(fit, X), fit.log_likelihood, atol=1e-6 * abs(fit.log_likelihood))
    assert latent_trait_log_likelihood(fit, X, per_child=True).shape == (1500,)
    assert np.allclose(fit.rotation @ fit.rotation.T, np.eye(2))


def test_pair_probabilities_match_simulated_frequencies():
    truth = planted_model(seed=6, n_items=6)
    single, joint = latent_trait_pair_probabilities(truth)
    X = simulate_latent_trait(truth, n=40000, seed=7).astype(float)
    assert np.abs(single - X.mean(axis=0)).max() < 0.01
    assert np.abs(joint - X.T @ X / len(X)).max() < 0.01


def test_estimated_density_stays_close_to_normal_when_the_truth_is_normal():
    truth = planted_model(seed=8, n_items=10)
    X = simulate_latent_trait(truth, n=9000, seed=9)  # about our sample size; smaller samples overshoot in the tails
    density = fit_latent_density(truth, X, n_points=21)
    assert np.isclose(density.weights.sum(), 1.0)
    mean = density.weights @ density.nodes
    cov = (density.nodes - mean).T @ ((density.nodes - mean) * density.weights[:, None])
    assert np.abs(mean).max() < 0.1 and np.abs(cov - np.eye(2)).max() < 0.15
    assert 0 <= density.improvement < 25


def test_estimated_density_finds_two_lumps_when_positions_come_in_two_groups():
    truth = planted_model(seed=8, n_items=10)
    rng = np.random.default_rng(10)
    theta = np.where(rng.random(3000)[:, None] < 0.5, [-1.5, 0.0], [1.5, 0.0]) + 0.2 * rng.standard_normal((3000, 2))
    p = 1 / (1 + np.exp(-(theta @ truth.slopes.T + truth.intercepts)))
    X = (rng.random(p.shape) < p).astype(np.uint8)
    density = fit_latent_density(truth, X, n_points=21)
    near_center = np.abs(density.nodes[:, 0]) < 0.6
    assert density.weights[near_center].sum() < 0.1, "little mass between the two groups"
    assert density.improvement > 40


def test_drawn_positions_track_the_true_positions():
    truth = planted_model(seed=8, n_items=10)
    rng = np.random.default_rng(11)
    theta = rng.standard_normal((2000, 2))
    p = 1 / (1 + np.exp(-(theta @ truth.slopes.T + truth.intercepts)))
    X = (rng.random(p.shape) < p).astype(np.uint8)
    draws = draw_positions(truth, X, n_draws=3, seed=0, n_points=31)
    assert draws.shape == (2000, 3, 2)
    assert np.corrcoef(draws[:, 0, 0], theta[:, 0])[0, 1] > 0.6
    assert abs(draws[..., 0].std() - 1) < 0.15, "draws keep the population spread"
