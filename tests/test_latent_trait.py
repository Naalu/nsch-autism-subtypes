"""The continuous model: does it recover planted slopes, and is its likelihood on the right scale?"""

import numpy as np

from nsch_subtypes.latent_trait import LatentTraitResult, fit_latent_trait, simulate_latent_trait


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
