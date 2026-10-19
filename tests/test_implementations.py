"""The contract the clustering methods have to meet, run against both backends.

Every test is run twice, once with the library versions (``reference``) and
once with the from-scratch code (``ours``). While a from-scratch function
still raises ``NotImplementedError`` its tests are skipped, so CI stays green
before the code lands and starts checking it the moment it does.

The data is synthetic (``make_synthetic``), so the true classes are known
and "did the method find them" is a fair question.
"""

import numpy as np
import pytest

from nsch_subtypes import reference
from nsch_subtypes.backends import get_backend
from nsch_subtypes.synthetic import make_synthetic
from nsch_subtypes.validate import adjusted_rand_index, match_classes, relabel_to_match

BACKENDS = ["reference", "ours"]


def call_or_skip(function, *args, **kwargs):
    """Call the function; if it is still a stub, skip the test instead of failing it."""
    try:
        return function(*args, **kwargs)
    except NotImplementedError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def data():
    """2,000 fake children, 10 items, 3 well-separated planted classes."""
    flag_matrix, true_labels, true_rates = make_synthetic(n=2000, n_items=10, n_classes=3, seed=3, separation=0.75)
    return flag_matrix, true_labels, true_rates


# ---------------------------------------------------------------------------- #
# Latent class model
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend", BACKENDS)
def test_lca_returns_the_right_shapes_and_recovers_the_planted_classes(backend, data):
    flag_matrix, true_labels, true_rates = data
    fit = call_or_skip(get_backend(backend)["lca"], flag_matrix, 3, n_starts=5, seed=0)
    n_children, n_items = flag_matrix.shape

    assert fit.class_sizes.shape == (3,) and np.isclose(fit.class_sizes.sum(), 1)
    assert fit.condition_rates.shape == (3, n_items)
    assert fit.responsibilities.shape == (n_children, 3) and np.allclose(fit.responsibilities.sum(axis=1), 1)

    assert adjusted_rand_index(true_labels, fit.labels) >= 0.8, "should mostly recover who is in which class"
    perm = match_classes(true_rates, fit.condition_rates)
    assert np.abs(true_rates - fit.condition_rates[perm]).max() < 0.06, "fitted rates close to the generating rates"
    assert fit.bic() > 0 and 0 <= fit.entropy() <= 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_lca_with_one_class_matches_the_closed_form(backend, data):
    """With K=1 there is nothing to estimate but the column means, and the log-likelihood has a formula."""
    flag_matrix, _, _ = data
    fit = call_or_skip(get_backend(backend)["lca"], flag_matrix, 1, n_starts=1, seed=0)
    column_means = flag_matrix.mean(axis=0)
    expected_log_likelihood = (flag_matrix * np.log(column_means) + (1 - flag_matrix) * np.log(1 - column_means)).sum()
    assert np.isclose(fit.log_likelihood, expected_log_likelihood, atol=1.0)
    assert np.allclose(fit.condition_rates[0], column_means, atol=1e-3)


def test_our_lca_agrees_with_stepmix(data):
    flag_matrix, _, _ = data
    ours = call_or_skip(get_backend("ours")["lca"], flag_matrix, 3, n_starts=10, seed=0)
    theirs = reference.fit_lca_stepmix(flag_matrix, 3, n_starts=10, seed=0)
    assert abs(ours.log_likelihood - theirs.log_likelihood) < 1.0
    perm = match_classes(theirs.condition_rates, ours.condition_rates)
    assert np.abs(theirs.condition_rates - ours.condition_rates[perm]).max() < 0.02


# ---------------------------------------------------------------------------- #
# Hierarchical clustering
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend", BACKENDS)
def test_hierarchical_returns_labels_and_a_well_formed_tree(backend, data):
    flag_matrix, _, _ = data
    small = flag_matrix[:300]
    result = call_or_skip(get_backend(backend)["hierarchical"], small, 3)

    assert result.labels.shape == (300,) and set(np.unique(result.labels)) == {0, 1, 2}
    assert result.merges.shape == (299, 4), "n - 1 merges, four columns"
    assert np.all(np.diff(result.merges[:, 2]) >= -1e-12), "merge distances never decrease"
    assert result.merges[-1, 3] == 300, "the last merge contains everyone"


def test_our_hierarchical_agrees_with_scipy(data):
    flag_matrix, _, _ = data
    small = flag_matrix[:300]
    ours = call_or_skip(get_backend("ours")["hierarchical"], small, 4)
    theirs = reference.cluster_scipy(small, 4)
    assert adjusted_rand_index(ours.labels, theirs.labels) > 0.99


def test_our_hierarchical_treats_weights_like_repeated_rows(data):
    """Clustering distinct rows with counts must give the same answer as clustering the repeated rows."""
    flag_matrix, _, _ = data
    unique_rows, counts = np.unique(flag_matrix[:200], axis=0, return_counts=True)
    cluster = get_backend("ours")["hierarchical"]
    weighted = call_or_skip(cluster, unique_rows, 3, weights=counts)
    repeated = cluster(np.repeat(unique_rows, counts, axis=0), 3)
    assert adjusted_rand_index(np.repeat(weighted.labels, counts), repeated.labels) > 0.99


# ---------------------------------------------------------------------------- #
# k-means
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend", BACKENDS)
def test_kmeans_recovers_the_planted_classes(backend, data):
    flag_matrix, true_labels, _ = data
    result = call_or_skip(get_backend(backend)["kmeans"], flag_matrix, 3, n_starts=5, seed=0)
    assert result.labels.shape == (flag_matrix.shape[0],) and result.centers.shape == (3, flag_matrix.shape[1])
    assert result.inertia > 0
    assert adjusted_rand_index(true_labels, result.labels) >= 0.6


def test_our_kmeans_is_close_to_sklearn(data):
    flag_matrix, _, _ = data
    ours = call_or_skip(get_backend("ours")["kmeans"], flag_matrix, 3, n_starts=10, seed=0)
    theirs = reference.cluster_kmeans_sklearn(flag_matrix, 3, n_starts=10, seed=0)
    perm = match_classes(theirs.centers, ours.centers)
    assert adjusted_rand_index(relabel_to_match(ours.labels, perm), theirs.labels) >= 0.9
