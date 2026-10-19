"""Contract tests for the clustering methods, run against both backends.

For ``ours`` the tests skip while a function still raises NotImplementedError,
so CI passes before the from-scratch code lands and starts checking it the
moment it does.
"""

import numpy as np
import pytest

from nsch_subtypes import reference
from nsch_subtypes.backends import get_backend
from nsch_subtypes.synthetic import make_synthetic
from nsch_subtypes.validate import ari, match_classes, relabel

BACKENDS = ["reference", "ours"]


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def data():
    X, labels, rates = make_synthetic(n=2000, n_items=10, n_classes=3, seed=3, separation=0.75)
    return X, labels, rates


@pytest.mark.parametrize("backend", BACKENDS)
def test_lca_shapes_and_recovery(backend, data):
    X, labels, rates = data
    res = _call(get_backend(backend)["lca"], X, 3, n_starts=5, seed=0)
    n, j = X.shape
    assert res.class_sizes.shape == (3,) and np.isclose(res.class_sizes.sum(), 1)
    assert res.condition_rates.shape == (3, j)
    assert res.responsibilities.shape == (n, 3) and np.allclose(res.responsibilities.sum(axis=1), 1)
    assert ari(labels, res.labels) >= 0.8
    perm = match_classes(rates, res.condition_rates)
    assert np.abs(rates - res.condition_rates[perm]).max() < 0.06
    assert res.bic() > 0 and 0 <= res.entropy() <= 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_lca_one_class_is_closed_form(backend, data):
    X, _, _ = data
    res = _call(get_backend(backend)["lca"], X, 1, n_starts=1, seed=0)
    p = X.mean(axis=0)
    ll = (X * np.log(p) + (1 - X) * np.log(1 - p)).sum()
    assert np.isclose(res.log_likelihood, ll, atol=1.0)
    assert np.allclose(res.condition_rates[0], p, atol=1e-3)


def test_ours_lca_matches_stepmix(data):
    X, _, _ = data
    ours = _call(get_backend("ours")["lca"], X, 3, n_starts=10, seed=0)
    ref = reference.fit_lca_stepmix(X, 3, n_starts=10, seed=0)
    assert abs(ours.log_likelihood - ref.log_likelihood) < 1.0
    perm = match_classes(ref.condition_rates, ours.condition_rates)
    assert np.abs(ref.condition_rates - ours.condition_rates[perm]).max() < 0.02


@pytest.mark.parametrize("backend", BACKENDS)
def test_hierarchical_shapes_and_tree(backend, data):
    X, labels, _ = data
    Xs = X[:300]
    res = _call(get_backend(backend)["hierarchical"], Xs, 3)
    assert res.labels.shape == (300,) and set(np.unique(res.labels)) == {0, 1, 2}
    assert res.merges.shape == (299, 4)
    assert np.all(np.diff(res.merges[:, 2]) >= -1e-12)
    assert res.merges[-1, 3] == 300


def test_ours_hierarchical_matches_scipy(data):
    X, _, _ = data
    Xs = X[:300]
    ours = _call(get_backend("ours")["hierarchical"], Xs, 4)
    ref = reference.cluster_scipy(Xs, 4)
    assert ari(ours.labels, ref.labels) > 0.99


def test_ours_hierarchical_weights_equal_repeats(data):
    X, _, _ = data
    uniq, counts = np.unique(X[:200], axis=0, return_counts=True)
    cluster = get_backend("ours")["hierarchical"]
    weighted = _call(cluster, uniq, 3, weights=counts)
    repeated = cluster(np.repeat(uniq, counts, axis=0), 3)
    assert ari(np.repeat(weighted.labels, counts), repeated.labels) > 0.99


@pytest.mark.parametrize("backend", BACKENDS)
def test_kmeans_recovery(backend, data):
    X, labels, _ = data
    res = _call(get_backend(backend)["kmeans"], X, 3, n_starts=5, seed=0)
    assert res.labels.shape == (X.shape[0],) and res.centers.shape == (3, X.shape[1])
    assert res.inertia > 0
    assert ari(labels, res.labels) >= 0.6


def test_ours_kmeans_close_to_sklearn(data):
    X, _, _ = data
    ours = _call(get_backend("ours")["kmeans"], X, 3, n_starts=10, seed=0)
    ref = reference.cluster_kmeans_sklearn(X, 3, n_starts=10, seed=0)
    perm = match_classes(ref.centers, ours.centers)
    assert ari(relabel(ours.labels, perm), ref.labels) >= 0.9
