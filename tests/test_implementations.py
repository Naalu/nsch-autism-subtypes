"""The contract the clustering methods have to meet, run against both backends.

The shared contract tests run against both library versions (``reference``)
and from-scratch code (``ours``). Missing implementations fail rather than skip.
Additional checks cover numerical edge cases in the NumPy implementations.

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


def test_scratch_jaccard_handles_zero_rows_and_binary_dtypes():
    from nsch_subtypes.hierarchical import jaccard_distances

    X = np.array([[0, 0], [0, 0], [1, 0], [1, 1]], dtype=bool)
    expected = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0.5], [1, 1, 0.5, 0]])
    assert np.allclose(jaccard_distances(X), expected)


def test_scratch_lca_scores_returned_parameters_even_at_iteration_limit():
    from nsch_subtypes.lca import fit_lca
    from nsch_subtypes.validate import class_model_log_likelihood, class_probabilities

    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    # Check both the direct one-class solution and an EM fit stopped early.
    for k in (1, 2):
        fit = fit_lca(X, k, n_starts=2, max_iter=1)
        assert np.isfinite(fit.log_likelihood)
        assert np.allclose(fit.responsibilities, class_probabilities(X, fit.class_sizes, fit.condition_rates))
        assert np.isclose(fit.log_likelihood, class_model_log_likelihood(X, fit.class_sizes, fit.condition_rates))
        again = fit_lca(X, k, n_starts=2, max_iter=1)
        assert np.array_equal(fit.condition_rates, again.condition_rates)


def test_scratch_hierarchical_singleton_and_exact_cluster_count():
    from nsch_subtypes.hierarchical import cluster_hierarchical

    single = cluster_hierarchical(np.array([[0, 1]]), 1)
    assert single.labels.tolist() == [0] and single.merges.shape == (0, 4)
    # Identical rows tie at every merge, but the cut must still give exactly K groups.
    X = np.zeros((4, 2))
    for k in range(1, 5):
        result = cluster_hierarchical(X, k)
        assert len(np.unique(result.labels)) == k


def test_scratch_kmeans_inertia_matches_returned_labels_and_centers():
    from nsch_subtypes.partition import cluster_kmeans

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1]])
    fit = cluster_kmeans(X, 3, max_iter=1)
    assert np.isclose(fit.inertia, ((X - fit.centers[fit.labels]) ** 2).sum())
    assert np.array_equal(fit.labels, cluster_kmeans(X, 3, max_iter=1).labels)


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
    fit = get_backend(backend)["lca"](flag_matrix, 3, n_starts=5, seed=0)
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
    fit = get_backend(backend)["lca"](flag_matrix, 1, n_starts=1, seed=0)
    column_means = flag_matrix.mean(axis=0)
    expected_log_likelihood = (flag_matrix * np.log(column_means) + (1 - flag_matrix) * np.log(1 - column_means)).sum()
    assert np.isclose(fit.log_likelihood, expected_log_likelihood, atol=1.0)
    assert np.allclose(fit.condition_rates[0], column_means, atol=1e-3)


def test_our_lca_agrees_with_stepmix(data):
    flag_matrix, _, _ = data
    ours = get_backend("ours")["lca"](flag_matrix, 3, n_starts=10, seed=0)
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
    result = get_backend(backend)["hierarchical"](small, 3)

    assert result.labels.shape == (300,) and set(np.unique(result.labels)) == {0, 1, 2}
    assert result.merges.shape == (299, 4), "n - 1 merges, four columns"
    assert np.all(np.diff(result.merges[:, 2]) >= -1e-12), "merge distances never decrease"
    assert result.merges[-1, 3] == 300, "the last merge contains everyone"


def test_our_hierarchical_agrees_with_scipy_without_distance_ties():
    # Short binary profiles have ties whose valid resolutions can differ.
    # Many items make distinct Jaccard distances possible without perturbing X.
    from scipy.spatial.distance import pdist

    small = np.random.default_rng(1).integers(0, 2, size=(20, 4096))
    distances = pdist(small, metric="jaccard")
    assert len(np.unique(distances)) == len(distances)
    ours = get_backend("ours")["hierarchical"](small, 4)
    theirs = reference.cluster_scipy(small, 4)
    assert adjusted_rand_index(ours.labels, theirs.labels) > 0.99


def test_our_hierarchical_tied_merges_minimize_average_member_distance(data):
    from nsch_subtypes.hierarchical import jaccard_distances

    X = data[0][:60]
    distances = jaccard_distances(X)
    result = get_backend("ours")["hierarchical"](X, 4)
    groups = {i: [i] for i in range(len(X))}
    for merge_index, (group_a, group_b, merge_height, merged_size) in enumerate(result.merges):
        # Independently recompute distances from original members. This checks
        # the definition of average linkage rather than repeating its update formula.
        active_group_ids = list(groups)
        candidate_distances = []
        for position, first_group in enumerate(active_group_ids):
            for second_group in active_group_ids[position + 1 :]:
                member_pairs = np.ix_(groups[first_group], groups[second_group])
                candidate_distances.append(distances[member_pairs].mean())
        minimum_distance = min(candidate_distances)

        # Any pair tied at the minimum is valid, but the chosen pair's
        # reported height and membership count must both be correct.
        group_a, group_b = int(group_a), int(group_b)
        chosen_member_pairs = np.ix_(groups[group_a], groups[group_b])
        assert np.isclose(merge_height, minimum_distance)
        assert np.isclose(merge_height, distances[chosen_member_pairs].mean())
        merged_members = groups.pop(group_a) + groups.pop(group_b)
        assert merged_size == len(merged_members)
        groups[len(X) + merge_index] = merged_members


def test_our_hierarchical_treats_weights_like_repeated_rows(data):
    """Clustering distinct rows with counts must give the same answer as clustering the repeated rows."""
    flag_matrix, _, _ = data
    unique_rows, counts = np.unique(flag_matrix[:200], axis=0, return_counts=True)
    cluster = get_backend("ours")["hierarchical"]
    weighted = cluster(unique_rows, 3, weights=counts)
    repeated = cluster(np.repeat(unique_rows, counts, axis=0), 3)
    assert adjusted_rand_index(np.repeat(weighted.labels, counts), repeated.labels) > 0.99


def test_weighted_hierarchical_tree_can_be_plotted_without_changing_weights(tmp_path):
    from nsch_subtypes.hierarchical import cluster_hierarchical
    from nsch_subtypes.plots import dendrogram_plot

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    fit = cluster_hierarchical(X, 2, weights=np.array([10, 3, 5, 2]))
    # Plotting needs leaf counts; the fitted tree must retain child weights.
    original = fit.merges.copy()
    path = tmp_path / "weighted_tree.png"
    dendrogram_plot(fit.merges, 2, path)
    assert path.exists() and path.stat().st_size > 0
    assert np.array_equal(fit.merges, original)
    assert fit.merges[-1, 3] == 20


# ---------------------------------------------------------------------------- #
# k-means
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend", BACKENDS)
def test_kmeans_recovers_the_planted_classes(backend, data):
    flag_matrix, true_labels, _ = data
    result = get_backend(backend)["kmeans"](flag_matrix, 3, n_starts=5, seed=0)
    assert result.labels.shape == (flag_matrix.shape[0],) and result.centers.shape == (3, flag_matrix.shape[1])
    assert result.inertia > 0
    assert adjusted_rand_index(true_labels, result.labels) >= 0.6


def test_our_kmeans_is_close_to_sklearn(data):
    flag_matrix, _, _ = data
    ours = get_backend("ours")["kmeans"](flag_matrix, 3, n_starts=10, seed=0)
    theirs = reference.cluster_kmeans_sklearn(flag_matrix, 3, n_starts=10, seed=0)
    perm = match_classes(theirs.centers, ours.centers)
    assert adjusted_rand_index(relabel_to_match(ours.labels, perm), theirs.labels) >= 0.9


@pytest.mark.parametrize("method", ["lca", "hierarchical", "kmeans"])
def test_scratch_rejects_invalid_input(method):
    function = get_backend("ours")[method]
    invalid_inputs = [
        (np.empty((0, 2)), 1),  # No children.
        (np.array([[0, np.nan]]), 1),  # A missing condition flag.
        (np.array([[0, 2]]), 1),  # A condition flag outside 0/1.
        (np.zeros((2, 2)), 0),  # An impossible number of classes.
    ]
    for flag_matrix, n_groups in invalid_inputs:
        with pytest.raises(ValueError):
            function(flag_matrix, n_groups)


def test_scratch_objectives_improve_with_more_iterations():
    from nsch_subtypes.lca import fit_lca
    from nsch_subtypes.partition import cluster_kmeans

    X = np.random.default_rng(4).integers(0, 2, size=(50, 8))
    # The same seed restarts the same trajectory. Increasing the cap lets
    # us inspect successive rounds without exposing internal algorithm state.
    likelihoods = []
    for iteration_limit in range(1, 15):
        fit = fit_lca(X, 3, n_starts=1, max_iter=iteration_limit)
        likelihoods.append(fit.log_likelihood)

    inertias = []
    for iteration_limit in range(1, 10):
        partition = cluster_kmeans(X, 3, n_starts=1, max_iter=iteration_limit)
        inertias.append(partition.inertia)
    assert np.all(np.diff(likelihoods) >= -1e-8)
    assert np.all(np.diff(inertias) <= 1e-8)
