# Implementation spec for the from-scratch methods

What the pipeline expects from `lca.py`, `hierarchical.py` and (if adopted) `partition.py`. The function signatures and result classes are already in those files with docstrings; the bodies raise `NotImplementedError` until replaced. The pipeline runs against library versions (`--backend reference`) in the meantime, and `tests/test_implementations.py` runs the same checks against both.

If the methods themselves are new to you, `docs/background.md` section 3 explains each one in words before the math below. The module docstrings in the three files repeat the key ideas next to the code.

A good order of work: get `fit_lca` passing its tests first (it is what every later step uses), then `jaccard_distances` and `cluster_hierarchical`, then `cluster_kmeans` if we decide to adopt it. Run `uv run pytest tests/test_implementations.py -v` after every change; a test that is still skipping tells you which function is still a stub.

Ground rules:

- NumPy only inside these three files. SciPy, scikit-learn and StepMix are allowed everywhere else and are what we check against.
- Inputs are `numpy` arrays of 0/1 with no missing values; the pipeline guarantees that.
- Determinism: the same `seed` gives the same answer. Use `numpy.random.default_rng(seed)`.
- Don't change the signatures or the result classes. If something in them gets in the way, change the spec first.

## 1. `lca.fit_lca(X, n_classes, n_starts=20, seed=0, tol=1e-6, max_iter=500) -> LCAResult`

Bernoulli mixture (latent class model) fit by EM.

Notation: `n` children, `J` flags, `K` classes. Parameters `pi` (K,) class sizes, `theta` (K, J) condition rates. Responsibilities `r` (n, K).

Per start:

1. Initialize `theta` uniformly in (0.2, 0.8) and `pi = 1/K`.
2. E-step, in log space:
   `log_lik[i, k] = sum_j x[i, j] log theta[k, j] + (1 - x[i, j]) log(1 - theta[k, j])`
   `log_post[i, k] = log pi[k] + log_lik[i, k]`
   `r[i, k] = exp(log_post[i, k] - logsumexp_k(log_post[i, :]))`
   Total log-likelihood for this iteration: `sum_i logsumexp_k(log_post[i, :])`.
3. M-step: `pi[k] = mean_i r[i, k]`; `theta[k, j] = sum_i r[i, k] x[i, j] / sum_i r[i, k]`. Clip `theta` to `[1e-6, 1 - 1e-6]`.
4. Stop when the log-likelihood improves by less than `tol`, or after `max_iter`.

Return the start with the highest final log-likelihood. `K = 1` must work (rates are the column means; no random starts needed).

The result class computes `labels`, `n_parameters = (K - 1) + K J`, `bic(n)` and `entropy()` for you.

Checks the tests apply (`tests/test_implementations.py`):

- Log-likelihood never decreases across iterations (assert inside the loop while developing).
- `class_sizes` sum to 1; every row of `responsibilities` sums to 1; shapes are right.
- On `synthetic.make_synthetic(n=2000, n_items=10, n_classes=3, separation=0.75)`: ARI against the true labels at least 0.8, and the matched rates within 0.05 of the generating rates.
- Against StepMix on the same data with the same K: best log-likelihood within 1.0 and matched rates within 0.02. Use `validate.match_classes` for the matching.
- `K = 1` log-likelihood equals the closed form.

Things that bite: a column that is all 0 or all 1 (the clip handles it); label switching between starts (irrelevant inside `fit_lca`, handled by `validate.match_classes` outside); the `n_starts` loop is the slow part, so vectorize the E-step as matrix products (`X @ log(theta).T`).

## 2. `hierarchical.cluster_hierarchical(X, n_clusters, weights=None) -> HierarchicalResult` and `hierarchical.jaccard_distances(X) -> ndarray`

Agglomerative clustering, Jaccard distance, average linkage.

`jaccard_distances`: for rows `a`, `b`, `d = 1 - |a & b| / |a | b|`; define `d = 0` when both rows are all zero. Return the full (n, n) matrix with zeros on the diagonal. Vectorize with `X @ X.T` for the intersections and row sums for the unions.

`cluster_hierarchical`:

1. Start with `n` groups of one row each, weights `w` (all ones if `weights` is None).
2. Repeat `n - 1` times: find the pair of groups with the smallest distance; merge them; record `[i, j, distance, size]` where `i < j` are the group ids being merged, group ids for new groups count up from `n`, and `size` is the total weight of the new group. Update distances to every other group `C` with the weighted average
   `d(A+B, C) = (w_A d(A, C) + w_B d(B, C)) / (w_A + w_B)`.
3. Labels: undo the last `n_clusters - 1` merges, assign 0..n_clusters-1 to the remaining groups in order of first appearance.

The naive version is O(n^3) with an (n, n) matrix. On this project's data the pipeline collapses the 8,891 complete rows to 1,347 distinct profiles and passes counts as `weights`, so the naive version is fast enough (1,347^2 distances). A nearest-neighbor-chain version is optional.

Checks the tests apply:

- Identical labels to `reference.cluster_scipy` on synthetic data (ARI = 1), allowing for ties in distance.
- Merge distances non-decreasing; `merges` has shape (n - 1, 4) and plots with `scipy.cluster.hierarchy.dendrogram`.
- With `weights`, the same labels as running the unweighted version on the rows repeated `weights` times.

Note from the first runs on real data (October 2026): average linkage on these flags peels off outliers, so cutting to K leaves one cluster with nearly every child. The dendrogram is still a useful figure, but the pipeline also offers k-means as the cross-check (section 3). Which one the paper uses is an open decision; see `docs/validation-checklist.md`.

## 3. `partition.cluster_kmeans(X, n_clusters, n_starts=20, seed=0, max_iter=300) -> PartitionResult` (if adopted)

Lloyd's algorithm on the 0/1 matrix treated as real vectors.

Per start: pick `n_clusters` distinct rows at random as centers; repeat: assign each row to the nearest center by squared Euclidean distance; recompute centers as the mean of their rows (keep a center where it is if its cluster empties); stop when assignments do not change. Inertia = total squared distance to assigned centers. Return the start with the lowest inertia.

Checks: inertia never increases across iterations; on the synthetic data, ARI against `reference.cluster_kmeans_sklearn` at least 0.9 (k-means has local optima, so exact equality is not expected).

## 4. How the pipeline uses them

| Script | Calls | Backend flag |
|---|---|---|
| `03_fit_lca.py` | `fit_lca` for K = 1..8 | `--backend ours` |
| `04_crosscheck.py` | `cluster_hierarchical` (with `weights`) or `cluster_kmeans` | `--backend ours --method ...` |
| `05_replicate.py` | `fit_lca` twice | `--backend ours` |
| `06_bootstrap.py` | `fit_lca` 200 times | `--backend ours` |
| `08_forward_2025.py` | `fit_lca` once, plus `validate.class_probabilities` to apply the frozen model | `--backend ours` |

`validate.class_probabilities(X, class_sizes, condition_rates)` is a small helper that scores new children under a fixed model (the E-step formula with no update). It exists so the held-out checks can label children with a model fit elsewhere; `fit_lca` should have its own E-step.

Run the implementation tests with `uv run pytest tests/test_implementations.py -v`. They skip with a clear message while a function still raises `NotImplementedError`.
