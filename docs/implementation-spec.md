# Implementation spec for the from-scratch methods

`Note`: Implementation completed Oct. 31, 2026 

The from-scratch methods live in `lca.py`, `hierarchical.py`, and `partition.py`.
All four functions are implemented using NumPy. Their existing signatures and
result classes are shared with the library wrappers through `backends.py`.

Use `--backend ours` to select them or `--backend reference` to select the
library versions. Run `python -m pytest tests/test_implementations.py -v` for
contract and comparison checks, and `python -m pytest -m slow -v` to exercise
the full pipeline at small settings with both backends. These smoke settings
check integration, not the final scientific results.

Ground rules:

- NumPy only inside these three files. SciPy, scikit-learn and StepMix are allowed everywhere else and are what we check against.
- Inputs are `numpy` arrays of 0/1 with no missing values; the pipeline guarantees that.
- Determinism: the same `seed` gives the same answer. Use `numpy.random.default_rng(seed)`.
- Don't change the signatures or the result classes. If something in them gets in the way, change the spec first.

## Running the implementations

From the repository root, use `uv` to run the tests (this works on Windows, macOS and Linux as long as `uv` is installed):

```bash
uv run pytest -q -m "not slow"
uv run pytest -m slow -v
```

### Windows alternative

If the dependencies are already installed in a local `.venv`, you can run its Python executable directly instead:

```powershell
.venv/Scripts/python.exe -m pytest -q -m "not slow"
.venv/Scripts/python.exe -m pytest -m slow -v
.venv/Scripts/python.exe scripts/03_fit_lca.py --n-starts 20 --permutation-null
```

The iteration cap is `lca.MAX_ITER` (2000), shared by step 3, every refit in
steps 5 to 14 and the StepMix reference, so a converged fit means the same
thing everywhere. Step 3 accepts `--max-iter` to override it. With a cap of
500 the K=8 fit on the real data does not converge; at 2000 its best start
converges after about 1,400 iterations. K=1 through K=7 converge well under
500.

The smoke tests exercise steps 3 through 14 and figure generation with both
backends, using reduced starts, resamples, and simulation settings. They read
the committed analysis table; rebuilding the Census download is separate.
They do not replace a full 200-resample analysis sweep. The weighted sensitivity
fit in step 11 and the alternative methods in step 4 still use libraries by design.

## 1. `lca.fit_lca(X, n_classes, n_starts=20, seed=0, tol=1e-6, max_iter=MAX_ITER) -> LCAResult`

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

- Identical labels to `reference.cluster_scipy` on a tie-free binary fixture (ARI = 1). On short binary profiles, equal Jaccard distances can lead to different valid trees; a separate test checks every merge against the minimum average distance between original members.
- Merge distances non-decreasing; `merges` has shape (n - 1, 4). `plots.dendrogram_plot` converts weighted sizes to displayed leaf counts for SciPy without changing the weighted result.
- With `weights`, the same labels as running the unweighted version on the rows repeated `weights` times.

Note from the first runs on real data (October 2026): average linkage on these flags peels off outliers, so cutting to K leaves one cluster with nearly every child. The dendrogram is still a useful figure, but the pipeline also offers k-means as the cross-check (section 3). Which one the paper uses is an open decision; see `docs/validation-checklist.md`.

## 3. `partition.cluster_kmeans(X, n_clusters, n_starts=20, seed=0, max_iter=300) -> PartitionResult`

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

Run the implementation tests with `uv run pytest tests/test_implementations.py -v`. Missing implementations fail the tests.
