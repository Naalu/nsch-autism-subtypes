# How the from-scratch code works, and how it is tested

The methods the project is about are our own NumPy code: the latent class model (`lca.py`), average-linkage hierarchical clustering (`hierarchical.py`), k-means (`partition.py`) and the latent trait model (`latent_trait.py`). Every number in the paper comes from them. The first three have library counterparts in `reference.py` (StepMix, SciPy and scikit-learn) with the same signatures and result classes; `backends.py` maps `--backend ours` and `--backend reference` to one set or the other, and the tests check each of ours against its counterpart. The latent trait model has no library counterpart in the project; its tests check it against data simulated with known parameters.

Ground rules the code keeps:

- NumPy only inside these files. SciPy, scikit-learn and StepMix are used everywhere else and are what we check against.
- The same `seed` gives the same answer (`numpy.random.default_rng(seed)`).
- The signatures and result classes are shared with the library versions, so a script never needs to know which backend it has.

The iteration cap is `lca.MAX_ITER` (2,000), shared by step 3, every refit in steps 5 to 15 and the StepMix reference, so "converged" means the same thing everywhere. With a cap of 500 the K = 8 fit on the real data does not converge; at 2,000 its best start converges after about 1,400 iterations.

## 1. Latent class model: `lca.fit_lca(X, n_classes, n_starts=20, seed=0, tol=1e-6, max_iter=MAX_ITER) -> LCAResult`

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

Return the start with the highest final log-likelihood. `K = 1` needs no starts: the rates are the column means.

Missing flags (`NaN`) are allowed: a missing flag is left out of that child's likelihood and out of that condition's rate in the M-step, which is maximum likelihood when answers are missing at random. With no missing values the code takes the plain path above, bit for bit. The pipeline uses this only for the robustness refit that keeps children with blank answers.

The result class computes `labels`, `n_parameters = (K - 1) + K J`, `bic(n)` and `entropy()`.

What the tests check (`tests/test_implementations.py`):

- Log-likelihood never decreases across iterations.
- `class_sizes` sum to 1; every row of `responsibilities` sums to 1; shapes are right.
- On `synthetic.make_synthetic(n=2000, n_items=10, n_classes=3, separation=0.75)`: ARI against the true labels at least 0.8, and the matched rates within 0.05 of the generating rates.
- Against StepMix on the same data with the same K: best log-likelihood within 1.0 and matched rates within 0.02, with classes paired by `validate.match_classes`.
- `K = 1` log-likelihood equals the closed form.

Things that bite, and how the code handles them: a column that is all 0 or all 1 (the clip handles it); label switching between starts (irrelevant inside `fit_lca`, handled by `validate.match_classes` outside); the `n_starts` loop is the slow part, so the E-step is vectorized as matrix products (`X @ log(theta).T`).

## 2. Hierarchical clustering: `hierarchical.cluster_hierarchical(X, n_clusters, weights=None) -> HierarchicalResult` and `hierarchical.jaccard_distances(X) -> ndarray`

Agglomerative clustering, Jaccard distance, average linkage.

`jaccard_distances`: for rows `a`, `b`, `d = 1 - |a & b| / |a | b|`; define `d = 0` when both rows are all zero. Return the full (n, n) matrix with zeros on the diagonal. Intersections come from `X @ X.T` and unions from row sums.

`cluster_hierarchical`:

1. Start with `n` groups of one row each, weights `w` (all ones if `weights` is None).
2. Repeat `n - 1` times: find the pair of groups with the smallest distance; merge them; record `[i, j, distance, size]` where `i < j` are the group ids being merged, group ids for new groups count up from `n`, and `size` is the total weight of the new group. Update distances to every other group `C` with the weighted average
   `d(A+B, C) = (w_A d(A, C) + w_B d(B, C)) / (w_A + w_B)`.
3. Labels: undo the last `n_clusters - 1` merges, assign 0..n_clusters-1 to the remaining groups in order of first appearance.

The direct algorithm is O(n^3) with an (n, n) matrix. The pipeline collapses the 8,891 complete rows to 1,347 distinct profiles and passes the counts as `weights`, which makes it fast enough.

What the tests check:

- Identical labels to `reference.cluster_scipy` on a tie-free binary fixture (ARI = 1). On short binary profiles, equal Jaccard distances can lead to different valid trees; a separate test checks every merge against the minimum average distance between original members.
- Merge distances non-decreasing; `merges` has shape (n - 1, 4). `plots.dendrogram_plot` converts weighted sizes to displayed leaf counts for SciPy without changing the weighted result.
- With `weights`, the same labels as running the unweighted version on the rows repeated `weights` times.

On the real data average linkage chains: cut at K = 4, it puts 94% of children in one cluster. That is why k-means is the reported second method; `docs/validation-checklist.md`, decision 2, has the details. Trees are cut with the equivalent of SciPy's `cut_tree`, never `fcluster(..., "maxclust")`, which returns fewer clusters than asked when merge heights tie.

## 3. k-means: `partition.cluster_kmeans(X, n_clusters, n_starts=20, seed=0, max_iter=300) -> PartitionResult`

Lloyd's algorithm on the 0/1 matrix treated as real vectors.

Per start: pick `n_clusters` distinct rows at random as centers; repeat: assign each row to the nearest center by squared Euclidean distance; recompute centers as the mean of their rows (keep a center where it is if its cluster empties); stop when assignments do not change. Inertia = total squared distance to assigned centers. Return the start with the lowest inertia.

What the tests check: inertia never increases across iterations; on the synthetic data, ARI against `reference.cluster_kmeans_sklearn` at least 0.9 (k-means has local optima, so exact equality is not expected).

## 4. Latent trait model: `latent_trait.fit_latent_trait(X, n_dims=2, n_quadrature=None, max_iter=500, tol=1e-6, seed=0, max_slope=MAX_SLOPE) -> LatentTraitResult`

The multidimensional two-parameter logistic model: each child has a position on `n_dims` scales drawn from a standard normal, and the probability of each flag is the logistic of a weighted sum of the positions (the slopes) plus an intercept. Positions are integrated out on a Gauss-Hermite grid (`n_quadrature` points per dimension; 21, 15, 11 and 7 by default for D = 1 to 4). EM runs to a warm start, then L-BFGS-B polishes on the exact gradient, and the final log-likelihood is recomputed on a finer grid so the fit is not judged on the grid it was tuned on. Slopes are bounded at `MAX_SLOPE` (10); a fit that ends on the bound has a dimension collapsed onto one condition, and `result.at_bound` says so. Slopes are varimax-rotated for reading. `fit_latent_density` holds the item parameters fixed and estimates the distribution of positions on a grid without assuming a normal (step 14).

What the tests check (`tests/test_latent_trait.py`): on data simulated from known slopes the fit recovers the planted structure, and two dimensions beat one; with all slopes zero the log-likelihood equals the closed form for independent items, so it is on the same scale as the latent class model; the held-out scorer gives the same number as the fit on the fitting data, and the rotation is orthogonal; implied pair probabilities match simulated frequencies; drawn positions track the true ones; a dimension that collapses onto one item stops at the slope bound and is flagged; and the density estimate stays near the normal when the truth is normal and finds two lumps when positions come in two groups.

## 5. Which steps use which method

| Script | Method | Backend flag |
|---|---|---|
| `03_fit_lca.py` | `fit_lca` for K = 1 to 8 | yes |
| `04_crosscheck.py` | `cluster_kmeans` or `cluster_hierarchical` (with `weights`); six library methods with `--method alternatives` | yes |
| `05` to `11`, `13`, `14` | `fit_lca` for every refit | yes |
| `12_dimensions.py`, `13_heldout_selection.py`, `14_shape_and_stability.py`, `15_replication_calibration.py` | `fit_latent_trait` (ours only) | for the class fits |
| `15_replication_calibration.py` | `fit_lca` and `cluster_kmeans` on simulated data | yes |
| `16_outcomes.py` | `fit_latent_trait` for the axis scores (ours only); `fit_lca` for the refit without 2020 and 2021 | yes |

Two pieces use libraries by design whatever the backend: the survey-weighted refit in step 11 (StepMix, which takes weights) and the six alternative methods in step 4.

`validate.class_probabilities(X, class_sizes, condition_rates)` scores new children under a fixed model (the E-step with no update), so the held-out checks can label children with a model fit elsewhere.

## Running the tests

```bash
uv run pytest -q -m "not slow"           # unit tests on synthetic data, under a minute
uv run pytest -m slow -v                  # every pipeline step once at tiny settings, both backends, plus a regression test on the real data
```

On Windows without uv, call the environment's Python directly: `.venv/Scripts/python.exe -m pytest -q -m "not slow"`. The smoke test checks that the steps fit together, not the scientific results; those come from the full sweep.
