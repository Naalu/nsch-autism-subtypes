# Comparing the from-scratch and reference implementations

Does our code give the same answers as the standard libraries? This report compares our NumPy latent class
model with StepMix, and our k-means and hierarchical clustering with scikit-learn and SciPy, on the same
children and condition flags. The latent class sections use saved fits; classes are matched by their fitted
condition rates before comparing profiles, since their original numbers are arbitrary. The last section
refits both versions of the two second methods.

## What was compared

- K values: 1, 2, 3, 4; profile diagrams use K=4.
- Children: 8,891, ages 6 to 17, survey years 2016 to 2024. The 2025 holdout is excluded.
- Settings: reference: 20 starts, seed 0, max 2000 iterations, ours: 20 starts, seed 0, max 2000 iterations.
- All compared fits converged.

Our code writes to `results/` and StepMix to `results_reference/`; `scripts/compare_backends.py` reads
those by default and refuses fits with different children, row order or condition order. [The summary
table](tables/backend_comparison.csv) contains every shared K; [the condition-rate
table](tables/backend_condition_rates.csv) contains the matched chart values.

## What the comparison found

At K=4, adjusted Rand index (ARI) is **0.988052**. ARI measures agreement between class
assignments after allowing for arbitrary class numbering; it is not a percentage of correctly classified
children. The absolute log-likelihood gap is **0.050244**. The largest matched condition-rate
gap is **0.3518 percentage points**.

The two implementations can reach different local optima and stop at different times. Equal seed values
do not produce identical initializations across libraries. These comparisons measure agreement; they do
not establish that the latent classes are real subtypes or choose K.

### Model fit

![BIC from both implementations and their difference](figures/backend_comparison/backend_bic.png)

The left panel overlays BIC from both implementations. The right panel shows the small differences that
the main BIC scale can hide. A negative difference means lower BIC for ours.

### Matched profiles

![Reference and from-scratch profiles](figures/backend_comparison/backend_profiles.png)

Both panels use the same probability scale and reference class order. Values are fitted condition
probabilities, rather than raw within-class percentages.

### Condition-rate differences

![Signed differences between matched condition rates](figures/backend_comparison/backend_rate_differences.png)

Positive values mean a higher rate in ours; negative values mean a higher rate in the reference. The
units are percentage points, and the color scale is centered at zero.

## k-means and hierarchical clustering

`scripts/compare_clustering.py` fits both versions of each method on the same 8,891 children at K = 2 to 6
and writes [the clustering table](tables/backend_clustering.csv). k-means uses the settings of step 4: 200
random starts from each of five seeds, keeping the solution with the lowest total distance (inertia).

### k-means

| K | ARI, ours vs scikit-learn | Inertia, ours | Inertia, scikit-learn |
|---|---|---|---|
| 2 | 1.000 | 15,317.6 | 15,317.6 |
| 3 | 1.000 | 13,740.3 | 13,740.3 |
| 4 | 1.000 | 12,585.4 | 12,585.4 |
| 5 | 0.989 | 11,837.8 | 11,837.7 |
| 6 | 0.682 | 11,229.7 | 11,229.4 |

At K = 2 to 4, including the K = 4 used everywhere else, the two find the same clusters. At K = 5 and 6
they stop on different solutions whose inertia differs by less than 0.003%: these are the near-tied
solutions described in step 4, and scikit-learn, which picks starting centers by k-means++ rather than at
random, lands on the slightly better one. The two pieces of arithmetic agree exactly: assigning every child to the nearest of
scikit-learn's final centers with our distance code reproduces scikit-learn's clusters at every K, and the
means of those clusters match its centers (to 0.0007 at K = 5, where scikit-learn stopped just inside its
convergence tolerance; to rounding error elsewhere).

### Hierarchical clustering

Our version runs on the 1,347 distinct condition profiles, each weighted by the number of children who share
it; SciPy runs on all 8,891 children. Both build trees with 1,346 merges above height 0 (SciPy first merges
identical children at height 0).

| K | ARI, ours vs SciPy | ARI, SciPy vs SciPy on shuffled rows (5 orders) |
|---|---|---|
| 2 | 1.000 | 1.000 to 1.000 |
| 3 | 0.948 | 0.952 to 0.962 |
| 4 | 0.947 | 0.906 to 0.956 |
| 5 | 0.947 | 0.911 to 0.966 |
| 6 | 0.948 | 0.899 to 0.976 |

The two trees do not match exactly, and SciPy does not match itself either. Jaccard distances on 18 yes/no
flags tie often, and average linkage has to break each tie by taking one merge first; that choice changes
the averages computed for every later merge. SciPy breaks ties by row order, so shuffling the children
changes its tree. Our tree differs from SciPy's by about as much as SciPy's trees differ from each other: our
ARI sits inside SciPy's own range at K = 4 to 6 and just below it at K = 3, and the largest merge-height gap
is 0.031, against 0.027 to 0.061 between SciPy runs. Every version puts the same 517 children with no
conditions in one cluster and more than 8,300 of the rest in another; the differences are in which handful of
children form the small clusters. This is the chaining described in `docs/validation-checklist.md`, decision
2, and the reason k-means is the reported second method.

## What the numbers in the repository come from

Every number in the README, the validation checklist and the paper comes from the from-scratch code
(`results/` and `results_k*/`). The StepMix fits in `results_reference/` and the refits above are the check
on it; rerunning `make compare` after a sweep refreshes this report's tables and figures.
