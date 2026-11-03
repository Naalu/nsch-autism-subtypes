# What the pipeline writes, and how to extend it

This is the reference for anyone running, checking or changing the pipeline. It lists every file each step writes and what is in it, says which folder holds the numbers the paper quotes, and walks through the two most common changes: adding a condition and adding a check.

## Where results live

| Folder | What is in it | Committed? |
|---|---|---|
| `data/processed/` | The analysis table (one row per child, all years) and the codebook. Every step reads these. | Yes |
| `results/` | The **main folder**, written by our from-scratch code. Step 3's fits for K = 1 to 8, step 13's held-out likelihood and its choice of K. If steps 4 to 12 were run without `--k`, their outputs for that K land here too. | No |
| `results_k{K}/` | Steps 4 to 12, the step 6 bootstrap and step 15 at one K, run by `scripts/sweep_k.sh`. Each holds a copy of that K's step 3 fit, so the folder is self-contained. | No |
| `results_reference/`, `results_reference_k{K}/` | The same, made with StepMix / SciPy / scikit-learn (`BACKEND=reference`), as the check on our code. | No |
| `reports/figures/results/`, `reports/tables/` | The results figures and summary tables, drawn from `results/`. | Yes |
| `reports/figures/results_reference/`, `reports/tables/results_reference/` | The same figures and tables from a reference run, if one is drawn. Kept separate so they never overwrite the main ones. | Yes |
| `reports/figures/backend_comparison/` | Our code against StepMix: BIC overlay, matched profiles and rate differences. | Yes |
| `reports/implementation-comparison.md` | Manually maintained comparison notes; not written by the comparison script. | Yes |

**The canonical numbers are in `reports/tables/`.** Those files are regenerated from the result folders by `make figures` and committed, so the numbers in the paper can always be traced to a file in the repository. The result folders are scratch space: they can be deleted and rebuilt with `make all`.

The per-K folders must sit next to the main folder and be named after it (`results` and `results_k4`, or `results_reference` and `results_reference_k4`). Steps 13 and 14 and the figures script find them by that name.

## Comparing the implementations

The comparison script supplies information for report writing. It does not
create or overwrite Markdown reports; narrative notes are maintained manually.

`python scripts/compare_backends.py --k 4` reads saved fits from `results/` (ours)
and `results_reference/` (StepMix). It checks child identities and condition
order, matches the classes, and writes:

- `reports/tables/backend_comparison.csv`: likelihood, BIC, ARI, rate gaps, and convergence by K.
- `reports/tables/backend_condition_rates.csv`: the matched values behind the profile diagrams.
- `reports/figures/backend_comparison/backend_bic.png`: BIC overlays and differences.
- `reports/figures/backend_comparison/backend_profiles.png`: matched profiles side by side.
- `reports/figures/backend_comparison/backend_rate_differences.png`: signed differences in percentage points.
- `results/backend_comparison.json`: comparison metrics, the profile K, input locations, and recorded fit settings.

The comparison charts use the 20-start fits from `make fit` with each
backend; they say how closely the two implementations agree, not whether the
classes are real.

`make figures BACKEND=reference`, or
`python scripts/make_results_figures.py --results results_reference`, writes
figures to `reports/figures/results_reference/` and tables to
`reports/tables/results_reference/`. The main destinations are unchanged.

## Fields that recur

Most check outputs share these JSON keys:

| Key | Meaning |
|---|---|
| `k`, `backend` | The number of classes and which code fit them (`ours` or `reference`). |
| `ari` | Adjusted Rand index between the two labelings being compared (1 = same groups, 0 = chance). |
| `ci_low`, `ci_high` | 95% interval for `ari` from resampling children. It covers sampling of children only, not refitting. |
| `share_confident`, `ari_confident` | The share of children whose largest class probability is at least 0.8 under both models, and the ARI among just those children. A large gap between `ari` and `ari_confident` means the disagreement is about children near a boundary. |
| `pass`, `pass_line` | Whether `ari` reached the line set before fitting (0.65, `ARI_PASS` in `validate.py`). |
| `mean_abs_rate_difference_per_class` | For each class, the average gap in condition rates between the two models after matching classes. |
| `flags` | The condition names, in column order, so the arrays in the file can be read without the codebook. |

The profile CSVs (`profiles_*.csv`) have one row per class and one column per condition, holding the rate of that condition in the class. Two size columns appear and they mean different things: `class_size` is the model's estimated share of children in the class (a fraction), and `n` is the number of children whose most likely class it is (a count).

## File by file

### Step 3, `scripts/03_fit_lca.py` (main folder)

| File | Contents |
|---|---|
| `lca_k{K}.npz` | The fitted model for each K: `class_sizes` (K), `condition_rates` (K x 18), `responsibilities` (children x K, each child's class probabilities), `flags`, and `year` and `hhid` for every row so later steps can check they have the same children in the same order. Also `log_likelihood`, `n_iter`, `converged`. Load with `_common.load_lca_fit`, which does those checks. |
| `lca_bic.csv` | One row per K: `log_likelihood`, `n_parameters`, `bic`, `entropy` (0 to 1, how cleanly children are assigned), `smallest_class_share`, `min_avepp` (lowest average class probability among a class's own members), `converged`, and `bic_shuffled` (BIC on column-shuffled data, the no-structure baseline). |
| `lca_fits.json` | Settings, `k_fitted`, and `bic_argmin_k`. Step 3 does not choose K; that is step 13. |
| `profiles_k{K}.csv` | Condition rates by class for every K. |
| `figures/bic.png` | BIC against K, real and shuffled. |

### Step 4, `scripts/04_crosscheck.py`

| File | Contents |
|---|---|
| `crosscheck_kmeans.json`, `crosscheck_hierarchical.json` | The recurring keys, plus `ari_by_confidence` (ARI among children whose largest class probability is at least 0.6, 0.7, 0.8 and 0.9), `cluster_sizes_matched`, and `crosstab_lca_rows_method_cols` (how many children each pair of class and cluster shares). The k-means file also has `by_seed` (inertia and ARI for each seed's solution) and `ari_range_across_seeds`; the reported `ari` is the lowest-inertia solution's. |
| `method_comparison.csv` | `--method alternatives`: seven clustering methods against the latent classes, with `ari_all`, ARI at each confidence level, and `largest_cluster_share` (near 1 means the method put almost everyone in one cluster). |
| `profiles_{method}_k{K}.csv` | Condition rates in each cluster of the second method. |

### Step 5, `scripts/05_replicate.py`: `replicate.json`

Fit on 2016-21 (model A), refit on 2022-24 (model B), compare on the 2022-24 children. The recurring keys plus `n_fit`, `n_held_out`, `class_sizes_a`, `class_sizes_b_matched`, `rates_a`, `rates_b_matched`.

### Step 6, `scripts/06_bootstrap.py`

| File | Contents |
|---|---|
| `stability.csv` | One row per class: `n`, `mean_jaccard`, `sd_jaccard`, `verdict` (stable at 0.75 or above, dissolved at 0.5 or below). Step 13 reads this file to choose K. |
| `stability_jaccard.npy` | Every resample's Jaccard, classes x resamples. |

### Step 7, `scripts/07_describe.py`

| File | Contents |
|---|---|
| `shares_by_year.csv` | The same, one survey year at a time under the pooled model (`year`, `n`, then the `shares.csv` columns), to show whether the mix of classes drifts across the decade. |
| `shares.csv` | Weighted share of US autistic children aged 6-17 in each class: `share`, `se`, `ci_low`, `ci_high`, `n_modal` (children assigned), `weighted_count`. |
| `severity_by_class.csv` | Weighted share rated mild, moderate and severe in each class, each with a standard error. |
| `demographics.csv` | Weighted mean age and share female by class. |
| `count_vs_class.csv` | The severity models: `model`, `n_params`, `log_likelihood`, `bic`, `cv_log_loss`, `bic_vs_count`. |
| `severity_test.json` | The planned test: `criterion` (as written before fitting), `pass`, `bic_count`, `bic_both`, `bic_class_alone`, `class_alone_beats_count`, `best_model_by_bic`. |

### Step 8, `scripts/08_forward_2025.py`

`forward_2025.json` (the frozen 2016-24 model against a refit on 2025: the recurring keys plus `n_2025`, `shares_2025_frozen_model`, `rates_refit_matched`) and `shares_2025.csv` (same columns as `shares.csv`).

### Step 9, `scripts/09_age_bands.py`

`age_bands.json` holds `bands` with one entry per band (`6_11`, `12_17`), each with the recurring keys plus `n`, `class_sizes_band_fit_matched`, `class_shares_under_pooled_model`, `entropy_band_fit`. `profiles_age_{band}.csv` has the band fit's profiles.

### Step 10, `scripts/10_parallel_ever.py`

`parallel_ever.json`: the classes rebuilt from the "ever told" items against the main classes. The recurring keys plus `n_items_differing_between_rules`, `class_sizes_ever_matched`, `class_sizes_current`, and `prevalence_current` and `prevalence_ever` for each condition. Needs the `ever_*` columns in the analysis table, which step 2 writes.

### Step 11, `scripts/11_robustness.py`

| File | Contents |
|---|---|
| `robustness.json` | `weighted_fit` (survey-weighted refit against the main fit), `dropped_vs_kept` (children dropped for a missing answer against those kept), `seed_sensitivity` (`vs_main`, `pairwise`, `min_pairwise_ari` across seeds). |
| `profiles_weighted_k{K}.csv` | The weighted fit's profiles. |
| `dropped_vs_kept.csv` | The comparison of dropped and kept children as a table. |

### Step 12, `scripts/12_dimensions.py`

| File | Contents |
|---|---|
| `model_comparison.csv` | Latent trait models (D = 1 to 4) and latent class models (K = 1 to 8, read from `--fits`) in one table: `family`, `size` (D or K), `n_parameters`, `log_likelihood`, `bic`, `converged`. |
| `loadings_2d.csv`, `loadings_3d.csv` | Slopes for each condition on each dimension after varimax rotation, plus `intercept` and `prevalence`. |
| `bivariate_residuals.csv` | For every pair of conditions under each model: `observed_both`, `expected_both`, `chi_square`. Above 3.84 means the model misses that pair. |
| `factor_scores.npy` | Each child's position on the two dimensions. |
| `dimensions.json` | `model_comparison`, a summary of `bivariate_residuals`, `age_and_period` (score correlations with age and year, and loading congruence between age bands), `quadrants` (the classes against a two-cut rule on the scores), and `simulation` (latent class BIC curves on data simulated from the continuous model and from the class model, beside the real curve). |

### Step 13, `scripts/13_heldout_selection.py` (main folder)

| File | Contents |
|---|---|
| `heldout_selection.csv` | Each model fit on earlier years and scored on later ones: `split` (`2022_24` or `2025`), `family`, `size`, `n_parameters`, `train_ll`, `heldout_ll`, `heldout_ll_per_child`, `diff_from_best`, `diff_se` (paired standard error of that difference). |
| `stability_by_k.csv` | For each K with a bootstrap on file: `min_jaccard` (the least stable class), `mean_jaccard`, `all_stable`, and the `folder` it came from. |
| `k_rule_sensitivity.csv` | The K the rule picks at thresholds 0.6, 0.7, 0.75 and 0.8 with the first 50, 100 and 200 resamples of each `stability_jaccard.npy`: `k_passing`, `chosen_k_largest_passing` (our rule) and `k_most_stable` (the K with the highest least-stable class). |
| `model_choice.json` | `chosen_k`, `rule`, `k_with_bootstrap` (the K values the rule could consider), `k_untested_above_choice`, and `heldout` (the summary of the held-out comparison). Every later script without `--k` reads `chosen_k` from here. |

### Step 14, `scripts/14_shape_and_stability.py` (main folder)

| File | Contents |
|---|---|
| `latent_density.csv` | One row per dataset (`real`, and each simulated `continuum` and `classes` replicate): `improvement`, how many log-likelihood units the estimated distribution of positions beats the assumed bell curve by, plus `n_iter` and `converged`. |
| `latent_density.npz` | The grid (`nodes`) and the estimated weight on each grid point for every dataset, keyed as in the CSV (`real`, `continuum_0`, `classes_0`, ...), and `class_model_k`. |
| `stability_calibration.csv` | Bootstrap stability on simulated data: `data` (`continuum` or `5 classes`), `k`, `min_jaccard`, `mean_jaccard`, `all_stable`, `smallest_class_share`, `jaccard_by_class`. Compare with `stability_by_k.csv` from step 13. |
| `positions_2d.npz` | One posterior draw of each child's position in the two-dimensional model (`positions`), the rotated `slopes` and `intercepts`, `flags`, and `year` and `hhid` to line the rows up with the analysis table. |
| `shape_and_stability.json` | Settings and a summary of both checks. |

### Step 15, `scripts/15_replication_calibration.py`

The replication checks (held-out years, the 2025 forward test, both age bands, k-means) run on data simulated from the two-dimensional continuum and from the K-class model, with every real child's survey year and age kept so the splits are the real ones. Written into the K folder beside the real checks it is calibrating.

| File | What is in it |
|---|---|
| `replication_calibration.csv` | One row per `data` (`continuum`, `{K} classes`, `real`), `replicate` and `check`: `ari` and `n` (children compared). The `real` rows are copied from `replicate.json`, `forward_2025.json`, `age_bands.json` and `crosscheck_kmeans.json` when those are on file. |
| `replication_calibration.json` | The real values, the simulated minimum, median and maximum for each check, a one-line reading per check (whether the real ARI sits inside, below or above the continuum's range) and the settings. |

## Running it, and how long it takes

On a recent laptop, with the reference backend:

| Command | Time |
|---|---|
| `make test` | under a minute |
| `make smoke` | about five minutes |
| `make fit` (step 3, K = 1 to 8, 20 starts) | about 5 minutes |
| one K in the sweep, steps 4 to 12 | 10 to 15 minutes |
| one K's bootstrap (step 6, 200 resamples) | 20 to 40 minutes |
| `make calibrate` (steps 14 and 15) | about an hour, plus 10 to 15 minutes for step 15 |
| `make sweep` (K = 4 to 7) | 3 to 4 hours, mostly the bootstraps |

The sweep keeps going when a step fails, logs everything to `results/sweep.log`, and writes a `STATUS` file into each K folder. Steps 13 and 14 and the figures run at the end. To rerun only those after a sweep, use `make choose calibrate figures`.

The scripts, `make` and `scripts/sweep_k.sh` all default to our from-scratch code. Pass `--backend reference` to a script, or `BACKEND=reference` to `make` or the sweep, to run the library versions into `results_reference/`.

## Adding a condition

1. Add an entry to `config/conditions.yaml` with the Census variable names for the "ever told" and "currently has" questions (the comments at the top of that file explain each field). Check the variable names against the CAHMI crosswalk for every year.
2. Run `make download`. This rebuilds `data/processed/analysis.parquet` and `codebook.csv`. The prevalence rule may drop the new condition; the codebook says so.
3. Rerun everything from step 3. Every saved fit records the conditions it was fit on, and later steps stop with a message if the codebook no longer matches, so a stale fit cannot be paired with the new table by accident.
4. Rerun `make test`. The tests in `tests/test_data.py` check the encoding rule, not the particular conditions, so they should still pass.

## Adding a check

1. Write the reusable part (the statistic) as a function in `src/nsch_subtypes/validate.py`, with a docstring that says what question it answers and what a good value looks like. Add a unit test on made-up data where the answer is known.
2. Write a numbered script that loads the table with `load_analysis`, the fit with `load_lca_fit(args.results, k, table, flags)`, and K with `resolve_k(args)`. Use `make_parser` for the shared options and `save_json` for the output, and report ARI with `ari_with_interval` so the result has the recurring keys above.
3. Add the script to `scripts/sweep_k.sh`, to the smoke test in `tests/test_pipeline_smoke.py`, and to this file.
4. If the check belongs in the summary figure, add its JSON file to `checks_table` in `scripts/make_results_figures.py`.
5. Decide the pass line before running it on the real data, and write it in the script's docstring.
