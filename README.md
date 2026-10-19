# Do autistic children fall into distinct subtypes?

[![CI](https://github.com/Naalu/nsch-autism-subtypes/actions/workflows/ci.yml/badge.svg)](https://github.com/Naalu/nsch-autism-subtypes/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Autism is one diagnosis that covers very different children. Studies of hospital and clinic records keep finding three or four "subtypes" based on which other conditions a child has (ADHD, anxiety, epilepsy, and so on), but almost none of them check whether the same groups appear in a second, independent set of children.

This project asks the question on a national survey instead: ten years of the National Survey of Children's Health (2016 to 2025), 11,370 autistic children aged 6 to 17, and 18 parent-reported co-occurring conditions. We look for groups with a latent class model written from scratch, then put the groups through checks we fixed in advance:

- a second clustering method has to find the same groups;
- the groups have to come back in later survey years, including the 2025 file, which was released after the design was set;
- each group has to survive resampling;
- the groups have to tell us more about parent-rated severity than counting a child's conditions does.

If the groups fail those checks, that is the finding.

Course project for CS 472/572 (Unsupervised Learning) at Northern Arizona University, fall 2026. Chris Reger and Haley Berger.

## New here? Read in this order

1. `docs/background.md`: the question, the survey, and every method in plain language, with a glossary. No statistics background assumed.
2. `reports/data-overview.md`: what the data looks like, raw and processed, with figures.
3. The numbered scripts in `scripts/`, in order. Each begins with a summary of what it does and why.
4. `docs/validation-checklist.md`: how the design compares with published subtyping studies.

## What's here

```
config/conditions.yaml       the 18 conditions, the survey items behind each, and every item left out, with reasons
src/nsch_subtypes/           the package
  config.py                  reads the config file
  data.py                    download, select children, build the 0/1 flags (steps 1 and 2)
  lca.py                     latent class model fit by EM, from scratch (step 3)
  hierarchical.py            Jaccard distance + average linkage, from scratch (step 4)
  partition.py               k-means on the flags, from scratch (alternative for step 4)
  reference.py               StepMix, SciPy and scikit-learn versions, used to check ours
  validate.py                agreement, replication, stability, and the count-vs-class test
  latent_trait.py            the continuous alternative: a two-dimensional binary factor model
  describe.py                survey-weighted national shares with confidence intervals
  plots.py                   the figures
scripts/01_select.py ... 12_dimensions.py     one pipeline step each; sweep_k.sh and make_results_figures.py tie them together
data/processed/              the analysis table (committed), codebook, selection counts
reports/data-overview.md     the data in raw and processed form, with figures
reports/figures/results/     the results figures (classes, severity, loadings, checks by K, BIC curves)
paper/                       outline and draft sections
docs/background.md           start here if the data or the methods are new to you
docs/implementation-spec.md  what the from-scratch code has to do, and the tests it has to pass
docs/validation-checklist.md how the design maps onto the SUVAC subtyping checklist; related results
tests/                       run on synthetic data with known classes; no download needed
```

## Run it

Needs Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Naalu/nsch-autism-subtypes
cd nsch-autism-subtypes
uv sync --group dev
make test                      # unit tests on synthetic data
make data                      # downloads about 60 MB from the Census Bureau, selects and encodes (steps 1 and 2)
make all                       # steps 3 to 8 with the from-scratch code
make all BACKEND=reference     # the same steps with StepMix / SciPy / scikit-learn
```

`data/processed/analysis.parquet` is committed, so `make fit` and the later steps work without downloading anything. Every script takes `--help`. Outputs go to `results/` (tables as CSV and JSON, figures as PNG).

The survey can also be read through the [nsch](https://github.com/NAU-ASD3/nsch-py) package: `uv sync --extra nsch`, then `scripts/01_select.py --loader nsch`. That path is experimental until nsch publishes a release.

## The pipeline

| Step | Script | What it does | Pass line |
|---|---|---|---|
| 1 Select | `01_select.py` | Children aged 6 to 17 whose parent says they currently have autism, every year 2016 to 2025 | |
| 2 Encode | `02_encode.py` | One 0/1 flag per condition; items dropped if they changed wording, left the survey, or are under 1% prevalence | |
| 3 Group | `03_fit_lca.py` | Latent class model for K = 1 to 8, 20 random starts each, K chosen by BIC; optional column-shuffled baseline | |
| 4 Cross-check | `04_crosscheck.py` | A second method (hierarchical or k-means) cut at the same K, compared with the adjusted Rand index | ARI ≥ 0.65 |
| 5 Replicate | `05_replicate.py` | Fit on 2016 to 2021, refit on 2022 to 2024, both label the 2022 to 2024 children | ARI ≥ 0.65 |
| 6 Resample | `06_bootstrap.py` | 200 bootstrap refits; how much of each class comes back | mean Jaccard ≥ 0.75 |
| 7 Describe | `07_describe.py` | Weighted national share of each class with 95% CI; severity, age and sex by class; count-vs-class test | |
| 8 Forward | `08_forward_2025.py` | Frozen 2016 to 2024 model against a fresh fit on 2025 | ARI ≥ 0.65 |
| 9 Age bands | `09_age_bands.py` | Refit within ages 6 to 11 and 12 to 17; compare with the pooled classes | ARI ≥ 0.65 |
| 10 Parallel | `10_parallel_ever.py` | Refit on flags built from the "ever told" items | ARI ≥ 0.65 |
| 11 Robustness | `11_robustness.py` | Survey-weighted fit, dropped-children comparison, seed sensitivity | |
| 12 Dimensions | `12_dimensions.py` | One- and two-dimensional latent trait models against the classes; simulated continuum | |

`scripts/sweep_k.sh 4 5` reruns steps 4 to 12 at the given K values into `results_k{K}/`; `scripts/make_results_figures.py` draws the results figures from them.

Steps 1 to 3 are a complete project on their own. Steps 4, 5, 6, 8, 9 and 10 are independent checks of the classes from step 3; 11 and 12 ask whether the modeling choices and the class description itself hold up. Step 7 turns the result into national numbers. Survey weights are used in steps 7 and 8 only, to describe the classes, never to fit them.

The pass lines and the planned null result (the count test) were set in the project proposal on October 13, 2026, before any model was fit. Where they come from is explained in `docs/validation-checklist.md`.

## The data

Public files from the [Census Bureau](https://www.census.gov/programs-surveys/nsch/data/datasets.html); no application needed. `reports/data-overview.md` walks through them: what the raw rows look like, how selection and encoding work, how common each condition is, how conditions go together, and what the models see.

Two things to know up front. The flags are parent reports of a doctor's diagnosis ("a doctor told me, and my child currently has it"), so they mix having a condition with having had access to care. And each child appears once, so we can describe groups but not how a child moves between them.

## Status

October 19, 2026. All steps run end to end on the real files with the reference backend; the from-scratch implementations are being written against `docs/implementation-spec.md` and every number below will be re-verified with them.

What the first full run found (details and figures in `docs/validation-checklist.md` and `reports/figures/results/`):

- BIC kept falling through K=8, so K was chosen after the fact as the largest K at which every class is bootstrap-stable and held-out replication passes. That is K=4.
- The four classes are the corners of two near-independent axes, developmental-language and psychiatric: few other conditions (34% of U.S. autistic children aged 6 to 17), developmental-language (33%), psychiatric (18%), both (16%).
- They replicate across survey years (ARI 0.72), in the 2025 file released after the design was fixed (0.79), within age bands (0.70, 0.78), under the "ever told" items (0.66), and under resampling (Jaccard 0.91 or better). The second-method check fails (k-means ARI 0.57); no distance-based method finds cluster geometry in this data.
- Parent-rated severity follows the developmental-language axis and not the psychiatric one.
- A two-dimensional latent trait model fits better than any number of classes (BIC 105,298 against 106,428 for K=4), and data simulated from it reproduce the latent class BIC curve. The result is best described as two dimensions, not four subtypes.

Dates: analysis freeze November 13, 2026; paper and final presentation dates to be added. The data are public-use files with no identifiers; no IRB review was required.

## Methods and sources

Latent class model fit by EM (Dempster, Laird and Rubin 1977); K by BIC (Nylund, Asparouhov and Muthén 2007); average linkage (Sokal and Michener 1958); adjusted Rand index (Hubert and Arabie 1985) and its interpretation bands (Steinley 2004); bootstrap cluster stability (Hennig 2007); subtyping validation checklist (Agelink van Rentergem, Deserno and Geurts 2021). Clinic-based subtyping studies we compare against: Doshi-Velez, Ge and Kohane 2014; Lingren et al. 2016; Vargason et al. 2019; Montgomery et al. 2023; Litman et al. 2025.

Reference implementations: [StepMix](https://github.com/Labo-Lacourse/stepmix) (Morin et al. 2023), [SciPy](https://scipy.org), [scikit-learn](https://scikit-learn.org). Variable naming follows the [nsch](https://github.com/NAU-ASD3/nsch) and [nsch-py](https://github.com/NAU-ASD3/nsch-py) packages; the cross-year item audit used the [CAHMI crosswalk](https://www.childhealthdata.org/).

## Citing the data

National Survey of Children's Health, 2016 to 2025. U.S. Census Bureau for the Health Resources and Services Administration's Maternal and Child Health Bureau. Public-use files.

## License

MIT. See `LICENSE`.
