# Do autistic children fall into distinct subtypes?

[![CI](https://github.com/Naalu/nsch-autism-subtypes/actions/workflows/ci.yml/badge.svg)](https://github.com/Naalu/nsch-autism-subtypes/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Autism is one diagnosis that covers very different children. Studies of hospital and clinic records keep finding three or four "subtypes" based on which other conditions a child has (ADHD, anxiety, epilepsy, and so on), but almost none of them check whether the same groups appear in a second, independent set of children.

This project asks the question on a national survey instead: ten years of the National Survey of Children's Health (2016 to 2025), 11,370 autistic children aged 6 to 17, and 18 parent-reported co-occurring conditions. We look for groups with a latent class model (our own implementation is being written for the course; until it passes its tests, the numbers here come from the StepMix library), then put the groups through checks we fixed in advance:

- a second clustering method has to find the same groups;
- the groups have to come back in later survey years, including the 2025 file, which no modeling choice was made on;
- each group has to survive resampling;
- the groups have to tell us more about parent-rated severity than counting a child's conditions does.

We also fit a model with no groups, in which children vary along a few continuous scales, and ask which description the data prefer.

The short answer: four broad profiles, built from two main axes (developmental-language and psychiatric), replicate. Their borders are soft. The continuous model fits better, finer splits do not hold up, and a third, medical-sensory direction shows up as a third axis or as a small class that does not hold together.

Course project for CS 472/572 (Unsupervised Learning) at Northern Arizona University, fall 2026. Chris Reger and Haley Berger.

## New here? Read in this order

1. `docs/background.md`: the question, the survey, and every method in plain language, with a glossary. No statistics background assumed.
2. `reports/data-overview.md`: what the data looks like, raw and processed, with figures.
3. The numbered scripts in `scripts/`, in order. Each begins with a summary of what it does and why.
4. `docs/validation-checklist.md`: what every check found, the decisions made after seeing results, and how the design compares with published subtyping studies.
5. `docs/outputs.md`: every file the pipeline writes, and how to add a condition or a check.

## What's here

```
config/conditions.yaml       the 18 conditions, the survey items behind each, and every item left out, with reasons
src/nsch_subtypes/           the package
  config.py                  reads the config file
  data.py                    download, select children, build the 0/1 flags (steps 1 and 2)
  lca.py                     latent class model fit by EM, from scratch (step 3; in progress)
  hierarchical.py            Jaccard distance + average linkage, from scratch (step 4; in progress)
  partition.py               k-means on the flags, from scratch (step 4; in progress)
  reference.py               StepMix, SciPy and scikit-learn versions, used to check ours
  validate.py                agreement, replication, stability, and the count-vs-class test
  latent_trait.py            the continuous alternative: a binary factor model with 1 to 4 dimensions
  describe.py                survey-weighted national shares with confidence intervals
  plots.py                   shared figure styling
scripts/01_select.py ... 14_shape_and_stability.py  one pipeline step each; sweep_k.sh and make_results_figures.py tie them together
data/processed/              the analysis table (committed), codebook, selection counts
reports/data-overview.md     the data in raw and processed form, with figures
reports/figures/results/     the results figures (classes, severity, loadings, checks by K, BIC curves)
paper/                       outline and draft sections
docs/background.md           start here if the data or the methods are new to you
docs/implementation-spec.md  what the from-scratch code has to do, and the tests it has to pass
docs/validation-checklist.md what each check found, decisions taken after the first results, related studies
docs/outputs.md              every output file and its contents; how to extend the pipeline
docs/team-notes.md           where the project stands and what is waiting on what
tests/                       unit tests on synthetic data, plus an end-to-end smoke test (make smoke)
```

## Run it

Needs Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Naalu/nsch-autism-subtypes
cd nsch-autism-subtypes
uv sync --group dev
make test                      # unit tests on synthetic data (under a minute)
make smoke                     # every step once at tiny settings (about 5 minutes)
make fit                       # step 3: latent class models K = 1 to 8 (about 5 minutes)
make sweep                     # steps 4 to 14 at K = 4 to 7, then the figures (3 to 4 hours)
make download                  # optional: rebuild the analysis table from the Census files (about 60 MB)
```

`data/processed/analysis.parquet` is committed, so nothing needs the download. `make` uses the StepMix / SciPy / scikit-learn versions by default; add `BACKEND=ours` for the from-scratch code once it is in place. Every script takes `--help`. Outputs go to `results/` and `results_k{K}/`; `docs/outputs.md` lists them.

The survey can also be read through the [nsch](https://github.com/NAU-ASD3/nsch-py) package: `uv sync --extra nsch`, then `scripts/01_select.py --loader nsch`. That path is experimental until nsch publishes a release.

## The pipeline

| Step | Script | What it does | Pass line |
|---|---|---|---|
| 1 Select | `01_select.py` | Children aged 6 to 17 whose parent says they currently have autism, every year 2016 to 2025 | |
| 2 Encode | `02_encode.py` | One 0/1 flag per condition; items dropped if they changed wording, left the survey, or are under 1% prevalence | |
| 3 Group | `03_fit_lca.py` | Latent class model for K = 1 to 8, 20 random starts each; BIC, entropy and a column-shuffled baseline. Does not choose K | |
| 4 Cross-check | `04_crosscheck.py` | A second method (k-means; the proposal's hierarchical method; six others) cut at the same K, compared with the adjusted Rand index | ARI ≥ 0.65 |
| 5 Replicate | `05_replicate.py` | Fit on 2016 to 2021, refit on 2022 to 2024, both label the 2022 to 2024 children | ARI ≥ 0.65 |
| 6 Resample | `06_bootstrap.py` | 200 bootstrap refits; how much of each class comes back | mean Jaccard ≥ 0.75 |
| 7 Describe | `07_describe.py` | Weighted national share of each class with 95% CI; severity, age and sex by class; count-vs-class test | |
| 8 Forward | `08_forward_2025.py` | Frozen 2016 to 2024 model against a fresh fit on 2025 | ARI ≥ 0.65 |
| 9 Age bands | `09_age_bands.py` | Refit within ages 6 to 11 and 12 to 17; compare with the pooled classes | ARI ≥ 0.65 |
| 10 Parallel | `10_parallel_ever.py` | Refit on flags built from the "ever told" items | ARI ≥ 0.65 |
| 11 Robustness | `11_robustness.py` | Survey-weighted fit, dropped-children comparison, seed sensitivity | |
| 12 Dimensions | `12_dimensions.py` | Latent trait models with 1 to 4 dimensions against the classes; pairwise fit; simulations from each model | |
| 13 Choose | `13_heldout_selection.py` | Held-out likelihood for every model on later years; K = the largest K whose classes all come back under resampling | |
| 14 Calibrate | `14_shape_and_stability.py` | The same stability and latent-shape checks on simulated data with no classes and with real classes, as a yardstick | |

`make sweep` (or `scripts/sweep_k.sh 4 5`) runs steps 4 to 12 at each K into `results_k{K}/`, then steps 13 and 14 and `scripts/make_results_figures.py`.

Steps 1 to 3 are a complete project on their own. Steps 4, 5, 8, 9 and 10 are independent checks of the classes from step 3; step 6 is the stability check that also decides K; 11 and 12 ask whether the modeling choices and the class description itself hold up; 13 asks how much structure carries over to new children; 14 measures what the stability and shape checks look like when the answer is known. Step 7 turns the result into national numbers. Survey weights are used only to describe the classes (steps 7 and 8, and the weighted-fit check in 11), never to fit the main model.

The pass lines and the planned null result (the count test) were set in the project proposal on October 13, 2026, before any model was fit. The rule for K was set after BIC failed to settle; `docs/validation-checklist.md` explains both.

## The data

Public files from the [Census Bureau](https://www.census.gov/programs-surveys/nsch/data/datasets.html); no application needed. `reports/data-overview.md` walks through them: what the raw rows look like, how selection and encoding work, how common each condition is, how conditions go together, and what the models see.

Two things to know up front. The flags are parent reports of a doctor's diagnosis ("a doctor told me, and my child currently has it"), so they mix having a condition with having had access to care. And each child appears once, so we can describe groups but not how a child moves between them.

## Status

October 20, 2026. All steps run end to end on the real files with the reference backend. The from-scratch implementations are being written against `docs/implementation-spec.md`, and every number below will be re-verified with them.

What the runs found (details in `docs/validation-checklist.md`, figures in `reports/figures/results/`):

- BIC kept falling through K = 8, and so did held-out likelihood, so neither picks a K. K was chosen after the fact as the largest K at which every class is bootstrap-stable. That is K = 4. Simulation shows that a smooth continuum with no groups is also stable at K = 4 and 5, so this K means "the finest division that comes back reliably", not proof of four groups.
- The four classes are the combinations of high and low on two axes, developmental-language and psychiatric. Weighted shares of U.S. autistic children aged 6 to 17: fewer co-occurring conditions 34%, developmental-language 33%, psychiatric 18%, both 16%.
- They replicate across survey years (ARI 0.72), in the 2025 file (0.79), within age bands (0.70, 0.78) and under resampling (Jaccard 0.91 or better). The "ever told" items agree at 0.66, just over the line. The second-method check fails (k-means ARI 0.54); the disagreement is about children near a border.
- Parent-rated severity follows the developmental-language axis and not the psychiatric one. Classes add to a plain condition count, but two continuous scores predict severity better than the classes do.
- A two-dimensional latent trait model fits better than any number of classes and predicts new survey years as well as eight classes with a third of the parameters. A third dimension picks up medical and sensory conditions, the same signal as a small fifth class that does not hold together under resampling. Finer class solutions fail replication.
- Estimated from the data, the distribution of children on the two scales is lumpier than a bell curve but about half as lumpy as four separate groups would make it: some concentration at typical combinations on top of continuous variation. The map of children in that space (`reports/figures/results/map.png`) shows one cloud with the classes as regions of it.

Dates: analysis freeze November 13, 2026; paper and final presentation dates to be added. The data are public-use files with no identifiers; no IRB review was required.

## Methods and sources

Latent class model fit by EM (Dempster, Laird and Rubin 1977); BIC (Nylund, Asparouhov and Muthén 2007); K by cluster stability (Ben-Hur, Elisseeff and Guyon 2002; Lange et al. 2004); classes against continuous factors (Bauer and Curran 2004; Lubke and Neale 2006; Markon and Krueger 2006); average linkage (Sokal and Michener 1958); adjusted Rand index (Hubert and Arabie 1985) and its interpretation bands (Steinley 2004); bootstrap cluster stability (Hennig 2007); subtyping validation checklist (Agelink van Rentergem, Deserno and Geurts 2021). Clinic-based subtyping studies we compare against: Doshi-Velez, Ge and Kohane 2014; Lingren et al. 2016; Vargason et al. 2019; Montgomery et al. 2023; Litman et al. 2025.

Reference implementations: [StepMix](https://github.com/Labo-Lacourse/stepmix) (Morin et al. 2023), [SciPy](https://scipy.org), [scikit-learn](https://scikit-learn.org). Variable naming follows the [nsch](https://github.com/NAU-ASD3/nsch) and [nsch-py](https://github.com/NAU-ASD3/nsch-py) packages; the cross-year item audit used the [CAHMI crosswalk](https://www.childhealthdata.org/).

## Citing the data

National Survey of Children's Health, 2016 to 2025. U.S. Census Bureau for the Health Resources and Services Administration's Maternal and Child Health Bureau. Public-use files.

## License

MIT. See `LICENSE`.
