# Do autistic children fall into distinct subtypes?

[![CI](https://github.com/Naalu/nsch-autism-subtypes/actions/workflows/ci.yml/badge.svg)](https://github.com/Naalu/nsch-autism-subtypes/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Clinic-based studies keep reporting three or four "subtypes" of autism based on which other conditions a child has, and almost none of them check whether the groups show up again in a second sample. This project asks the question on nationally representative data: ten years of the National Survey of Children's Health (NSCH, 2016–2025), 11,370 autistic children aged 6–17, and 18 parent-reported co-occurring conditions. We cluster the children with a latent class model written from scratch, then put the classes through checks fixed in advance: a second method has to find the same groups, the groups have to come back in later survey years (including the 2025 file, released after the design was set), each group has to survive resampling, and the groups have to tell us more about parent-rated severity than a plain count of conditions does. If they don't, that is the finding.

Course project for CS 472/572 (Unsupervised Learning) at Northern Arizona University, fall 2026. Chris Reger and Haley Berger.

## What's here

```
config/conditions.yaml     the 18 condition flags, the raw survey items behind each, and every item we left out, with reasons
src/nsch_subtypes/         the package
  data.py                  download, select, encode (steps 1-2)
  lca.py                   latent class model fit by EM, from scratch (step 3)
  hierarchical.py          Jaccard + average linkage, from scratch (step 4)
  partition.py             k-means on the flags, from scratch (alternative for step 4)
  reference.py             StepMix, SciPy and scikit-learn versions, used to check ours
  validate.py              class matching, ARI, bootstrap stability, the count-vs-class test
  describe.py              survey-weighted shares with design-based confidence intervals
scripts/01_select.py ... 08_forward_2025.py   one pipeline step each, one input and one output
data/processed/            the analysis table (committed), codebook, selection counts
reports/data-overview.md   the data in raw and processed form, with figures
docs/implementation-spec.md    what the from-scratch code has to do, and the tests it has to pass
docs/validation-checklist.md   how the design maps onto the SUVAC subtyping checklist; related results
tests/                     run on synthetic data; no download needed
```

## Run it

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Naalu/nsch-autism-subtypes
cd nsch-autism-subtypes
uv sync --group dev
make test                      # unit tests on synthetic data
make data                      # downloads ~60 MB from the Census Bureau, selects and encodes (steps 1-2)
make all                       # steps 3-8 with the from-scratch code
make all BACKEND=reference     # the same steps with StepMix / SciPy / scikit-learn
```

`data/processed/analysis.parquet` is committed, so `make fit` and later steps work without downloading anything. Every script takes `--help`. Outputs go to `results/` (tables as CSV/JSON, figures as PNG).

The survey can also be read through the [nsch](https://github.com/NAU-ASD3/nsch-py) package: `uv sync --extra nsch` then `scripts/01_select.py --loader nsch`. That path is experimental until nsch publishes a release.

## The pipeline

| Step | Script | What it does | Pass line |
|---|---|---|---|
| 1 Select | `01_select.py` | Children aged 6–17 whose parent says they currently have ASD, every year 2016–2025 | |
| 2 Encode | `02_encode.py` | One 0/1 flag per condition; items dropped if they changed wording, left the survey, or are under 1% prevalence | |
| 3 Group | `03_fit_lca.py` | Latent class model for K = 1…8, 20 random starts each, K chosen by BIC; optional column-shuffled baseline | |
| 4 Cross-check | `04_crosscheck.py` | A second method (hierarchical or k-means) cut at the same K, compared with the adjusted Rand index | ARI ≥ 0.65 |
| 5 Replicate | `05_replicate.py` | Fit on 2016–21, refit on 2022–24, both label the 2022–24 children | ARI ≥ 0.65 |
| 6 Resample | `06_bootstrap.py` | 200 bootstrap refits; cluster-wise Jaccard stability | mean Jaccard ≥ 0.75 |
| 7 Describe | `07_describe.py` | Weighted national share of each class with 95% CI; severity, age and sex by class; count-vs-class test | |
| 8 Forward | `08_forward_2025.py` | Frozen 2016–24 model vs a fresh fit on 2025 | ARI ≥ 0.65 |

Steps 1–3 are a complete project on their own; 4–6 and 8 are independent checks of step 3's classes; 7 turns the result into national numbers. Survey weights are used in step 7 and 8 only, to describe the classes, never to fit them.

The pass lines and the planned null result (the count test) were set in the project proposal on October 13, 2026, before any model was fit. Where they come from is in `docs/validation-checklist.md`.

## The data

Public files from the [Census Bureau](https://www.census.gov/programs-surveys/nsch/data/datasets.html); no application needed. `reports/data-overview.md` walks through them: what the raw rows look like, how the selection and encoding work, the prevalence of each condition, how conditions co-occur, and what the models see.

Two things to know up front. The flags are parent reports of a provider's diagnosis ("a doctor told me, and my child currently has it"), so they mix prevalence with access to care. And each child appears once, so we can describe groups but not how a child moves between them.

## Status

October 2026. Steps 1–2 run on the real files. Steps 3–8 run end to end with the reference backend; the from-scratch implementations are being brought in against `docs/implementation-spec.md`. Two design questions are open and documented in `docs/validation-checklist.md`: which second method to use for the cross-check (average linkage on these data puts nearly every child in one cluster), and how to choose K when BIC does not turn up.

Results will be committed to `results/` when the analysis freezes (planned: November 13, 2026).

## Methods and sources

Latent class model fit by EM (Dempster, Laird & Rubin 1977); K by BIC (Nylund, Asparouhov & Muthén 2007); average linkage (Sokal & Michener 1958); adjusted Rand index (Hubert & Arabie 1985) and its interpretation bands (Steinley 2004); bootstrap cluster stability (Hennig 2007); subtyping validation checklist (Agelink van Rentergem, Deserno & Geurts 2021). Clinic-based subtyping studies we compare against: Doshi-Velez, Ge & Kohane 2014; Lingren et al. 2016; Vargason et al. 2019; Montgomery et al. 2023; Litman et al. 2025.

Reference implementations: [StepMix](https://github.com/Labo-Lacourse/stepmix) (Morin et al. 2023), [SciPy](https://scipy.org), [scikit-learn](https://scikit-learn.org). Variable naming follows the [nsch](https://github.com/NAU-ASD3/nsch) and [nsch-py](https://github.com/NAU-ASD3/nsch-py) packages; the cross-year item audit used the [CAHMI crosswalk](https://www.childhealthdata.org/).

## Citing the data

National Survey of Children's Health, 2016–2025. U.S. Census Bureau for the Health Resources and Services Administration's Maternal and Child Health Bureau. Public-use files.

## License

MIT. See `LICENSE`.
