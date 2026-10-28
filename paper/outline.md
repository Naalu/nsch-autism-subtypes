# Paper outline

Working title: Four profiles on two axes: what replicates when co-occurring conditions in a national sample of autistic children are clustered

Alternative: Checking autism subtypes against a national survey: what replicates and what does not

The claim, in three beats: four broad profiles, the combinations of two main axes, replicate. Their borders are soft: a continuous model fits better and predicts new years as well as twice as many classes, and finer splits do not replicate. Medical and sensory conditions form a third direction that appears as a third axis or as a small unstable class.

Target length: 8 to 10 pages plus supplement, in whatever format the course asks for. Every result sentence below has a file behind it; the path is given so the draft can cite numbers without re-deriving them.

## Abstract (150 words)

Problem, data, method, three findings, one caveat. Draft in `abstract.md` once results are frozen.

## 1. Introduction (1 page)

- Autism is one diagnosis covering very different children; co-occurring conditions are a natural place to look for subtypes.
- Published subtype studies (table in `docs/validation-checklist.md`) report three or four groups from clinic and registry samples. Agelink van Rentergem et al. (2021): of 156 studies, 9% used an independent replication sample and 13% a second method.
- Bauer and Curran (2003, 2004): mixture models find classes in continuous non-normal data. So "we found K groups" is not evidence of K groups.
- This paper: a national survey, checks fixed in advance, and a direct comparison of a class model against a continuous one.
- One-paragraph preview of the three findings.

## 2. Data (1 page)

Source: `reports/data-overview.md`, `data/processed/selection_counts.csv`, `data/processed/codebook.csv`.

- NSCH 2016 to 2025, one file per year, 431,222 children; selection to 11,370 autistic children aged 6 to 17 (table of counts per year).
- 18 conditions, the "ever told" / "currently has" item pairs, the encoding rule, the five excluded items and why.
- Complete-case: 8,891 of 9,710 for 2016 to 2024; 1,505 of 1,660 for 2025. Dropped children differ little (`results_k4/dropped_vs_kept.csv`).
- What the flags are and are not: parent reports of a provider's diagnosis.
- Figure: prevalence and co-occurrence (`reports/figures/prevalence.png`, `cooccurrence.png`).

## 3. Methods (1.5 pages)

Source: `docs/background.md` section 3, `docs/implementation-spec.md`, draft in `methods.md`.

- Latent class model, EM, implemented from scratch and checked against StepMix (numbers from StepMix until that check is done).
- Choice of K: BIC planned; BIC and held-out likelihood both keep improving; the stability rule (stated as post hoc).
- Checks, each with its pass line and the reference it comes from: second method (ARI 0.65, Steinley 2004); held-out years; 2025 forward test; bootstrap (Jaccard 0.75, Hennig 2007); age bands; parallel items; count-vs-class on severity.
- The continuous alternative: latent trait models with 1 to 4 dimensions on the same likelihood scale; pairwise residuals; simulation in both directions; held-out likelihood.
- Survey weights: description only; weighted-fit robustness check.

## 4. Results (3 pages)

### 4.1 How many classes
`results/lca_bic.csv`, `reports/tables/stability_by_k.csv`, `reports/figures/results/model_fit.png`. BIC falls monotonically; shuffled baseline rises; drops shrink to 6 at K=8. Held-out likelihood also improves through K=8. Stability rule gives K=4 (least stable class 0.91 at K=4, 0.62 at K=5). Calibration: a simulated continuum is stable through K=5, so the rule finds the finest reproducible division, not groups; a real 8% class would be stable (0.87), so the fifth class's failure is not about size (`reports/figures/results/stability_vs_simulation.png`). Table: checks by K (`reports/tables/checks_by_k.csv`, `reports/figures/results/checks_by_k.png`).

### 4.2 The four classes are a two-by-two
`results_k4/lca_k4.npz`, `reports/figures/results/classes.png`, `reports/tables/class_names_k4.csv`, `results_k4/shares.csv`. Profiles, weighted national shares with intervals, age and sex by class.

### 4.3 What replicates
Held-out years 0.72, 2025 0.79 with shares (`reports/figures/results/shares_pooled_vs_2025.png`), bootstrap 0.91 to 0.93, age bands 0.70 and 0.78, "ever told" 0.66 (marginal). Second method 0.54 (0.54 to 0.57 across seeds): fails, but 0.93 among children placed with probability 0.9 or more. Method table (`reports/tables/method_comparison.csv`) and the dendrogram.

### 4.4 Severity
`results_k4/severity_by_class.csv`, `results_k4/count_vs_class.csv`, `reports/figures/results/severity_by_class.png`. About 20% severe in the two developmental classes, 2% to 4% in the other two. Count plus class beats count; class alone does not; factor scores beat count plus class. Both readings of the pre-fit wording.

### 4.5 Classes or dimensions
`results_k4/dimensions.json`, `loadings_2d.csv`, `loadings_3d.csv`, `bivariate_residuals.csv`, `results/heldout_selection.csv`, `reports/figures/results/loadings.png`, `model_fit.png`. Two-dimensional model wins on BIC with 53 parameters and matches K=8 on held-out years; loadings split cleanly; quadrant rule ARI 0.79; third dimension is medical-sensory and cuts poorly fit pairs from 49 to 36; data simulated from the continuum reproduce the real BIC curve, data simulated from four classes do not. Shape of the latent distribution: gain over the bell curve 94 for real data, 16 to 23 for simulated continua, 177 to 242 for simulated classes (`latent_density.png`). The map (`map.png`): one cloud, classes as regions, uncertain children on the borders. Agreement by confidence (`agreement_by_confidence.png`): every check approaches 1 among confidently classified children.

### 4.6 The K=5 sensitivity analysis
`results_k5/`. The severe intellectual-disability-and-medical class: 9% weighted, 42% severe, Jaccard 0.62, held-out-years replication 0.646 (fails), weighted fit 0.58. The same signal as the third dimension. Exists; membership unstable.

## 5. Discussion (1.5 pages)

- What was found, in the three-beat form (`docs/team-notes.md` has the short version).
- Against the literature: Litman's 37/19/34/10 and the "broadly affected" class in Vargason and Litman; the psychiatric and developmental split in Lingren and Doshi-Velez; Zhang et al. 2025's two genetic factors (earlier diagnosis and early communication difficulty against later diagnosis, adolescent behavior and ADHD and mental health) as the counterpart of our two axes, including the age pattern; Kim et al. 2019 and Wittkopf et al. 2023 on dimensional and hybrid structure of core traits, which no study has done for co-occurring conditions.
- Why distance methods fail on this data (chaining, near-tied k-means optima) and what that says about cross-method replication as a criterion for sparse binary indicators.
- The post hoc K rule, said up front. The 2025 test as confirmatory.
- Future work: a factor mixture model, which allows classes and continuous variation within them.
- Limitations: parent report and access to care; one observation per child, so no temporal or predictive validation; entropy 0.72; the 8.4% dropped; "currently has" wording; the proposal's hierarchical method failing; results from StepMix until the from-scratch code is verified.
- What a clinician can take from it: two dials, not four boxes; severity ratings track the developmental dial.

## 6. Reproducibility (half page)

Repository, `make all`, committed analysis table and codebook, locked dependencies, CI, from-scratch versus library agreement, seeds. Statement that no IRB review was required.

## Supplement

- S1 full BIC table with entropy and smallest class share for K=1 to 8, real and shuffled.
- S2 K=5 and K=8 profiles.
- S3 method comparison table and dendrogram.
- S4 age-band profiles (`results_k4/profiles_age_6_11.csv`, `profiles_age_12_17.csv`).
- S5 "ever told" profiles and prevalence under both rules.
- S6 robustness: weighted fit, dropped vs kept, seeds.
- S7 latent trait loadings and intercepts (2-D and 3-D); bivariate residuals; simulation table in both directions; held-out likelihood table.
- S8 stability calibration table and latent density estimates for every simulated dataset.
- S9 the codebook.

## Division of labor (proposal, to confirm)

Chris: data, methods sections 3 (checks and the continuous model), results 4.1, 4.3, 4.5, reproducibility, figures. Haley: methods section 3 (EM and hierarchical), results 4.2, 4.4, 4.6, discussion first draft. Both: introduction, abstract, limitations.
