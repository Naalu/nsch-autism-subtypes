# Validation checklist

Clustering always produces groups, so a subtyping study has to show that its groups are more than an artifact of one sample and one algorithm. Agelink van Rentergem, Deserno and Geurts (2021) reviewed 156 autism subtyping studies from 2001 to 2020 and distilled the ways of showing that into a seven-item checklist, SUVAC (SUbtyping VAlidation Checklist). The table below says how this project handles each item. The right-hand column is the share of the 156 reviewed studies that used each strategy; no study used all seven, and most used one or two.

The methods named in the table (ARI, bootstrap Jaccard, entropy) are explained in plain language in `docs/background.md` section 4.

| SUVAC item | What it asks | This project | Share of reviewed studies |
|---|---|---|---|
| Cross-method replication | Is more than one subtyping method applied? | Yes. Latent class model (EM) and a second method with no probability model, compared with the adjusted Rand index at the same K. Pass line ARI ≥ 0.65. `scripts/04_crosscheck.py`. | 13% |
| Subtype separation | Is distinctiveness or assignment certainty reported? | Yes. Relative entropy and smallest class share for every K (`results/lca_bic.csv`), class probabilities carried into the weighted shares, and cluster-wise bootstrap Jaccard (`scripts/06_bootstrap.py`). | 38% |
| Independent replication | Is the analysis repeated in a second sample? | Yes, twice. Fit on 2016–21, refit on 2022–24, ARI on the 2022–24 children (`scripts/05_replicate.py`). Then the 2025 file, released after the design was fixed, labeled by the frozen 2016–24 model and by a fresh refit (`scripts/08_forward_2025.py`). | 9% |
| Temporal stability | Is the analysis repeated with the same participants later? | No. The NSCH surveys each child once. Stated as a limitation. | 3% |
| External validation | Are subtypes compared on variables not used to form them? | Yes. Parent-rated severity, never shown to the models, compared across classes; the count-vs-class test asks whether the classes predict severity beyond a plain count of conditions (`scripts/07_describe.py`). Foregone care and emergency-department use can be added as further outcomes. | 88% |
| Parallel validation | Is the analysis repeated with a second set of variables for the same constructs? | Planned. Refit with the "ever told" items in place of the "currently has" items; the encoder already builds both. | 6% |
| Predictive validation | Do subtypes predict variables measured later? | No, for the same reason as temporal stability. | 3% |
| Predictions stated in advance | Were the validation predictions formulated before the analysis? | Pass lines (ARI ≥ 0.65, mean Jaccard ≥ 0.75) and the planned null result were in the proposal presented on October 13, 2026, before any model was fit. The first commit of this repository carries them in `validate.py`. | not scored |

## Pass lines and where they come from

- **ARI ≥ 0.65.** Steinley (2004) gives rough bands for the adjusted Rand index: 0.90 and up excellent, 0.80–0.90 good, 0.65–0.80 moderate, below 0.65 poor. We use the floor of "moderate" because the comparisons are between different children (held-out years) or very different methods, where near-perfect agreement is not expected. In a simulation with three equal groups, moving about one child in eight to a wrong group gives an ARI near 0.65.
- **Mean bootstrap Jaccard ≥ 0.75** for a stable class, ≤ 0.5 for a dissolved one, from Hennig (2007). 200 resamples, twice the default in `fpc::clusterboot`.
- **Number of classes by BIC**, following Nylund, Asparouhov and Muthén (2007). If BIC keeps falling as K grows, which it did in the first runs, the write-up reports the whole curve, the entropy and smallest-class share at each K, and the column-shuffled baseline (`03_fit_lca.py --permutation-null`), and says how K was chosen.

## Open decisions (October 2026)

1. **Cross-check method.** The proposal named Jaccard distance with average linkage. On this data average linkage peels off outliers: at any K, one cluster holds nearly every child, and the 517 children with no conditions form a second. k-means on the 0/1 flags (as in Vargason et al. 2019) gives balanced clusters and ARI near 0.7 against the latent classes at K = 3. Options: switch the cross-check to k-means and keep the dendrogram as a descriptive figure, or keep hierarchical and report that it fails. Either way the change from the proposal is documented here.
2. **Choice of K** when BIC does not turn up. See above.

## Related results to compare against

Unsupervised subtyping has no benchmark with ground truth, so "state of the art" means the published subtype solutions closest to ours and the strongest sensible methods on our own data.

| Study | Sample | Inputs | Method | Classes | Validation used |
|---|---|---|---|---|---|
| Doshi-Velez, Ge & Kohane 2014 | ~4,900 autistic patients, one hospital's records; 496 from another region | Diagnosis-code time series, ages 0–15 | Hierarchical clustering | 3 small distinct groups (seizure, multisystem, psychiatric) + 1 large unresolved | Replication sample |
| Lingren et al. 2016 | 20,658 autistic patients, children's hospital records | Comorbidity codes | Clustering (secondary analysis) | Psychiatric, developmental, seizure | None reported |
| Vargason, Frye, McGuinness & Hahn 2019 | 3,278 autistic children, private insurance claims | Co-occurring condition patterns, first 5 years | k-means | 3: high rates of everything (23.7%), developmental delay (26.5%), low rates (49.8%) | None reported |
| Montgomery et al. 2023, Australian Autism Biobank | 754 of 1,151 autistic children, ages 2–17 | 37 indicators: autism traits, cognition, psychiatric, medical | Latent profile analysis | 4, from fewer support needs (29%) to high medical and psychiatric comorbidity (26%) | Authors call for replication |
| Litman et al. 2025, *Nature Genetics* | >5,000 autistic children aged 4–18 in SPARK, with genetics | Binary, categorical and continuous traits incl. co-occurring conditions | Generative mixture model | 4: social/behavioral challenges (37%), mixed ASD with developmental delay (19%), moderate challenges (34%), broadly affected (10%) | Independent cohort |
| This project | 9,710 autistic children aged 6–17, NSCH 2016–24; 1,660 in 2025 | 18 parent-reported current conditions | Latent class model (EM), from scratch | Chosen by BIC | Cross-method, two independent replications, bootstrap, external (severity) |

Two comparisons are informative even without shared data. First, whether a "broadly affected" or high-everything class appears and how large it is (Vargason 24%, Litman 10%); our count-vs-class test asks whether such a class is more than a gradient. Second, whether our survey-weighted national shares look like the SPARK class sizes; SPARK is a volunteer registry, so a difference would itself be a finding.

On our own data the method comparisons are: the from-scratch code against StepMix and SciPy (correctness), k-means against the latent classes (a second method), the column-shuffled baseline (no structure), and the condition count against the classes on severity (the planned null).
