# Methods (draft)

Draft for section 3. Numbers in brackets are placeholders to fill from the files named in `outline.md`. Written to be cut, not padded.

## 3.1 Design

We fixed the analysis before fitting any model. The proposal of October 13, 2026 named the clustering method, the second method, the pass lines for every check and the planned null test. The 2025 survey file was released on October 5, 2026 and was not opened until the 2016 to 2024 analysis was complete. Two decisions were made after seeing results and are marked as such below: the rule for choosing the number of classes, and the switch of the second method from hierarchical clustering to k-means.

## 3.2 The latent class model

Each child contributes a row of 18 binary flags. The latent class model (Lazarsfeld and Henry 1968; Goodman 1974) assumes K unobserved classes, each with its own probability for every flag, and conditional independence of the flags within a class. The likelihood of a child's row is the class-size-weighted sum over classes of the product of the 18 Bernoulli probabilities. We fit it by expectation-maximization (Dempster, Laird and Rubin 1977): the E-step computes each child's posterior class probabilities under the current parameters, the M-step replaces class sizes and flag probabilities with their posterior-weighted averages. Each fit used 20 random starts and kept the start with the highest log-likelihood. Flag probabilities were clipped to [1e-6, 1 - 1e-6]; the E-step was computed in log space.

The model was implemented from scratch in NumPy as part of the course requirement and checked against the StepMix package (Morin et al. 2023) on synthetic data with planted classes and on the real data. [Agreement statement once the from-scratch results are in.] Children with any missing flag were excluded from fitting; the resulting sample is 8,891 of 9,710 for 2016 to 2024.

## 3.3 Number of classes

We fit K = 1 to 8 and planned to choose K by the Bayesian information criterion (Nylund, Asparouhov and Muthén 2007), reporting relative entropy and the smallest class share alongside it. We also fit the same models to a copy of the data in which each column was independently shuffled, which preserves every flag's prevalence and removes every association between flags; this gives the BIC curve expected when there is nothing to find.

BIC fell at every K and would have selected the upper limit of whatever range we fit. We therefore adopted, after seeing the sweep, the rule: the largest K at which every class is stable under the bootstrap (section 3.4) and the held-out-years replication passes its line. This gave K = 4, and we report K = 5 as a sensitivity analysis. Because the 2025 result was visible when this rule was set, the 2025 test should be read as confirmatory.

## 3.4 Checks

All comparisons between two labelings of the same children use the adjusted Rand index (Hubert and Arabie 1985), which is 1 for identical partitions and about 0 for independent ones and does not depend on how classes are numbered. Classes from different fits were matched by the Hungarian assignment on the Euclidean distance between their profile vectors. The pass line of 0.65 is the floor of Steinley's (2004) "moderate" band.

Second method. k-means (Lloyd's algorithm, 20 starts) on the flag matrix, cut at the same K, clusters matched to classes by profile. The proposal's second method, average-linkage hierarchical clustering on Jaccard distance, was run and is reported; see section 4.3 for why it was replaced.

Held-out years. One model fit on 2016 to 2021 (n = 4,575) and one on 2022 to 2024 (n = 4,316) at the chosen K. Both label the 2022 to 2024 children; the ARI between the two labelings is the statistic.

2025 forward test. The frozen 2016 to 2024 model labels the 2025 children (n = 1,505) without refitting; a fresh model is fit on 2025 alone; ARI between the two labelings. Weighted 2025 class shares are reported beside the pooled shares.

Bootstrap stability. 200 resamples with replacement, refit at the chosen K, classes matched to the original by profile. For each original class, the Jaccard overlap with the best-matching bootstrap class, computed over the distinct children present in the resample. Hennig's (2007) reading: mean Jaccard of 0.75 or above is stable, 0.5 or below has dissolved.

Age bands. Separate fits on ages 6 to 11 (n = 3,917) and 12 to 17 (n = 4,974), matched to the pooled model; ARI between the pooled model's labels and the band model's labels on the band's children.

Parallel items. Flags rebuilt from the "ever told" items in place of the "currently has" items, refit at the chosen K, ARI against the main labels.

Robustness. Refit with survey weights as frequency weights (ARI against the unweighted fit); comparison of excluded and included children on age, sex, severity and reported conditions; refits from five further seeds (pairwise ARI).

## 3.5 The planned null test

Parent-rated autism severity (mild, moderate, severe) was never shown to the clustering. We fit three ordinal logistic regressions of severity, each adjusted for age and sex: on the count of flagged conditions, on class membership, and on both. Models are compared by BIC and by five-fold cross-validated log-loss. If class membership adds nothing to the count, the classes are a condition-count gradient with new names.

## 3.6 Classes or dimensions

To ask whether the data prefer discrete classes or continuous dimensions, we fit one- and two-dimensional latent trait models (the multidimensional two-parameter logistic model, equivalent to binary factor analysis) to the same flag matrix by marginal maximum likelihood with EM over a 15-point-per-dimension Gauss-Hermite grid. The marginal log-likelihood is on the same scale as the latent class log-likelihood, so BIC compares them directly; the two-dimensional model has 53 free parameters against 75 for K = 4. Slopes were varimax-rotated for interpretation, which does not change the likelihood.

Following Bauer and Curran (2004), we then simulated a dataset of the same size from the fitted two-dimensional model, which by construction contains no classes, and ran the latent class BIC sweep on it.

## 3.7 Description

Survey weights were used only to describe: class shares with design-based standard errors (strata = year by state by sampling stratum; Taylor linearization with the with-replacement approximation), severity distribution, mean age and share female by class, with pooled weights divided by the number of years. Class shares are posterior-probability averages rather than modal counts, so assignment uncertainty is carried into the estimate.

## 3.8 Software

Python 3.12; NumPy, SciPy, scikit-learn, statsmodels, StepMix 3.0. All code, the committed analysis table and codebook, locked dependency versions and continuous integration are at [repository URL]. Every step is a script with a fixed seed; `make all` reproduces the results.
