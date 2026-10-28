# Methods (draft)

Draft for section 3. Numbers in brackets are placeholders to fill from the files named in `outline.md`.

## 3.1 Design

The proposal of October 13, 2026 named the clustering method, the second method, the pass line for every check and the planned null test, before any model was fit. The 2025 survey file had been released on October 5. It was processed and summarized with the other years, but no model was fit to it and no modeling choice used it until the forward test (section 3.4). Three decisions were made after seeing results and are marked as such below: the rule for choosing the number of classes, the switch of the second method from hierarchical clustering to k-means, and the comparison against continuous models.

## 3.2 The latent class model

Each child contributes a row of 18 binary flags. The latent class model (Lazarsfeld and Henry 1968; Goodman 1974) assumes K unobserved classes, each with its own probability for every flag, and conditional independence of the flags within a class. The likelihood of a child's row is the class-size-weighted sum over classes of the product of the 18 Bernoulli probabilities. We fit it by expectation-maximization (Dempster, Laird and Rubin 1977): the E-step computes each child's posterior class probabilities under the current parameters, the M-step replaces class sizes and flag probabilities with their posterior-weighted averages. Each fit used 20 random starts and kept the start with the highest log-likelihood.

The results reported here were computed with StepMix 3.0 (Morin et al. 2023). We implemented the same model from scratch in NumPy as part of the course requirement and checked it against StepMix on synthetic data with planted classes and on the real data. [Agreement statement once the from-scratch results are in.] Children with any missing flag were excluded from fitting; the resulting sample is 8,891 of 9,710 for 2016 to 2024.

## 3.3 Number of classes

We fit K = 1 to 8 and planned to choose K by the Bayesian information criterion (Nylund, Asparouhov and Muthén 2007), reporting relative entropy, the smallest class share and the lowest average posterior probability alongside it. We also fit the same models to a copy of the data in which each column was independently shuffled, which keeps every flag's prevalence and removes every association between flags; this gives the BIC curve expected when there is nothing to find.

BIC fell at every K and would have selected the upper limit of whatever range we fit. Held-out log-likelihood (section 3.6) also increased through K = 8. After seeing this we adopted a stability rule: K is the largest number of classes at which every class has a mean bootstrap Jaccard of at least 0.75 (section 3.4), following the stability approach to model order of Ben-Hur, Elisseeff and Guyon (2002) and Lange et al. (2004) and Hennig's (2007) threshold. The bootstrap was run at K = 4 to 8. The rule gave K = 4; we report K = 5 as a sensitivity analysis. Because the rule uses only the bootstrap, the cross-method, held-out-year, age-band and parallel-item checks remain tests of the chosen classes. The 2025 results were visible when the rule was set, so the 2025 test should be read as confirmatory.

To calibrate the rule we ran the same bootstrap (50 resamples, 10 starts per fit) on two simulated datasets of the real size: one from the fitted two-dimensional latent trait model (section 3.6), which contains no classes, and one from the K = 5 latent class model. The first shows how stable classes are when a continuum is partitioned; the second, whether a class as small as our fifth would be stable if it were real.

## 3.4 Checks

All comparisons between two labelings of the same children use the adjusted Rand index (ARI; Hubert and Arabie 1985), which is 1 for identical partitions and about 0 for independent ones and does not depend on how classes are numbered. Classes from different fits were matched by the Hungarian assignment on the distance between their profile vectors. The pass line of 0.65 is the floor of Steinley's (2004) "moderate" band. Each ARI is reported with a 95% interval from 500 resamples of children with the two labelings held fixed, which reflects sampling of children but not refitting, and with the ARI among children whose largest class probability is at least 0.8 under both models.

Second method. k-means on the flag matrix, cut at the same K, clusters matched to classes by profile. On these data k-means has several near-tied optima, so we ran 200 starts from each of five seeds and report the lowest-inertia solution together with the range of ARI across seeds. The proposal's second method, average-linkage hierarchical clustering on Jaccard distance, was run and is reported along with five other methods; see section 4.3 for why it was replaced. Trees were cut with `scipy.cluster.hierarchy.cut_tree`, because `fcluster` with the `maxclust` criterion returns fewer clusters than requested when merge heights tie.

Held-out years. One model fit on 2016 to 2021 (n = 4,575) and one on 2022 to 2024 (n = 4,316) at the chosen K. Both label the 2022 to 2024 children; the ARI between the two labelings is the statistic.

2025 forward test. The frozen 2016 to 2024 model labels the 2025 children (n = 1,505) without refitting; a fresh model is fit on 2025 alone; ARI between the two labelings. Weighted 2025 class shares are reported beside the pooled shares.

Bootstrap stability. 200 resamples with replacement (100 at K = 7), refit at the given K, classes matched to the original by profile. For each original class, the Jaccard overlap with the best-matching bootstrap class, computed over the distinct children present in the resample. Hennig's (2007) reading: mean Jaccard of 0.75 or above is stable, 0.5 or below has dissolved.

Age bands. Separate fits on ages 6 to 11 (n = 3,917) and 12 to 17 (n = 4,974), matched to the pooled model; ARI between the pooled model's labels and the band model's labels on the band's children.

Parallel items. Flags rebuilt from the "ever told" items in place of the "currently has" items, refit at the chosen K, ARI against the main labels.

Robustness. Refit with survey weights as frequency weights (ARI against the unweighted fit); comparison of excluded and included children on age, sex, severity and reported conditions; refits from five further seeds (pairwise ARI).

## 3.5 The planned null test

Parent-rated autism severity (mild, moderate, severe) was never shown to the clustering. We fit ordinal logistic regressions of severity, each adjusted for age and sex, on the count of flagged conditions (`count`), on class membership (`class`), and on both (`both`), and compare them by BIC and five-fold cross-validated log-loss. The test as written before fitting reads "if `class` and `both` do not beat `count`, the classes are a gradient in disguise". Because that wording can be read as requiring either model or both models to beat the count, we report both readings. After the first results we added comparators that summarize the same 18 flags in other ways: posterior class probabilities in place of the modal class, the two factor scores from the latent trait model (section 3.6), factor scores plus class, and all 18 flags entered separately.

## 3.6 Classes or dimensions

To ask whether the data are better described by discrete classes or by continuous dimensions, we fit latent trait models with D = 1 to 4 dimensions (the multidimensional two-parameter logistic model, equivalent to binary factor analysis) to the same flag matrix by marginal maximum likelihood. The latent positions have a standard normal prior and are integrated out on a Gauss-Hermite tensor grid (21, 15, 11 and 7 points per dimension for D = 1 to 4). Estimation uses EM to a warm start followed by L-BFGS on the exact gradient, and the final log-likelihood is recomputed on a finer grid (41, 31, 19 and 11 points). The marginal log-likelihood is on the same scale as the latent class log-likelihood, so BIC compares the two families directly; the two-dimensional model has 53 free parameters against 75 for K = 4. Slopes were varimax-rotated for interpretation, which does not change the likelihood.

Three further comparisons. First, local dependence: for every pair of flags, a Pearson chi-square comparing the observed share of children with both against the share each model implies (values above 3.84 flag a pair the model misses). Second, simulation in both directions, following Bauer and Curran (2004): three datasets of the real size simulated from the fitted two-dimensional model, which contains no classes, and three from the K = 4 class model, each run through the latent class BIC sweep for K = 1 to 6. Third, held-out log-likelihood: every class and trait model fit on 2016 to 2021 and scored on 2022 to 2024, and fit on 2016 to 2024 and scored on 2025, with standard errors for each model's difference from the best computed from the per-child differences. Held-out likelihood is used to say how much structure generalizes and which family captures it more economically, not to choose K.

Fourth, the shape of the latent distribution. Holding the two-dimensional model's item parameters fixed, we estimated the distribution of positions nonparametrically as weights on an evenly spaced 41 by 41 grid over [-4, 4] squared, by EM (the nonparametric maximum likelihood estimate of a mixing distribution; Laird 1978), and report the gain in log-likelihood over the standard normal. Because this estimator is lumpy even when the true distribution is smooth, the same procedure, including refitting the latent trait model, was applied to five datasets simulated from the two-dimensional model and five simulated from the K = 4 class model. For plotting, each child's position is a single draw from its posterior under the normal prior (a plausible value), which keeps the population spread that posterior means lose.

The comparison between a mixture of classes and continuous factors follows Lubke and Neale (2006) and Markon and Krueger (2006). A factor mixture model (Lubke and Muthén 2005), which combines both, was not fit and is left for future work.

## 3.7 Description

Survey weights were used only to describe: class shares with design-based standard errors (strata = year by state by sampling stratum; Taylor linearization with the with-replacement approximation), severity distribution, mean age and share female by class, with pooled weights divided by the number of years. Estimates for a single class are domain estimates: every child stays in the variance calculation, with a zero contribution outside the class, so strata without class members are not dropped. Class shares are posterior-probability averages rather than modal counts, so assignment uncertainty is carried into the estimate.

## 3.8 Software

Python 3.12; NumPy, SciPy, scikit-learn, statsmodels, StepMix 3.0. All code, the committed analysis table and codebook, locked dependency versions and continuous integration are at [repository URL]. Every step is a script with a fixed seed; `make all` reproduces the results.
