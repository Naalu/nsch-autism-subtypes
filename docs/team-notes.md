# Team notes

Where the project stands, what is waiting on what, and how to pick it up. Updated October 20, 2026.

## Where we are

The pipeline runs end to end on the real 2016 to 2025 files with the library backend (`--backend reference`). Every check in the proposal has been run at K = 4 to 8, plus the checks added after the first results (steps 9 to 13). The findings are summarized in the README status section and argued in `docs/validation-checklist.md`; the figures are in `reports/figures/results/`.

Short version: four broad profiles, the combinations of high and low on two axes (developmental-language and psychiatric), replicate. Their borders are soft: a continuous model fits better and predicts new years as well as eight classes, finer class solutions do not replicate, and medical and sensory conditions show up as a third axis or as a small unstable fifth class. Severity follows the developmental axis.

Every number is from StepMix and will be re-verified with the from-scratch code once it lands. The paper will carry footnotes until then.

## What waits on the from-scratch code

The three stubs are in `src/nsch_subtypes/lca.py`, `hierarchical.py` and `partition.py`; the contract is `docs/implementation-spec.md`. Order of value:

1. `fit_lca`. Every step from 3 onward uses it. When it passes `tests/test_implementations.py`, run

       BACKEND=ours bash scripts/sweep_k.sh 4 5

   and compare the printed table with the reference one in `results/sweep.log` (the sweep prints the agreement between the two backends at the end). Agreement within rounding is the statement the paper needs ("results were identical with our implementation and with StepMix").

2. `jaccard_distances` and `cluster_hierarchical`. Expected result on the real data: one cluster with nearly every child, the 517 zero-condition children in a second, and a few tiny outlier clusters. That is correct behavior; it matches SciPy (`reference.cluster_scipy`) and the test checks that it does. The dendrogram it produces is a figure in the paper, as the picture of why distance methods fail here.

3. `cluster_kmeans`. The library version is what the paper reports now. Note that k-means on this data has several near-tied solutions, so expect the from-scratch version to match scikit-learn's ARI against the classes only within the 0.54 to 0.57 range seen across seeds, not exactly.

## Things decided, so nobody re-decides them

- K = 4 is the main result, chosen as the largest K whose classes are all bootstrap-stable. Step 14 shows a simulated continuum is stable at K = 4 and 5 too, so the paper calls K = 4 the finest reproducible division, not evidence of four groups. K = 5 is the sensitivity analysis. Held-out likelihood is reported as "how much structure, which kind of model", not as the K rule. Reasoning in the checklist doc under "Decisions".
- The second method is k-means, 200 starts from each of five seeds, lowest inertia reported with the seed range. The hierarchical result is reported as the failure it is, with the explanation.
- The paper's framing: four broad profiles on two main axes replicate, borders are soft, the structure is mostly continuous. Not "four subtypes", and not "no structure".
- The severity test is reported under both readings of the pre-fit wording, with the comparator models.
- A factor mixture model (classes plus continuous variation within them) is future work, not in this project.
- Survey weights are used only to describe. The weighted-fit check agrees with the unweighted one at ARI 0.78, and at 0.998 among confidently placed children.
- Children with any missing flag are dropped (8.4%). The dropped-vs-kept table is in `results_k4/dropped_vs_kept.csv`; they are about the same age and slightly more often rated severe (12% against 10%), which goes in limitations.

## Things not yet done

- The AI-disclosure line in the README.
- The K = 7 bootstrap here used 100 resamples instead of 200 (the run was interrupted). Rerun with `make sweep KS=7` on a machine that can leave it running, if 200 everywhere matters for the paper.
- Paper and final presentation dates in the README.
- Updating the checkpoint slide deck with the results figures.
- The paper itself: outline and methods draft in `paper/`.

## How to rerun anything

    uv sync --group dev                      # once
    make test                                # unit tests, synthetic data
    make smoke                               # every script once at tiny settings
    make fit                                 # step 3 into results/
    make sweep KS="4 5"                      # steps 4 to 12 at K=4 and K=5, then step 13 and the figures
    make choose figures                      # just step 13 and the figures, from existing outputs
    make download                            # only to rebuild the analysis table from the Census files

`results/` and `results_k*/` are gitignored. When the analysis freezes we commit the two K folders deliberately with `git add -f`.
