# Team notes

Where the project stands, what is waiting on what, and how to pick it up. Updated October 19, 2026.

## Where we are

The pipeline runs end to end on the real 2016 to 2025 files with the library backend (`--backend reference`). Every check in the proposal has been run, plus four added after the first results. The findings are summarized in the README status section and argued in `docs/validation-checklist.md`; the figures are in `reports/figures/results/`.

Short version: four classes replicate, finer solutions do not, and the four are the corners of two axes (developmental-language and psychiatric). A two-dimensional continuous model fits better than any number of classes. Severity follows the developmental axis only.

Every number is from StepMix and will be re-verified with the from-scratch code once it lands. The paper will carry footnotes until then.

## What the from-scratch code unlocks

The three stubs are in `src/nsch_subtypes/lca.py`, `hierarchical.py` and `partition.py`; the contract is `docs/implementation-spec.md`. Order of value:

1. `fit_lca`. Every step from 3 onward uses it. When it passes `tests/test_implementations.py`, run

       BACKEND=ours bash scripts/sweep_k.sh 4 5

   and compare the printed table with the reference one in `results/sweep.log`. Agreement within rounding is the statement the paper needs ("results were identical with our implementation and with StepMix").

2. `jaccard_distances` and `cluster_hierarchical`. Expected result on the real data: one cluster with nearly every child, the 517 zero-condition children in a second, and a few tiny outlier clusters. That is correct behavior; it matches SciPy (`reference.cluster_scipy`) and the test checks that it does. The dendrogram it produces is a figure in the paper, as the picture of why distance methods fail here.

3. `cluster_kmeans`, if we decide the course requirement wants a third method. The library version is what the paper reports now.

## Things decided, so nobody re-decides them

- K=4 is the main result, K=5 the sensitivity analysis. Rule and reasoning in the checklist doc under "Decisions taken".
- The second method is k-means; the hierarchical result is reported as the failure it is, with the explanation.
- The paper's framing is "two dimensions, four stable regions", not "four subtypes".
- Survey weights are used only to describe. The weighted-fit check shows it does not matter (ARI 0.79).
- Children with any missing flag are dropped (8.4%). The dropped-vs-kept table is in `results_k4/dropped_vs_kept.csv`; they are slightly older and slightly more often rated severe, which goes in limitations.

## Things not yet done

- The AI-disclosure line in the README.
- Paper and final presentation dates in the README.
- Updating the checkpoint slide deck with the results figures.
- The paper itself: outline and methods draft in `paper/`.

## How to rerun anything

    uv sync --group dev                      # once
    make test                                # unit tests, synthetic data
    make data                                # steps 1 and 2 (downloads Census files)
    make all BACKEND=reference               # steps 3 to 8 into results/
    bash scripts/sweep_k.sh 4 5              # steps 4 to 12 at K=4 and K=5 into results_k4/, results_k5/
    uv run python scripts/make_results_figures.py --k 4

`results/` and `results_k*/` are gitignored. When the analysis freezes we commit the two K folders deliberately with `git add -f`.
