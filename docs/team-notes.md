# Team notes

Where the project stands, what is waiting on what, and how to pick it up.

## Where we are

The pipeline runs end to end on the real 2016 to 2025 files with our from-scratch code, which is the default backend. Every check in the proposal has been run at K = 4 to 8 with 200 bootstrap resamples, plus the checks added after the first results (steps 9 to 15). The findings are summarized in the README and argued in `docs/validation-checklist.md` and the paper; the figures are in `reports/figures/results/`.

Short version: the conditions vary along two continuous axes. Four broad profiles, the combinations of high and low on a developmental-language and a psychiatric axis, replicate, but only as well as the same cut of a smooth continuum does (step 15). A continuous model fits better and predicts new years as well as eight classes, finer class solutions do not replicate, and medical and sensory conditions show up as a third axis or as a small unstable fifth class. Severity follows the developmental axis.

The same pipeline run with StepMix, SciPy and scikit-learn (`BACKEND=reference`, into `results_reference/`) is the check on our code. The latent class fits agree at ARI 0.988, 0.978 and 0.966 at K = 4, 5 and 6, and every K = 4 check with a pass line moves by 0.02 or less. Refit directly, our k-means finds the same clusters as scikit-learn at K = 2 to 4, and our hierarchical tree differs from SciPy's only by tie order, about as much as SciPy differs from itself on shuffled rows. `reports/implementation-comparison.md` has both comparisons.

## Things decided, so nobody re-decides them

- K = 4 is the main result, chosen as the largest K whose classes are all bootstrap-stable. Step 14 shows a simulated continuum is stable at K = 4 and 5 too, so the paper calls K = 4 the finest reproducible division, not evidence of four groups. K = 5 is the sensitivity analysis. Held-out likelihood is reported as "how much structure, which kind of model", not as the K rule. Reasoning in the checklist doc under "Decisions".
- Our from-scratch code is the default backend and the source of every reported number; StepMix is the check.
- The second method is k-means, 200 starts from each of five seeds, lowest inertia reported with the seed range. The hierarchical result is reported as the failure it is, with the explanation.
- The paper's framing: four broad profiles on two main axes replicate, borders are soft, the structure is mostly continuous. Not "four subtypes", and not "no structure".
- The severity test is reported under both readings of the pre-fit wording, with the comparator models.
- A factor mixture model (classes plus continuous variation within them) is future work, not in this project.
- Survey weights are used only to describe. The weighted-fit check agrees with the unweighted one at ARI 0.77, and at 0.999 among confidently placed children.
- Children with any missing flag are dropped (8.4%). The dropped-vs-kept table is in `results_k4/dropped_vs_kept.csv`; they are about the same age and slightly more often rated severe (12% against 10%), which goes in limitations.
- The update presentation covers the instructor's three asks (closest published results, SUVAC, raw and processed data), the from-scratch implementation and the four profiles with their replication checks. The classes-versus-continuum results (the map, the density figure, the stability simulation) are for the final presentation.

## Things not yet done

- The AI-disclosure line in the README.
- The Week 13 update slides, from the results figures.
- The paper sections still marked as stubs (`make paper` prints how many): abstract, the latent class model, the four profiles, severity, K = 5 and the discussion. `paper/outline.md` has the division of labor.
- Confirm the Ward-linkage ARI in `reports/tables/method_comparison.csv` (0.34) with a fresh `scripts/04_crosscheck.py --method alternatives` run; a cloud rerun gave 0.24, probably from how tied merges are broken.
- At the analysis freeze: commit the result folders with `git add -f`, and the final `paper/main.pdf`.
- The final slides need the repository link (and the website link, if we build the site).

## How to rerun anything

    uv sync --group dev                      # once
    make test                                # unit tests, synthetic data
    make smoke                               # every script once at tiny settings, both backends
    make fit                                 # step 3 into results/
    make sweep                               # steps 4 to 16 at K = 4 to 8, then steps 13 and 14 and the figures (4 to 5 hours)
    make choose calibrate figures            # just steps 13 to 15 and the figures, from existing outputs
    make paper                               # build paper/main.pdf
    BACKEND=reference make fit sweep         # the same with StepMix, into results_reference/
    make compare                             # our fits against StepMix's
    make download                            # only to rebuild the analysis table from the Census files

`results*/` folders are gitignored. When the analysis freezes we commit the K folders deliberately with `git add -f`.
