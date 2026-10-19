"""Step 6 of 8. Does each class survive when the sample is shaken?

What happens here
-----------------
The bootstrap: draw a new sample of the same size by picking children at
random with replacement (some appear twice, some not at all), refit the
model, and see whether each original class comes back. Repeat ``--n-boot``
times (200 by default; this is the slowest step).

For each class and each resample we record the Jaccard overlap between the
original class and the refit cluster that matches it best. Reading the
average (Hennig 2007):

    >= 0.85   highly stable
    >= 0.75   stable                       <- our pass line
    >= 0.60   there is a pattern, but membership is doubtful
    <= 0.50   the class dissolved

Outputs, under ``results/``::

    stability.csv            one row per class: n, mean and sd of Jaccard, verdict
    stability_jaccard.npy    the full K x n_boot array
    figures/stability.png

Usage::

    python scripts/06_bootstrap.py --backend reference --n-boot 200
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import MAIN_YEARS, backend_or_exit, load_analysis, load_lca_fit, log, make_parser, read_best_k
from nsch_subtypes.data import to_flag_matrix
from nsch_subtypes.plots import stability_plot
from nsch_subtypes.validate import bootstrap_stability, match_classes, relabel_to_match


def main() -> None:
    parser = make_parser("Bootstrap stability per class")
    parser.add_argument("--n-boot", type=int, default=200, help="number of resamples")
    parser.add_argument("--n-starts", type=int, default=20, help="random starts per refit")
    parser.add_argument("--k", type=int, default=None, help="override the K chosen in step 3")
    args = parser.parse_args()

    fit_lca = backend_or_exit(args.backend, "lca")["lca"]
    k = args.k or read_best_k(args.results)
    table, flags = load_analysis(MAIN_YEARS)
    flag_matrix = to_flag_matrix(table, flags)
    original = load_lca_fit(args.results, k)
    original_labels = original["responsibilities"].argmax(axis=1)

    def refit_and_relabel(resampled_matrix: np.ndarray, n_classes: int) -> np.ndarray:
        """Fit on a resample and renumber its classes to match the original model's."""
        fit = fit_lca(resampled_matrix, n_classes, n_starts=args.n_starts, seed=args.seed)
        perm = match_classes(original["condition_rates"], fit.condition_rates)
        return relabel_to_match(fit.labels, perm)

    def report_progress(done: int) -> None:
        if done % 20 == 0:
            log(f"  {done}/{args.n_boot}")

    log(f"{args.n_boot} bootstrap refits at K={k} with {args.n_starts} starts each")
    stability = bootstrap_stability(
        flag_matrix,
        original_labels,
        refit_and_relabel,
        k,
        n_boot=args.n_boot,
        seed=args.seed,
        progress=report_progress,
    )

    summary = pd.DataFrame(
        {
            "class": range(k),
            "n": np.bincount(original_labels, minlength=k),
            "mean_jaccard": stability.mean,
            "sd_jaccard": stability.jaccard.std(axis=1),
            "verdict": stability.verdicts(),
        }
    )
    summary.to_csv(args.results / "stability.csv", index=False)
    np.save(args.results / "stability_jaccard.npy", stability.jaccard)
    stability_plot(stability.mean, args.results / "figures" / "stability.png")
    log(summary.to_string(index=False))


if __name__ == "__main__":
    main()
