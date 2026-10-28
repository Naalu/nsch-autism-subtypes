"""Step 3 of 14. Fit the latent class model for K = 1 to 8.

This is the main model. Steps 4 to 12 check what comes out of it and step 13
compares it with the alternatives and decides which K to report.

What happens here
-----------------
1. Load the 2016 to 2024 children with every flag answered (8,891 rows,
   18 columns of 0/1).
2. For each K from ``--k-min`` to ``--k-max``, fit a K-class model from
   ``--n-starts`` random starts and record its log-likelihood, BIC, entropy,
   smallest class, and how confidently children are assigned (the average
   posterior probability of the class each child is assigned to, "AvePP";
   0.7 and above is the usual rule of thumb). Every fit is saved, together
   with the year and household id of each row, so later steps can check they
   are using the same children.
3. Optionally (``--permutation-null``) shuffle each column independently and
   fit the same models to the shuffled data, to show what the BIC curve looks
   like when there is nothing to find.

This step does not choose K. On these data BIC falls at every K tried
(``lca_fits.json`` records where its minimum is, which is just the largest K),
so the choice is made in step 13.

Outputs, under ``--results``::

    lca_bic.csv          one row per K: log-likelihood, BIC, entropy, smallest class, AvePP
    lca_k{K}.npz         the fitted model for each K
    lca_fits.json        which K were fit, and where the BIC minimum fell
    profiles_k{K}.csv    condition rates by class, for every K
    figures/bic.png

Usage::

    python scripts/03_fit_lca.py --backend reference --k-max 8 --permutation-null
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import MAIN_YEARS, backend_or_exit, load_analysis, load_lca_fit, log, make_parser, save_json, save_lca_fit
from nsch_subtypes.data import to_flag_matrix
from nsch_subtypes.describe import class_profiles
from nsch_subtypes.plots import bic_curve
from nsch_subtypes.validate import shuffle_each_column


def average_posterior_by_class(fit) -> np.ndarray:
    """AvePP: for children assigned to each class, their mean probability of that class."""
    labels = fit.labels
    return np.array([fit.responsibilities[labels == c, c].mean() if (labels == c).any() else np.nan for c in range(fit.n_classes)])


def main() -> None:
    parser = make_parser("Fit latent class models for a range of K")
    parser.add_argument("--k-min", type=int, default=1, help="smallest number of classes to try")
    parser.add_argument("--k-max", type=int, default=8, help="largest number of classes to try")
    parser.add_argument("--n-starts", type=int, default=20, help="random starts per K; the best is kept")
    parser.add_argument("--permutation-null", action="store_true", help="also fit on column-shuffled data as a no-structure baseline")
    args = parser.parse_args()

    fit_lca = backend_or_exit(args.backend, "lca")["lca"]
    table, flags = load_analysis(MAIN_YEARS)
    flag_matrix = to_flag_matrix(table, flags)
    n_children, n_flags = flag_matrix.shape
    log(f"{n_children} children x {n_flags} flags, backend={args.backend}, starts={args.n_starts}")

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "figures").mkdir(exist_ok=True)

    # ---- fit one model per K -------------------------------------------------
    ks = list(range(args.k_min, args.k_max + 1))
    summary_rows = []
    bics = []
    for k in ks:
        fit = fit_lca(flag_matrix, k, n_starts=args.n_starts, seed=args.seed)
        bic = fit.bic()
        bics.append(bic)
        summary_rows.append(
            {
                "k": k,
                "log_likelihood": fit.log_likelihood,
                "n_parameters": fit.n_parameters,
                "bic": bic,
                "entropy": fit.entropy(),
                "smallest_class_share": float(fit.class_sizes.min()),
                "min_avepp": average_posterior_by_class(fit).min() if k > 1 else 1.0,
                "converged": fit.converged,
            }
        )
        save_lca_fit(args.results, fit, table, flags)
        log(f"K={k}: ll={fit.log_likelihood:.1f} BIC={bic:.1f} entropy={fit.entropy():.3f} smallest class={fit.class_sizes.min():.3f}")

    # ---- the no-structure baseline ------------------------------------------
    null_bics = None
    if args.permutation_null:
        shuffled = shuffle_each_column(flag_matrix, seed=args.seed)
        null_bics = [fit_lca(shuffled, k, n_starts=args.n_starts, seed=args.seed).bic() for k in ks]
        for row, null_bic in zip(summary_rows, null_bics, strict=False):
            row["bic_shuffled"] = null_bic

    # ---- save ------------------------------------------------------------------
    pd.DataFrame(summary_rows).to_csv(args.results / "lca_bic.csv", index=False)
    save_json(
        {
            "k_fitted": ks,
            "bic_argmin_k": ks[int(np.argmin(bics))],
            "backend": args.backend,
            "n_starts": args.n_starts,
            "n_children": n_children,
            "flags": flags,
        },
        args.results / "lca_fits.json",
    )
    bic_curve(ks, bics, args.results / "figures" / "bic.png", null_bic=null_bics)
    for k in ks:
        fit = load_lca_fit(args.results, k, table, flags)
        class_profiles(flag_matrix, fit.labels, flags, n_classes=k).to_csv(args.results / f"profiles_k{k}.csv", index=False)
    log(f"BIC is lowest at K={ks[int(np.argmin(bics))]} of {ks[0]} to {ks[-1]}; K is chosen in step 13")


if __name__ == "__main__":
    main()
