"""Step 3 of 8. Fit the latent class model for several K and choose K by BIC.

This is the main analysis. Everything after it is a check on what comes out
of it.

What happens here
-----------------
1. Load the 2016-2024 children with every flag answered (about 8,900 rows,
   18 columns of 0/1).
2. For each K from ``--k-min`` to ``--k-max``, fit a K-class latent class
   model from ``--n-starts`` random starts and record its log-likelihood,
   BIC, and entropy. Save the fitted model for every K so later steps can
   use any of them.
3. Optionally (``--permutation-null``) shuffle each column independently and
   fit the same models to the shuffled data. That shows what the BIC curve
   looks like when there is no structure to find.
4. Pick the K with the lowest BIC, save its condition profiles, and draw the
   BIC curve and a profile heatmap.

Outputs, under ``results/``::

    lca_bic.csv              one row per K: log-likelihood, BIC, entropy, smallest class share
    lca_k{K}.npz             class sizes, condition rates, responsibilities for each K
    lca_best.json            the chosen K and its summary
    profiles_k{K}.csv        condition rates by class for the chosen K
    figures/bic.png, figures/profiles_k{K}.png

Usage::

    python scripts/03_fit_lca.py --backend reference --k-max 8 --permutation-null
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import MAIN_YEARS, backend_or_exit, load_analysis, log, make_parser, save_json
from nsch_subtypes.data import to_flag_matrix
from nsch_subtypes.describe import class_profiles
from nsch_subtypes.plots import bic_curve, profile_heatmap
from nsch_subtypes.validate import shuffle_each_column


def main() -> None:
    parser = make_parser("Fit latent class models and choose K by BIC")
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
                "converged": fit.converged,
            }
        )
        np.savez_compressed(
            args.results / f"lca_k{k}.npz",
            class_sizes=fit.class_sizes,
            condition_rates=fit.condition_rates,
            responsibilities=fit.responsibilities,
            flags=np.array(flags),
        )
        log(f"K={k}: ll={fit.log_likelihood:.1f} BIC={bic:.1f} entropy={fit.entropy():.3f} smallest class={fit.class_sizes.min():.3f}")

    # ---- the no-structure baseline ------------------------------------------
    null_bics = None
    if args.permutation_null:
        shuffled = shuffle_each_column(flag_matrix, seed=args.seed)
        null_bics = [fit_lca(shuffled, k, n_starts=args.n_starts, seed=args.seed).bic() for k in ks]
        for row, null_bic in zip(summary_rows, null_bics, strict=False):
            row["bic_shuffled"] = null_bic

    # ---- choose K and save ----------------------------------------------------
    best_k = ks[int(np.argmin(bics))]
    pd.DataFrame(summary_rows).to_csv(args.results / "lca_bic.csv", index=False)
    best_row = next(row for row in summary_rows if row["k"] == best_k)
    save_json(
        {"best_k": best_k, "backend": args.backend, "n_starts": args.n_starts, "n_children": int(n_children), "flags": flags, **best_row},
        args.results / "lca_best.json",
    )

    bic_curve(ks, bics, args.results / "figures" / "bic.png", null_bic=null_bics)

    best_fit = np.load(args.results / f"lca_k{best_k}.npz")
    labels = best_fit["responsibilities"].argmax(axis=1)
    profiles = class_profiles(flag_matrix, labels, flags)
    profiles.to_csv(args.results / f"profiles_k{best_k}.csv", index=False)
    profile_heatmap(profiles, flags, args.results / "figures" / f"profiles_k{best_k}.png", title=f"Latent classes, K={best_k} (BIC)")
    log(f"chose K={best_k}")


if __name__ == "__main__":
    main()
