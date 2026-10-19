"""Step 3. Fit the latent class model for a range of K and pick K by BIC.

Fits on the pooled 2016-2024 complete rows. Writes, under results/:

    lca_bic.csv          BIC, log-likelihood, entropy per K (and the shuffled-data BIC if asked)
    lca_k{K}.npz         class sizes, condition rates, responsibilities for every K
    lca_best.json        the chosen K and its summary
    figures/bic.png, figures/profiles_k{K}.png

    python scripts/03_fit_lca.py --backend reference --k-max 8 --permutation-null
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import MAIN_YEARS, backend_or_exit, load_analysis, log, parser, save_json
from nsch_subtypes.data import flag_matrix
from nsch_subtypes.describe import class_profiles
from nsch_subtypes.plots import bic_curve, profile_heatmap
from nsch_subtypes.validate import permute_columns


def main() -> None:
    p = parser("Fit latent class models and choose K by BIC")
    p.add_argument("--k-min", type=int, default=1)
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--n-starts", type=int, default=20)
    p.add_argument("--permutation-null", action="store_true", help="also fit on column-shuffled data as a no-structure baseline")
    args = p.parse_args()

    fit_lca = backend_or_exit(args.backend, "lca")["lca"]
    table, flags = load_analysis(MAIN_YEARS)
    X = flag_matrix(table, flags)
    log(f"{X.shape[0]} children x {X.shape[1]} flags, backend={args.backend}, starts={args.n_starts}")
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "figures").mkdir(exist_ok=True)

    ks = list(range(args.k_min, args.k_max + 1))
    rows, bics = [], []
    for k in ks:
        res = fit_lca(X, k, n_starts=args.n_starts, seed=args.seed)
        bic = res.bic()
        bics.append(bic)
        rows.append(
            {
                "k": k,
                "log_likelihood": res.log_likelihood,
                "n_parameters": res.n_parameters,
                "bic": bic,
                "entropy": res.entropy(),
                "smallest_class_share": float(res.class_sizes.min()),
                "converged": res.converged,
            }
        )
        np.savez_compressed(
            args.results / f"lca_k{k}.npz",
            class_sizes=res.class_sizes,
            condition_rates=res.condition_rates,
            responsibilities=res.responsibilities,
            flags=np.array(flags),
        )
        log(f"K={k}: ll={res.log_likelihood:.1f} BIC={bic:.1f} entropy={res.entropy():.3f} smallest class={res.class_sizes.min():.3f}")

    null_bics = None
    if args.permutation_null:
        Xp = permute_columns(X, seed=args.seed)
        null_bics = [fit_lca(Xp, k, n_starts=args.n_starts, seed=args.seed).bic() for k in ks]
        for row, nb in zip(rows, null_bics, strict=False):
            row["bic_shuffled"] = nb

    best_k = ks[int(np.argmin(bics))]
    pd.DataFrame(rows).to_csv(args.results / "lca_bic.csv", index=False)
    best = next(r for r in rows if r["k"] == best_k)
    save_json(
        {"best_k": best_k, "backend": args.backend, "n_starts": args.n_starts, "n_children": int(X.shape[0]), "flags": flags, **best},
        args.results / "lca_best.json",
    )

    bic_curve(ks, bics, args.results / "figures" / "bic.png", null_bic=null_bics)
    z = np.load(args.results / f"lca_k{best_k}.npz")
    labels = z["responsibilities"].argmax(axis=1)
    profiles = class_profiles(X, labels, flags)
    profiles.to_csv(args.results / f"profiles_k{best_k}.csv", index=False)
    profile_heatmap(profiles, flags, args.results / "figures" / f"profiles_k{best_k}.png", title=f"Latent classes, K={best_k} (BIC)")
    log(f"chose K={best_k}")


if __name__ == "__main__":
    main()
