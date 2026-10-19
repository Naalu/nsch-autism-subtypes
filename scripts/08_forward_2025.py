"""Step 8. Forward test on the 2025 survey year.

The 2025 file was released by Census on October 5, 2026, after this project's
design and pass lines were fixed, and nobody on the team looked at it before
the 2016-2024 analysis was frozen. The test:

1. Label the 2025 children with the frozen 2016-2024 model (no refitting).
2. Refit the model from scratch on 2025 alone at the same K.
3. ARI between the two labelings. Pass line: ARI >= 0.65.
4. Weighted 2025 class shares under the frozen model, for comparison with
   the 2016-2024 shares.

    python scripts/08_forward_2025.py --backend reference
"""

from __future__ import annotations

import json

from _common import backend_or_exit, load_analysis, load_lca, log, parser, save_json
from nsch_subtypes.data import flag_matrix
from nsch_subtypes.describe import class_shares
from nsch_subtypes.validate import ARI_PASS, ari, match_classes, posterior, profile_distance, relabel


def main() -> None:
    p = parser("Forward test on 2025")
    p.add_argument("--n-starts", type=int, default=20)
    p.add_argument("--k", type=int, default=None)
    args = p.parse_args()

    fit_lca = backend_or_exit(args.backend, "lca")["lca"]
    k = args.k or json.loads((args.results / "lca_best.json").read_text())["best_k"]
    frozen = load_lca(args.results, k)
    table, flags = load_analysis([2025])
    X = flag_matrix(table, flags)
    log(f"{X.shape[0]} children in 2025 with every flag answered; K={k}")

    resp_frozen = posterior(X, frozen["class_sizes"], frozen["condition_rates"])
    labels_frozen = resp_frozen.argmax(axis=1)
    refit = fit_lca(X, k, n_starts=args.n_starts, seed=args.seed)
    perm = match_classes(frozen["condition_rates"], refit.condition_rates)
    labels_refit = relabel(refit.labels, perm)
    score = ari(labels_frozen, labels_refit)

    shares = class_shares(table, resp_frozen, pooled_years=1)
    shares.to_csv(args.results / "shares_2025.csv", index=False)
    save_json(
        {
            "k": k,
            "backend": args.backend,
            "n_2025": int(X.shape[0]),
            "ari": score,
            "pass": score >= ARI_PASS,
            "pass_line": ARI_PASS,
            "mean_abs_rate_difference_per_class": profile_distance(frozen["condition_rates"], refit.condition_rates, perm),
            "shares_2025_frozen_model": shares["share"].tolist(),
            "rates_refit_matched": refit.condition_rates[perm],
            "flags": flags,
        },
        args.results / "forward_2025.json",
    )
    log(f"ARI, frozen 2016-24 model vs refit on 2025: {score:.3f} ({'pass' if score >= ARI_PASS else 'below the pass line'})")
    for _, r in shares.iterrows():
        log(f"  2025 share, class {int(r['class'])}: {r['share']:.1%} (95% CI {r['ci_low']:.1%} to {r['ci_high']:.1%})")


if __name__ == "__main__":
    main()
