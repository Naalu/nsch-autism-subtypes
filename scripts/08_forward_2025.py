"""Step 8 of 14. The forward test: a survey year nobody had seen when the analysis was designed.

The 2025 NSCH file was released by Census on October 5, 2026, after this
project's design and pass lines were fixed. That makes it the cleanest
replication we can get: there was no way to tune anything to it.

What happens here
-----------------
1. Label the 2025 children with the frozen 2016-2024 model, with no refitting.
2. Fit a fresh model on 2025 alone, at the same K.
3. Match the fresh model's classes to the frozen ones and score the agreement
   between the two labelings with the adjusted Rand index. Pass line: ARI >= 0.65.
4. Report the weighted 2025 class shares under the frozen model, to set
   beside the 2016-2024 shares from step 7.

Outputs, under ``results/``::

    shares_2025.csv       weighted share of each class in 2025 (frozen model)
    forward_2025.json     ARI, pass/fail, profile gaps, the refit's matched profiles

Usage::

    python scripts/08_forward_2025.py --backend reference
"""

from __future__ import annotations

from _common import backend_or_exit, load_analysis, load_lca_fit, log, make_parser, resolve_k, save_json
from nsch_subtypes.data import to_flag_matrix
from nsch_subtypes.describe import class_shares
from nsch_subtypes.validate import (
    ARI_PASS,
    ari_with_interval,
    class_probabilities,
    match_classes,
    mean_rate_difference,
    relabel_to_match,
)


def main() -> None:
    parser = make_parser("Forward test on 2025")
    parser.add_argument("--n-starts", type=int, default=20)
    parser.add_argument("--k", type=int, default=None, help="number of classes (default: the choice from step 13)")
    args = parser.parse_args()

    fit_lca = backend_or_exit(args.backend, "lca")["lca"]
    k = resolve_k(args)
    table, flags = load_analysis([2025])
    frozen = load_lca_fit(args.results, k, flags=flags)  # fit on 2016-24, so only the conditions are checked
    flag_matrix = to_flag_matrix(table, flags)
    log(f"{flag_matrix.shape[0]} children in 2025 with every flag answered; K={k}")

    # 1. The frozen model's view of the 2025 children.
    frozen_probabilities = class_probabilities(flag_matrix, frozen.class_sizes, frozen.condition_rates)
    frozen_labels = frozen_probabilities.argmax(axis=1)

    # 2 and 3. A fresh fit, matched to the frozen classes, then compared.
    refit = fit_lca(flag_matrix, k, n_starts=args.n_starts, seed=args.seed)
    perm = match_classes(frozen.condition_rates, refit.condition_rates)
    refit_labels = relabel_to_match(refit.labels, perm)
    agreement = ari_with_interval(frozen_labels, refit_labels, frozen_probabilities.max(axis=1), refit.responsibilities.max(axis=1), seed=args.seed)
    score = agreement["ari"]

    # 4. Weighted shares for a single year, so pooled_years=1.
    shares = class_shares(table, frozen_probabilities, pooled_years=1)
    shares.to_csv(args.results / "shares_2025.csv", index=False)

    save_json(
        {
            "k": k,
            "backend": args.backend,
            "n_2025": int(flag_matrix.shape[0]),
            **agreement,
            "pass": score >= ARI_PASS,
            "pass_line": ARI_PASS,
            "mean_abs_rate_difference_per_class": mean_rate_difference(frozen.condition_rates, refit.condition_rates, perm),
            "shares_2025_frozen_model": shares["share"].tolist(),
            "rates_refit_matched": refit.condition_rates[perm],
            "flags": flags,
        },
        args.results / "forward_2025.json",
    )
    verdict = "pass" if score >= ARI_PASS else "below the pass line"
    log(f"ARI, frozen 2016-24 model vs refit on 2025: {score:.3f} (95% interval {agreement['ci_low']:.3f} to {agreement['ci_high']:.3f}; {verdict})")
    log(f"   among the {agreement['share_confident']:.0%} of children both models classify with probability >= 0.8: ARI {agreement['ari_confident']:.3f}")
    for _, row in shares.iterrows():
        log(f"  2025 share, class {int(row['class'])}: {row['share']:.1%} (95% CI {row['ci_low']:.1%} to {row['ci_high']:.1%})")


if __name__ == "__main__":
    main()
