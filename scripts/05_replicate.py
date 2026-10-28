"""Step 5 of 14. Do the classes come back in a different set of children?

Most published subtype studies cluster one sample once and stop. The classic
failure is that the clusters describe that particular sample and vanish in
the next one. This step checks for that using survey years as independent
samples.

What happens here
-----------------
1. Fit model A on the 2016-2021 children and model B on the 2022-2024
   children, both at the K chosen in step 3.
2. Label the 2022-2024 children twice: once with model A (a model that never
   saw them) and once with model B (fit on them).
3. Match B's classes to A's by profile and score the agreement between the
   two labelings with the adjusted Rand index. Pass line: ARI >= 0.65.
4. Also report how far apart the matched class profiles are.

If the ARI is high, a model fit on earlier children describes later children
about as well as a model fit on those children themselves, which is what we
would expect of real subtypes.

Output: ``results/replicate.json``.

Usage::

    python scripts/05_replicate.py --backend reference
"""

from __future__ import annotations

from _common import backend_or_exit, load_analysis, log, make_parser, resolve_k, save_json
from nsch_subtypes.data import to_flag_matrix
from nsch_subtypes.validate import (
    ARI_PASS,
    ari_with_interval,
    class_probabilities,
    match_classes,
    mean_rate_difference,
    relabel_to_match,
)

FIT_YEARS = range(2016, 2022)  # model A
HELD_OUT_YEARS = range(2022, 2025)  # model B, and the children both models label


def main() -> None:
    parser = make_parser("Held-out-years replication")
    parser.add_argument("--n-starts", type=int, default=20)
    parser.add_argument("--k", type=int, default=None, help="number of classes (default: the choice from step 13)")
    args = parser.parse_args()

    fit_lca = backend_or_exit(args.backend, "lca")["lca"]
    k = resolve_k(args)

    table_early, flags = load_analysis(FIT_YEARS)
    table_late, _ = load_analysis(HELD_OUT_YEARS)
    X_early = to_flag_matrix(table_early, flags)
    X_late = to_flag_matrix(table_late, flags)
    log(f"model A: {X_early.shape[0]} children (2016-21); model B: {X_late.shape[0]} children (2022-24); K={k}")

    model_a = fit_lca(X_early, k, n_starts=args.n_starts, seed=args.seed)
    model_b = fit_lca(X_late, k, n_starts=args.n_starts, seed=args.seed)

    # Label the held-out children with model A without refitting it.
    probabilities_from_a = class_probabilities(X_late, model_a.class_sizes, model_a.condition_rates)
    labels_from_a = probabilities_from_a.argmax(axis=1)
    # Model B's own labels, renumbered to match A's classes.
    perm = match_classes(model_a.condition_rates, model_b.condition_rates)
    labels_from_b = relabel_to_match(model_b.labels, perm)

    agreement = ari_with_interval(labels_from_a, labels_from_b, probabilities_from_a.max(axis=1), model_b.responsibilities.max(axis=1), seed=args.seed)
    score = agreement["ari"]
    profile_gap = mean_rate_difference(model_a.condition_rates, model_b.condition_rates, perm)

    save_json(
        {
            "k": k,
            "backend": args.backend,
            **agreement,
            "pass": score >= ARI_PASS,
            "pass_line": ARI_PASS,
            "n_fit": int(X_early.shape[0]),
            "n_held_out": int(X_late.shape[0]),
            "class_sizes_a": model_a.class_sizes,
            "class_sizes_b_matched": model_b.class_sizes[perm],
            "mean_abs_rate_difference_per_class": profile_gap,
            "rates_a": model_a.condition_rates,
            "rates_b_matched": model_b.condition_rates[perm],
            "flags": flags,
        },
        args.results / "replicate.json",
    )
    verdict = "pass" if score >= ARI_PASS else "below the pass line"
    log(f"ARI on the 2022-24 children, model A vs model B: {score:.3f} (95% interval {agreement['ci_low']:.3f} to {agreement['ci_high']:.3f}; {verdict})")
    log(f"   among the {agreement['share_confident']:.0%} of children both models classify with probability >= 0.8: ARI {agreement['ari_confident']:.3f}")
    log("mean absolute difference in condition rates per matched class: " + ", ".join(f"{gap:.3f}" for gap in profile_gap))


if __name__ == "__main__":
    main()
