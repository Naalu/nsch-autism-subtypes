"""Step 5. Held-out years: do the classes come back in different children?

Fits model A on 2016-2021 and model B on 2022-2024 at the K chosen in step 3.
Both models label the 2022-2024 children; the ARI between those two labelings
is the test. Pass line: ARI >= 0.65. Also reports how far apart the matched
class profiles are.

    python scripts/05_replicate.py --backend reference
"""

from __future__ import annotations

import json

from _common import backend_or_exit, load_analysis, log, parser, save_json
from nsch_subtypes.data import flag_matrix
from nsch_subtypes.validate import ARI_PASS, ari, match_classes, posterior, profile_distance, relabel

FIT_YEARS = range(2016, 2022)
HELD_OUT_YEARS = range(2022, 2025)


def main() -> None:
    p = parser("Held-out-years replication")
    p.add_argument("--n-starts", type=int, default=20)
    p.add_argument("--k", type=int, default=None)
    args = p.parse_args()

    fit_lca = backend_or_exit(args.backend, "lca")["lca"]
    k = args.k or json.loads((args.results / "lca_best.json").read_text())["best_k"]
    table_a, flags = load_analysis(FIT_YEARS)
    table_b, _ = load_analysis(HELD_OUT_YEARS)
    X_a, X_b = flag_matrix(table_a, flags), flag_matrix(table_b, flags)
    log(f"model A: {X_a.shape[0]} children (2016-21); model B: {X_b.shape[0]} children (2022-24); K={k}")

    model_a = fit_lca(X_a, k, n_starts=args.n_starts, seed=args.seed)
    model_b = fit_lca(X_b, k, n_starts=args.n_starts, seed=args.seed)

    labels_a_on_b = posterior(X_b, model_a.class_sizes, model_a.condition_rates).argmax(axis=1)
    perm = match_classes(model_a.condition_rates, model_b.condition_rates)
    labels_b = relabel(model_b.labels, perm)
    score = ari(labels_a_on_b, labels_b)
    dist = profile_distance(model_a.condition_rates, model_b.condition_rates, perm)

    save_json(
        {
            "k": k,
            "backend": args.backend,
            "ari": score,
            "pass": score >= ARI_PASS,
            "pass_line": ARI_PASS,
            "n_fit": int(X_a.shape[0]),
            "n_held_out": int(X_b.shape[0]),
            "class_sizes_a": model_a.class_sizes,
            "class_sizes_b_matched": model_b.class_sizes[perm],
            "mean_abs_rate_difference_per_class": dist,
            "rates_a": model_a.condition_rates,
            "rates_b_matched": model_b.condition_rates[perm],
            "flags": flags,
        },
        args.results / "replicate.json",
    )
    log(f"ARI on the 2022-24 children, model A vs model B: {score:.3f} ({'pass' if score >= ARI_PASS else 'below the pass line'})")
    log("mean absolute difference in condition rates per matched class: " + ", ".join(f"{d:.3f}" for d in dist))


if __name__ == "__main__":
    main()
