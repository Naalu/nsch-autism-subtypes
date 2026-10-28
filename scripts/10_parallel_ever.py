"""Step 10 of 14. Parallel validation: do the classes survive a different operationalization of the same conditions?

Why this check exists
---------------------
The SUVAC checklist asks whether the analysis is repeated with a second set
of variables measuring the same constructs. The survey gives us one for free:
every condition has an "ever told" item as well as the "currently has" item
the main analysis uses. Building the flags from "ever told" instead changes
every column (more 1s, since children who outgrew a condition are now
flagged) without changing which children or which conditions are involved.

What happens here
-----------------
1. Rebuild the flag matrix for the pooled 2016-2024 children using the
   "ever told" items for every condition. Conditions with no follow-up item
   are unchanged. The same children are used (those complete under the main
   rule) so the two labelings line up row for row.
2. Fit a K-class model to the "ever" flags.
3. Match its classes to the main model's and score agreement with the
   adjusted Rand index. Pass line, as elsewhere: ARI >= 0.65.

Outputs, under ``--results``::

    parallel_ever.json      ARI, profile gaps, class sizes, prevalence under each rule
    profiles_ever_k{K}.csv

Usage::

    python scripts/10_parallel_ever.py --results results_k4 --k 4 --backend reference
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import MAIN_YEARS, backend_or_exit, load_analysis, load_lca_fit, log, make_parser, resolve_k, save_json
from nsch_subtypes.data import to_flag_matrix
from nsch_subtypes.validate import ARI_PASS, ari_with_interval, match_classes, mean_rate_difference, relabel_to_match


def main() -> None:
    parser = make_parser("Parallel validation with the 'ever told' items")
    parser.add_argument("--k", type=int, default=None, help="number of classes (default: the choice from step 13)")
    parser.add_argument("--n-starts", type=int, default=20)
    args = parser.parse_args()

    fit_lca = backend_or_exit(args.backend, "lca")["lca"]
    k = resolve_k(args)
    table, flags = load_analysis(MAIN_YEARS)
    main_fit = load_lca_fit(args.results, k, table, flags)

    # Line the 'ever' flags up with the main analysis rows, child for child.
    ever_columns = [f"ever_{flag}" for flag in flags]
    if not set(ever_columns) <= set(table.columns):
        raise SystemExit("The analysis table has no 'ever told' columns; rerun scripts/02_encode.py.")
    ever = table[ever_columns].set_axis(flags, axis=1)
    keep = ~ever.isna().any(axis=1).to_numpy()
    X_current = to_flag_matrix(table, flags)[keep]
    X_ever = ever.to_numpy()[keep].astype(np.uint8)
    main_labels = main_fit.labels[keep]
    mean_current, mean_ever = X_current.sum(1).mean(), X_ever.sum(1).mean()
    log(f"{keep.sum()} of {len(table)} children have every 'ever' item answered; mean conditions current={mean_current:.2f}, ever={mean_ever:.2f}")

    ever_fit = fit_lca(X_ever, k, n_starts=args.n_starts, seed=args.seed)
    perm = match_classes(main_fit.condition_rates, ever_fit.condition_rates)
    ever_labels = relabel_to_match(ever_fit.labels, perm)
    agreement = ari_with_interval(main_labels, ever_labels, main_fit.responsibilities.max(axis=1)[keep], ever_fit.responsibilities.max(axis=1), seed=args.seed)
    score = agreement["ari"]
    gaps = mean_rate_difference(main_fit.condition_rates, ever_fit.condition_rates, perm)

    profiles = pd.DataFrame(ever_fit.condition_rates[perm], columns=flags)
    profiles.insert(0, "class_size", ever_fit.class_sizes[perm])
    profiles.insert(0, "class", range(k))
    profiles.to_csv(args.results / f"profiles_ever_k{k}.csv", index=False)

    save_json(
        {
            "k": k,
            "backend": args.backend,
            "n": int(keep.sum()),
            **agreement,
            "pass": score >= ARI_PASS,
            "pass_line": ARI_PASS,
            "n_items_differing_between_rules": int((X_current != X_ever).any(axis=0).sum()),
            "mean_abs_rate_difference_per_class": gaps,
            "class_sizes_ever_matched": ever_fit.class_sizes[perm],
            "class_sizes_current": main_fit.class_sizes,
            "prevalence_current": dict(zip(flags, X_current.mean(0).round(4), strict=True)),
            "prevalence_ever": dict(zip(flags, X_ever.mean(0).round(4), strict=True)),
            "flags": flags,
        },
        args.results / "parallel_ever.json",
    )
    verdict = "pass" if score >= ARI_PASS else "below the pass line"
    log(
        f"ARI, 'currently has' classes vs 'ever told' classes: {score:.3f} ({agreement['ci_low']:.3f} to {agreement['ci_high']:.3f}; {verdict}); "
        f"profile gaps {[round(float(g), 3) for g in gaps]}"
    )


if __name__ == "__main__":
    main()
