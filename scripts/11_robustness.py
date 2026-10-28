"""Step 11 of 14. Three small robustness checks on the chosen model.

Each answers a question a reviewer will ask.

1. Does fitting with the survey weights change the classes? The main
   analysis fits unweighted and uses weights only to describe. Here the model
   is refit with each child's weight as a frequency weight and the labels
   are compared with the unweighted fit.

2. Who did we drop? The models use children with every flag answered. This
   compares the dropped children with the kept ones on age, sex, severity
   and how many conditions they did report, so the paper can say whether
   complete-case analysis is likely to bias anything.

3. Is the solution the global one? Refit from several different seeds and
   report pairwise agreement between the resulting labelings. If every seed
   lands on the same classes, the 20-start fit is not a lucky local optimum.

The weighted refit uses the reference backend regardless of ``--backend``,
because the from-scratch interface has no weight argument; that is noted in
the output.

Outputs, under ``--results``::

    robustness.json
    dropped_vs_kept.csv

Usage::

    python scripts/11_robustness.py --results results_k4 --k 4
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import ANALYSIS_PATH, MAIN_YEARS, backend_or_exit, kept_flags, load_analysis, load_lca_fit, log, make_parser, resolve_k, save_json
from nsch_subtypes.data import to_flag_matrix
from nsch_subtypes.validate import adjusted_rand_index, ari_with_interval, match_classes, mean_rate_difference, relabel_to_match


def weighted_refit(X: np.ndarray, weights: np.ndarray, k: int, n_starts: int, seed: int):
    """Fit StepMix with the survey weights as sample weights (pseudo-maximum likelihood).

    Uses the same tolerance and iteration cap as ``reference.fit_lca_stepmix``
    so the comparison with the unweighted fit is not confounded by different
    convergence settings. Returns (class_sizes, condition_rates, responsibilities).
    """
    from stepmix import StepMix

    model = StepMix(n_components=k, measurement="binary", n_init=n_starts, random_state=seed, abs_tol=1e-6, max_iter=500, verbose=0, progress_bar=0)
    model.fit(X.astype(float), sample_weight=weights / weights.mean())
    params = model.get_parameters()
    return np.asarray(params["weights"]), np.asarray(params["measurement"]["pis"]), model.predict_proba(X.astype(float))


def main() -> None:
    parser = make_parser("Robustness: weighted fit, dropped children, seed sensitivity")
    parser.add_argument("--k", type=int, default=None, help="number of classes (default: the choice from step 13)")
    parser.add_argument("--n-starts", type=int, default=20)
    parser.add_argument("--seeds", default="1,2,3,4,5", help="comma-separated seeds for the sensitivity check")
    args = parser.parse_args()

    fit_lca = backend_or_exit(args.backend, "lca")["lca"]
    k = resolve_k(args)
    table, flags = load_analysis(MAIN_YEARS)
    main_fit = load_lca_fit(args.results, k, table, flags)
    main_labels = main_fit.labels
    X = to_flag_matrix(table, flags)
    out: dict = {"k": k, "backend": args.backend}

    # ---- 1. weighted refit ----------------------------------------------------------
    sizes_w, rates_w, resp_w = weighted_refit(X, table["weight"].to_numpy(), k, args.n_starts, args.seed)
    perm = match_classes(main_fit.condition_rates, rates_w)
    agreement = ari_with_interval(
        main_labels, relabel_to_match(resp_w.argmax(axis=1), perm), main_fit.responsibilities.max(axis=1), resp_w.max(axis=1), seed=args.seed
    )
    ari_w = agreement["ari"]
    weighted_profiles = pd.DataFrame(rates_w[perm], columns=flags)
    weighted_profiles.insert(0, "class_size_weighted_model", sizes_w[perm])
    weighted_profiles.insert(0, "class", range(k))
    weighted_profiles.to_csv(args.results / f"profiles_weighted_k{k}.csv", index=False)
    out["weighted_fit"] = {
        "note": "StepMix with survey weights as sample weights (pseudo-maximum likelihood); class sizes from this model are population shares",
        "ari_vs_unweighted": ari_w,
        "ari_interval": [agreement["ci_low"], agreement["ci_high"]],
        "ari_confident": agreement["ari_confident"],
        "share_confident": agreement["share_confident"],
        "ari_by_confidence": agreement["ari_by_confidence"],
        "class_sizes_weighted_matched": sizes_w[perm],
        "class_sizes_unweighted": main_fit.class_sizes,
        "mean_abs_rate_difference_per_class": mean_rate_difference(main_fit.condition_rates, rates_w, perm),
    }
    log(
        f"weighted vs unweighted fit: ARI {ari_w:.3f} ({agreement['ci_low']:.3f} to {agreement['ci_high']:.3f}); "
        f"sizes weighted {np.round(sizes_w[perm], 3).tolist()} vs {np.round(main_fit.class_sizes, 3).tolist()}"
    )
    log(f"   among the {agreement['share_confident']:.0%} of children both fits classify with probability >= 0.8: ARI {agreement['ari_confident']:.3f}")

    # ---- 2. dropped vs kept ------------------------------------------------------------
    full = pd.read_parquet(ANALYSIS_PATH)
    full = full[full["year"].isin(list(MAIN_YEARS))]
    all_flags = kept_flags()
    complete = ~full[all_flags].isna().any(axis=1)
    full = full.assign(group=np.where(complete, "kept", "dropped"))
    full["n_reported"] = full[all_flags].sum(axis=1)  # NaN treated as 0 here: conditions the parent did report
    full["n_missing"] = full[all_flags].isna().sum(axis=1)
    rows = []
    for group, g in full.groupby("group"):
        rows.append(
            {
                "group": group,
                "n": len(g),
                "mean_age": g["age"].mean(),
                "share_female": (g["sex"] == "female").mean(),
                "severity_mild": (g["severity"] == "mild").mean(),
                "severity_moderate": (g["severity"] == "moderate").mean(),
                "severity_severe": (g["severity"] == "severe").mean(),
                "severity_missing": g["severity"].isna().mean(),
                "mean_conditions_reported": g["n_reported"].mean(),
                "mean_items_missing": g["n_missing"].mean(),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.results / "dropped_vs_kept.csv", index=False)
    log("dropped vs kept children:")
    log(comparison.round(3).to_string(index=False))
    out["dropped_vs_kept"] = comparison.to_dict(orient="records")

    # ---- 3. seed sensitivity ------------------------------------------------------------
    seeds = [int(s) for s in args.seeds.split(",")]
    labelings = []
    for seed in seeds:
        fit = fit_lca(X, k, n_starts=args.n_starts, seed=seed)
        perm = match_classes(main_fit.condition_rates, fit.condition_rates)
        labelings.append((seed, relabel_to_match(fit.labels, perm), fit.log_likelihood))
    pairwise = [{"seed_a": a, "seed_b": b, "ari": adjusted_rand_index(la, lb)} for i, (a, la, _) in enumerate(labelings) for (b, lb, _) in labelings[i + 1 :]]
    vs_main = [{"seed": s, "ari_vs_main": adjusted_rand_index(main_labels, lab), "log_likelihood": ll} for s, lab, ll in labelings]
    out["seed_sensitivity"] = {"vs_main": vs_main, "pairwise": pairwise, "min_pairwise_ari": min((p["ari"] for p in pairwise), default=float("nan"))}
    log(f"seed sensitivity: ARI vs main fit {[round(v['ari_vs_main'], 3) for v in vs_main]}; min pairwise {out['seed_sensitivity']['min_pairwise_ari']:.3f}")

    save_json(out, args.results / "robustness.json")


if __name__ == "__main__":
    main()
