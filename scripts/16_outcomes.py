"""Step 16 of 16. What do the classes predict? Three held-out outcomes, and the pandemic years.

Why this step exists
--------------------
Step 7 asked whether a child's class tells us anything about the parent's
rating of autism severity beyond a simple count of conditions. A subtype
that matters should also say something about what a child needs. The survey
asks two more questions the clustering never sees:

- did the child need health care in the past 12 months and not get it
  (``forgone_care``: no, yes), and
- how many times did the child visit an emergency room in the past 12 months
  (``er_visits``: none, one, two or more).

Three survey years fall in the COVID-19 pandemic, when care was disrupted.
The rules for handling them were written into
``docs/validation-checklist.md`` (decision 5) before anything here was run.

What happens here
-----------------
1. Fit the two-dimensional latent trait model (as in step 7) and place the
   2025 children on its axes and in the frozen classes, without refitting.
2. For each outcome (severity, forgone care, ER visits), run step 7's
   comparison of count, class and both, with the same comparators, now
   adjusted for survey year as well as age and sex.
3. Score every model on the 2025 children, fit on 2016 to 2024 only.
4. The period test: does adding a class-by-period interaction (before
   2016-19, during 2020-21, after 2022-24) improve the count-plus-class model?
   Likelihood-ratio test, judged at 0.05 / 3 for three outcomes.
5. Weighted rates of each outcome by class, overall and in each period.
6. Refit the K-class model without 2020 and 2021 and compare it with the main
   classes on the same children.

Outputs, under ``--results``::

    outcome_models_{outcome}.csv     every model's BIC and cross-validated log-loss
    outcome_by_class_{outcome}.csv   weighted share at each level, per class
    outcome_by_period_{outcome}.csv  the same, per class and period
    outcome_heldout_2025.csv         every model's log-loss on the 2025 children
    outcomes.json                    the planned tests, the period tests and the refit

Usage::

    python scripts/16_outcomes.py --results results_k4 --k 4
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import MAIN_YEARS, backend_or_exit, load_analysis, load_lca_fit, log, make_parser, resolve_k, save_json
from nsch_subtypes.data import to_flag_matrix
from nsch_subtypes.describe import outcome_by_class
from nsch_subtypes.latent_trait import fit_latent_trait, latent_trait_scores
from nsch_subtypes.validate import (
    ARI_PASS,
    SEVERITY_ORDER,
    ari_with_interval,
    class_probabilities,
    match_classes,
    outcome_count_vs_class,
    outcome_heldout_loss,
    outcome_period_test,
    relabel_to_match,
)

# Each outcome's levels, lowest first.
OUTCOMES = {
    "severity": SEVERITY_ORDER,
    "forgone_care": ["no", "yes"],
    "er_visits": ["none", "one", "two or more"],
}

# Survey years grouped by the pandemic, as fixed in the validation checklist (decision 5).
PERIODS = {"before": range(2016, 2020), "during": range(2020, 2022), "after": range(2022, 2025)}
PANDEMIC_YEARS = [2020, 2021]
PERIOD_ALPHA = 0.05 / len(OUTCOMES)


def planned_test(models: pd.DataFrame) -> dict:
    """The verdicts step 7 reports, for one outcome's model table."""
    by_model = models.set_index("model")
    return {
        "both_beats_count": bool(
            by_model.loc["both", "bic"] < by_model.loc["count", "bic"] and by_model.loc["both", "cv_log_loss"] < by_model.loc["count", "cv_log_loss"]
        ),
        "class_alone_beats_count": bool(by_model.loc["class", "bic"] < by_model.loc["count", "bic"]),
        "best_model_by_bic": by_model["bic"].idxmin(),
        "best_model_by_cv_log_loss": by_model["cv_log_loss"].idxmin(),
        "bic": by_model["bic"].round(1).to_dict(),
        "cv_log_loss": by_model["cv_log_loss"].round(4).to_dict(),
    }


def main() -> None:
    parser = make_parser("Held-out outcomes and the pandemic years")
    parser.add_argument("--k", type=int, default=None, help="number of classes (default: the choice from step 13)")
    parser.add_argument("--n-starts", type=int, default=20, help="random starts for the refit without 2020 and 2021")
    args = parser.parse_args()

    fit_lca = backend_or_exit(args.backend, "lca")["lca"]
    k = resolve_k(args)
    table, flags = load_analysis(MAIN_YEARS)
    table_2025, _ = load_analysis([2025])
    X, X_2025 = to_flag_matrix(table, flags), to_flag_matrix(table_2025, flags)
    fit = load_lca_fit(args.results, k, table, flags)
    log(f"{len(table)} children 2016-2024, {len(table_2025)} in 2025; K={k}")

    # ---- the frozen models applied to 2025 ------------------------------------------------
    log("fitting the two-dimensional latent trait model")
    trait = fit_latent_trait(X, n_dims=2, seed=args.seed)
    probs_2025 = class_probabilities(X_2025, fit.class_sizes, fit.condition_rates)
    labels_2025 = probs_2025.argmax(axis=1)
    scores_2025 = latent_trait_scores(trait, X_2025)

    summary: dict = {"k": k, "backend": args.backend, "periods": {name: [min(s), max(s)] for name, s in PERIODS.items()}, "outcomes": {}}
    heldout = []
    for outcome, levels in OUTCOMES.items():
        log(f"--- {outcome} ({' < '.join(levels)})")

        # Step 7's comparison, adjusted for survey year.
        models = outcome_count_vs_class(
            table, outcome, levels, fit.labels, flags, seed=args.seed, responsibilities=fit.responsibilities, factor_scores=trait.scores, adjust_for_year=True
        )
        models.to_csv(args.results / f"outcome_models_{outcome}.csv", index=False)
        verdict = planned_test(models)
        log(models.round(3).to_string(index=False))
        log(f"   count + class beats count: {'yes' if verdict['both_beats_count'] else 'no'}; best by BIC: {verdict['best_model_by_bic']}")

        # The same models fit on 2016-2024 and scored on 2025.
        scored = outcome_heldout_loss(
            table,
            fit.labels,
            table_2025,
            labels_2025,
            outcome,
            levels,
            flags,
            train_responsibilities=fit.responsibilities,
            test_responsibilities=probs_2025,
            train_scores=trait.scores,
            test_scores=scores_2025,
        )
        scored.insert(0, "outcome", outcome)
        heldout.append(scored)
        log("   2025 log-loss vs count: " + ", ".join(f"{m} {d:+.4f}" for m, d in zip(scored["model"], scored["vs_count"], strict=True)))

        # Did the class links change across the pandemic periods?
        period_test = outcome_period_test(table, outcome, levels, fit.labels, flags, PERIODS)
        period_test["changed"] = bool(period_test["p_value"] < PERIOD_ALPHA)
        period_test["alpha"] = PERIOD_ALPHA
        log(f"   class x period: LR {period_test['lr_statistic']:.1f} on {period_test['df']} df, p = {period_test['p_value']:.3g}")

        # Weighted rates by class, overall and in each period.
        outcome_by_class(table, fit.labels, outcome, levels).to_csv(args.results / f"outcome_by_class_{outcome}.csv", index=False)
        per_period = []
        for name, span in PERIODS.items():
            rates = outcome_by_class(table, fit.labels, outcome, levels, rows_mask=table["year"].isin(list(span)).to_numpy())
            rates.insert(0, "period", name)
            per_period.append(rates)
        pd.concat(per_period, ignore_index=True).to_csv(args.results / f"outcome_by_period_{outcome}.csv", index=False)

        summary["outcomes"][outcome] = {"levels": levels, "n": int(table[outcome].notna().sum()), "planned_test": verdict, "period_test": period_test}

    heldout_table = pd.concat(heldout, ignore_index=True)
    heldout_table.to_csv(args.results / "outcome_heldout_2025.csv", index=False)
    for outcome in OUTCOMES:
        rows = heldout_table[heldout_table["outcome"] == outcome].set_index("model")
        summary["outcomes"][outcome]["heldout_2025"] = {
            "n": int(rows["n_test"].iloc[0]),
            "best_model": rows["test_log_loss"].idxmin(),
            "log_loss": rows["test_log_loss"].round(4).to_dict(),
        }

    # ---- the classes without the pandemic years --------------------------------------------
    outside = ~table["year"].isin(PANDEMIC_YEARS).to_numpy()
    log(f"refitting K={k} on the {int(outside.sum())} children outside {PANDEMIC_YEARS}")
    refit = fit_lca(X[outside], k, n_starts=args.n_starts, seed=args.seed)
    perm = match_classes(fit.condition_rates, refit.condition_rates)
    agreement = ari_with_interval(fit.labels[outside], relabel_to_match(refit.labels, perm), seed=args.seed)
    rate_gap = np.abs(fit.condition_rates - refit.condition_rates[perm])
    worst_class, worst_flag = np.unravel_index(rate_gap.argmax(), rate_gap.shape)
    summary["without_pandemic_years"] = {
        **agreement,
        "pass": agreement["ari"] >= ARI_PASS,
        "pass_line": ARI_PASS,
        "n": int(outside.sum()),
        "class_sizes_main": fit.class_sizes,
        "class_sizes_refit_matched": refit.class_sizes[perm],
        "largest_rate_difference": float(rate_gap.max()),
        "largest_rate_difference_at": {"class": int(worst_class), "flag": flags[worst_flag]},
    }
    log(
        f"   ARI with the main classes {agreement['ari']:.3f} (95% interval {agreement['ci_low']:.3f} to {agreement['ci_high']:.3f}); "
        f"largest rate difference {rate_gap.max():.3f} ({flags[worst_flag]}, class {worst_class})"
    )

    save_json(summary, args.results / "outcomes.json")
    log(f"wrote outcome tables and outcomes.json to {args.results}")


if __name__ == "__main__":
    main()
