"""Step 7 of 14. Describe the classes in plain numbers, and run the count-vs-class test.

What happens here
-----------------
1. Using the survey weights for the first time, estimate what share of U.S.
   autistic children aged 6 to 17 falls in each class, with 95% confidence
   intervals.
2. Within each class, the weighted distribution of parent-rated severity
   (mild / moderate / severe) and the mean age and share female.
3. The planned null test: does knowing a child's class add anything to
   counting the child's conditions? The criterion, set before any model was
   fit: the model with count and class together must beat the count alone
   on both BIC and cross-validated log-loss. Comparators added later (class
   probabilities, the continuous model's two scores, all 18 flags) are
   reported beside it so the classes are judged against the other ways of
   summarizing the same data.

Outputs, under ``results/``::

    shares.csv              weighted share of each class with 95% CI
    severity_by_class.csv   weighted severity distribution per class
    demographics.csv        age and sex per class
    count_vs_class.csv      every severity model side by side
    severity_test.json      the planned test's verdict

Usage::

    python scripts/07_describe.py
"""

from __future__ import annotations

from _common import MAIN_YEARS, load_analysis, load_lca_fit, log, make_parser, resolve_k, save_json
from nsch_subtypes.data import to_flag_matrix
from nsch_subtypes.describe import class_shares, demographics_by_class, severity_by_class
from nsch_subtypes.latent_trait import fit_latent_trait
from nsch_subtypes.validate import severity_count_vs_class


def main() -> None:
    parser = make_parser("Describe the classes", backend=False)
    parser.add_argument("--k", type=int, default=None, help="number of classes (default: the choice from step 13)")
    args = parser.parse_args()

    k = resolve_k(args)
    table, flags = load_analysis(MAIN_YEARS)
    fit = load_lca_fit(args.results, k, table, flags)

    # ---- national shares -------------------------------------------------------------
    shares = class_shares(table, fit.responsibilities, pooled_years=len(MAIN_YEARS))
    shares.to_csv(args.results / "shares.csv", index=False)
    log("weighted share of US autistic children aged 6-17 in each class, averaged over 2016-24 (unweighted model, weighted description):")
    for _, row in shares.iterrows():
        log(f"  class {int(row['class'])}: {row['share']:.1%}  (95% CI {row['ci_low']:.1%} to {row['ci_high']:.1%}, n={int(row['n_modal'])})")

    # ---- who is in each class ----------------------------------------------------------
    severity_by_class(table, fit.labels).to_csv(args.results / "severity_by_class.csv", index=False)
    demographics_by_class(table, fit.labels).to_csv(args.results / "demographics.csv", index=False)

    # ---- the planned null test, with comparators ------------------------------------------------
    scores = fit_latent_trait(to_flag_matrix(table, flags), n_dims=2, seed=args.seed).scores
    comparison = severity_count_vs_class(table, fit.labels, flags, seed=args.seed, responsibilities=fit.responsibilities, factor_scores=scores)
    comparison.to_csv(args.results / "count_vs_class.csv", index=False)
    by_model = comparison.set_index("model")
    passed = bool(by_model.loc["both", "bic"] < by_model.loc["count", "bic"] and by_model.loc["both", "cv_log_loss"] < by_model.loc["count", "cv_log_loss"])
    save_json(
        {
            "k": k,
            "criterion": "count + class beats count alone on BIC and on cross-validated log-loss (set before fitting)",
            "pass": passed,
            "bic_count": by_model.loc["count", "bic"],
            "bic_both": by_model.loc["both", "bic"],
            "bic_class_alone": by_model.loc["class", "bic"],
            "class_alone_beats_count": bool(by_model.loc["class", "bic"] < by_model.loc["count", "bic"]),
            "best_model_by_bic": by_model["bic"].idxmin(),
        },
        args.results / "severity_test.json",
    )
    log("severity models (lower is better):")
    log(comparison.round(3).to_string(index=False))
    log(f"planned test (count + class beats count): {'pass' if passed else 'fail'}")


if __name__ == "__main__":
    main()
