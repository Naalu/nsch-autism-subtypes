"""Step 7 of 8. Describe the classes in plain numbers, and run the count-vs-class test.

What happens here
-----------------
1. Using the survey weights for the first time, estimate what share of U.S.
   autistic children aged 6 to 17 falls in each class, with 95% confidence
   intervals.
2. Within each class, the weighted distribution of parent-rated severity
   (mild / moderate / severe) and the mean age and share female.
3. The planned null test: does knowing a child's class predict severity any
   better than counting the child's conditions? Three ordinal
   regression models are compared by BIC and cross-validated log-loss.

Outputs, under ``results/``::

    shares.csv              weighted share of each class with 95% CI
    severity_by_class.csv   weighted severity distribution per class
    demographics.csv        age and sex per class
    count_vs_class.csv      the three severity models side by side

Usage::

    python scripts/07_describe.py
"""

from __future__ import annotations

from _common import MAIN_YEARS, load_analysis, load_lca_fit, log, make_parser, read_best_k
from nsch_subtypes.describe import class_shares, demographics_by_class, severity_by_class
from nsch_subtypes.validate import severity_count_vs_class


def main() -> None:
    parser = make_parser("Describe the classes", backend=False)
    parser.add_argument("--k", type=int, default=None, help="override the K chosen in step 3")
    args = parser.parse_args()

    k = args.k or read_best_k(args.results)
    table, flags = load_analysis(MAIN_YEARS)
    fit = load_lca_fit(args.results, k)
    responsibilities = fit["responsibilities"]
    labels = responsibilities.argmax(axis=1)

    # ---- national shares -------------------------------------------------------------
    shares = class_shares(table, responsibilities, pooled_years=len(MAIN_YEARS))
    shares.to_csv(args.results / "shares.csv", index=False)
    log("weighted share of US autistic children aged 6-17 in each class (2016-24 pooled):")
    for _, row in shares.iterrows():
        log(f"  class {int(row['class'])}: {row['share']:.1%}  (95% CI {row['ci_low']:.1%} to {row['ci_high']:.1%}, n={int(row['n_modal'])})")

    # ---- who is in each class ----------------------------------------------------------
    severity_by_class(table, labels).to_csv(args.results / "severity_by_class.csv", index=False)
    demographics_by_class(table, labels).to_csv(args.results / "demographics.csv", index=False)

    # ---- the count-vs-class test ----------------------------------------------------
    comparison = severity_count_vs_class(table, labels, flags, seed=args.seed)
    comparison.to_csv(args.results / "count_vs_class.csv", index=False)
    log("severity ~ count vs class (lower is better):")
    log(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
