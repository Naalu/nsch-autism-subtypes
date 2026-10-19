"""Step 7. Describe the classes: national shares, profiles, severity, and the count test.

The only step that uses survey weights. Writes under results/:

    shares.csv            weighted share of each class with 95% CI
    profiles_k{K}.csv     rate of each flag in each class (from step 3)
    severity_by_class.csv weighted severity distribution per class
    demographics.csv      age and sex per class
    count_vs_class.csv    the planned null test on parent-rated severity

    python scripts/07_describe.py
"""

from __future__ import annotations

import json

from _common import MAIN_YEARS, load_analysis, load_lca, log, parser
from nsch_subtypes.data import flag_matrix
from nsch_subtypes.describe import class_shares, demographics_by_class, severity_by_class
from nsch_subtypes.validate import count_vs_class


def main() -> None:
    p = parser("Describe the classes", backend=False)
    p.add_argument("--k", type=int, default=None)
    args = p.parse_args()

    k = args.k or json.loads((args.results / "lca_best.json").read_text())["best_k"]
    table, flags = load_analysis(MAIN_YEARS)
    X = flag_matrix(table, flags)  # noqa: F841
    lca = load_lca(args.results, k)
    resp = lca["responsibilities"]
    labels = resp.argmax(axis=1)

    shares = class_shares(table, resp, pooled_years=len(MAIN_YEARS))
    shares.to_csv(args.results / "shares.csv", index=False)
    log("weighted share of US autistic children aged 6-17 in each class (2016-24 pooled):")
    for _, r in shares.iterrows():
        log(f"  class {int(r['class'])}: {r['share']:.1%}  (95% CI {r['ci_low']:.1%} to {r['ci_high']:.1%}, n={int(r['n_modal'])})")

    severity_by_class(table, labels).to_csv(args.results / "severity_by_class.csv", index=False)
    demographics_by_class(table, labels).to_csv(args.results / "demographics.csv", index=False)

    test = count_vs_class(table, labels, flags, seed=args.seed)
    test.to_csv(args.results / "count_vs_class.csv", index=False)
    log("severity ~ count vs class (lower is better):")
    log(test.to_string(index=False))


if __name__ == "__main__":
    main()
