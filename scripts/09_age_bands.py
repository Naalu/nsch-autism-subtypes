"""Check 9. Are the classes an age effect in disguise?

Why this check exists
---------------------
At K=4 the psychiatric class is the oldest (mean age about 13) and the
developmental class the youngest (about 10.6). Anxiety and depression are
diagnosed more as children get older; developmental delay is diagnosed early.
So a skeptic could say the "psychiatric axis" is partly just "older child".

What happens here
-----------------
1. Split the pooled 2016-2024 children into two age bands, 6 to 11 and
   12 to 17 (these are also the two NSCH questionnaire forms for this range).
2. Fit a K-class model inside each band on its own.
3. Match each band's classes to the pooled model's classes by profile, and
   score agreement between the band model's labels and the pooled model's
   labels on the same children (adjusted Rand index).
4. Report the class profiles and shares within each band.

If the same four corners appear in both bands with similar profiles, the axes
are not an age artifact. Class shares are allowed to differ between bands
(older children will have more psychiatric diagnoses); profiles should not.

Outputs, under ``--results``::

    age_bands.json           ARI per band, profile gaps, shares per band
    profiles_age_6_11.csv, profiles_age_12_17.csv

Usage::

    python scripts/09_age_bands.py --results results_k4 --k 4 --backend reference
"""

from __future__ import annotations

import pandas as pd

from _common import MAIN_YEARS, backend_or_exit, load_analysis, load_lca_fit, log, make_parser, read_best_k, save_json
from nsch_subtypes.data import to_flag_matrix
from nsch_subtypes.validate import adjusted_rand_index, class_probabilities, match_classes, mean_rate_difference, relabel_to_match

BANDS = {"6_11": (6, 11), "12_17": (12, 17)}


def main() -> None:
    parser = make_parser("Refit within age bands and compare with the pooled classes")
    parser.add_argument("--k", type=int, default=None, help="override the K chosen in step 3")
    parser.add_argument("--n-starts", type=int, default=20)
    args = parser.parse_args()

    fit_lca = backend_or_exit(args.backend, "lca")["lca"]
    k = args.k or read_best_k(args.results)
    pooled = load_lca_fit(args.results, k)
    table, flags = load_analysis(MAIN_YEARS)

    out = {"k": k, "backend": args.backend, "bands": {}}
    for name, (lo, hi) in BANDS.items():
        in_band = table["age"].between(lo, hi).to_numpy()
        band_table = table[in_band].reset_index(drop=True)
        X = to_flag_matrix(band_table, flags)

        # Pooled model's view of these children (no refit) vs a fit on the band alone.
        pooled_labels = class_probabilities(X, pooled["class_sizes"], pooled["condition_rates"]).argmax(axis=1)
        band_fit = fit_lca(X, k, n_starts=args.n_starts, seed=args.seed)
        perm = match_classes(pooled["condition_rates"], band_fit.condition_rates)
        band_labels = relabel_to_match(band_fit.labels, perm)

        score = adjusted_rand_index(pooled_labels, band_labels)
        gaps = mean_rate_difference(pooled["condition_rates"], band_fit.condition_rates, perm)

        profiles = pd.DataFrame(band_fit.condition_rates[perm], columns=flags)
        profiles.insert(0, "class_size", band_fit.class_sizes[perm])
        profiles.insert(0, "class", range(k))
        profiles.to_csv(args.results / f"profiles_age_{name}.csv", index=False)

        out["bands"][name] = {
            "n": int(X.shape[0]),
            "ari_pooled_vs_band_fit": score,
            "mean_abs_rate_difference_per_class": gaps,
            "class_sizes_band_fit_matched": band_fit.class_sizes[perm],
            "class_shares_under_pooled_model": [float((pooled_labels == c).mean()) for c in range(k)],
            "entropy_band_fit": band_fit.entropy(),
        }
        log(f"ages {lo}-{hi}: n={X.shape[0]}, ARI pooled-model vs band-fit = {score:.3f}, profile gaps = {[round(float(g), 3) for g in gaps]}")
        log(f"   class sizes in band fit: {[round(float(s), 3) for s in band_fit.class_sizes[perm]]}")

    out["flags"] = flags
    save_json(out, args.results / "age_bands.json")


if __name__ == "__main__":
    main()
