"""Step 15 of 15. What do the replication checks score when there are no groups to find?

Why this step exists
--------------------
Steps 4, 5, 8 and 9 report how well the K classes come back in other
children (later survey years, the 2025 file, each age band) and under a
second method (k-means). On the real data those scores pass the 0.65 line.
Step 14 showed that bootstrap stability also passes on data simulated from a
smooth two-dimensional continuum with no classes in it, so "stable" does not
mean "groups". The same question has to be asked of the replication checks:
if the children were a continuum, would a four-class partition of one set of
them still reproduce in another set? If it would, then replication tells us
the partition is reproducible, which is worth knowing, but it does not tell
us the pieces are groups. If it would not, the real data have more
class-like structure than the map suggests.

What happens here
-----------------
1. Fit the two-dimensional latent trait model to the real 2016 to 2024
   children (as in steps 12 and 14) and load the K-class model from step 3.
2. For each of ``--replicates`` rounds and each of the two models, simulate
   flags for every real child (2016 to 2024 and 2025), keeping each child's
   real survey year and age so the splits below are the real splits. The
   continuum data contain no classes; the class data contain exactly K.
3. On each simulated dataset run the four checks exactly as the real steps
   do: a K-class fit on the pooled 2016 to 2024 rows (the analogue of step
   3), then held-out years (step 5), the 2025 forward test (step 8), age
   bands (step 9) and k-means against the classes (step 4).
4. Write every ARI beside the real values read from this folder's
   ``replicate.json``, ``forward_2025.json``, ``age_bands.json`` and
   ``crosscheck_kmeans.json``.

Reading the table: a real ARI inside the range the continuum produces says
the check cannot separate groups from a continuum at this K. A real ARI
below the continuum's range says the real partition reproduces less well
than a smooth cloud would; above the class model's range would be
surprising and worth a second look.

Outputs, under ``--results``::

    replication_calibration.csv      one row per (data, replicate, check): ari, n
    replication_calibration.json     the real values, the simulated ranges and the settings

Usage::

    python scripts/15_replication_calibration.py --results results_k4 --k 4
    python scripts/15_replication_calibration.py --results results_k4 --k 4 --replicates 2 --n-starts 5 --kmeans-starts 20 --kmeans-seeds 1   # quicker
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import MAIN_YEARS, backend_or_exit, load_analysis, load_lca_fit, log, make_parser, resolve_k, save_json
from nsch_subtypes.data import to_flag_matrix
from nsch_subtypes.latent_trait import fit_latent_trait, simulate_latent_trait
from nsch_subtypes.lca import simulate_classes
from nsch_subtypes.validate import ARI_PASS, adjusted_rand_index, class_probabilities, match_classes, relabel_to_match

FIT_YEARS = range(2016, 2022)  # step 5's model A
HELD_OUT_YEARS = range(2022, 2025)  # step 5's model B and the children both label
BANDS = ((6, 11), (12, 17))  # step 9
CHECKS = ("held-out years", "2025 forward test", "age 6-11 refit", "age 12-17 refit", "second method (k-means)")

# Where the real steps leave each check's ARI, relative to the results folder.
REAL_SOURCES = {
    "held-out years": ("replicate.json", ("ari",)),
    "2025 forward test": ("forward_2025.json", ("ari",)),
    "age 6-11 refit": ("age_bands.json", ("bands", "6_11", "ari")),
    "age 12-17 refit": ("age_bands.json", ("bands", "12_17", "ari")),
    "second method (k-means)": ("crosscheck_kmeans.json", ("ari",)),
}


def matched_ari(reference_rates: np.ndarray, reference_labels: np.ndarray, fit, mask=None) -> float:
    """ARI between a reference labeling and a fresh fit's labels, after matching the fit's classes to the reference profiles.

    ARI itself does not depend on class numbering; the matching is kept so
    the procedure is the one the real steps use.
    """
    perm = match_classes(reference_rates, fit.condition_rates)
    labels = relabel_to_match(fit.labels, perm)
    return adjusted_rand_index(reference_labels, labels)


def run_checks(
    X: np.ndarray, X_2025: np.ndarray, table: pd.DataFrame, k: int, backend, n_starts: int, kmeans_starts: int, kmeans_seeds: int, seed: int
) -> dict[str, tuple[float, int]]:
    """The four replication checks on one dataset. Returns {check: (ari, n children compared)}."""
    fit_lca, kmeans = backend["lca"], backend["kmeans"]
    out = {}

    # The analogue of step 3: the pooled K-class model these checks are about.
    pooled = fit_lca(X, k, n_starts=n_starts, seed=seed)

    # Step 5: model A on the early years labels the late children; model B is fit on them.
    early = table["year"].isin(list(FIT_YEARS)).to_numpy()
    late = table["year"].isin(list(HELD_OUT_YEARS)).to_numpy()
    model_a = fit_lca(X[early], k, n_starts=n_starts, seed=seed)
    model_b = fit_lca(X[late], k, n_starts=n_starts, seed=seed)
    labels_from_a = class_probabilities(X[late], model_a.class_sizes, model_a.condition_rates).argmax(axis=1)
    out["held-out years"] = (matched_ari(model_a.condition_rates, labels_from_a, model_b), int(late.sum()))

    # Step 8: the frozen pooled model labels the 2025 children; a fresh model is fit on them.
    frozen_labels = class_probabilities(X_2025, pooled.class_sizes, pooled.condition_rates).argmax(axis=1)
    refit = fit_lca(X_2025, k, n_starts=n_starts, seed=seed)
    out["2025 forward test"] = (matched_ari(pooled.condition_rates, frozen_labels, refit), int(X_2025.shape[0]))

    # Step 9: the pooled model's labels for each age band against a fit on the band alone.
    for lo, hi in BANDS:
        in_band = table["age"].between(lo, hi).to_numpy()
        pooled_labels = class_probabilities(X[in_band], pooled.class_sizes, pooled.condition_rates).argmax(axis=1)
        band_fit = fit_lca(X[in_band], k, n_starts=n_starts, seed=seed)
        out[f"age {lo}-{hi} refit"] = (matched_ari(pooled.condition_rates, pooled_labels, band_fit), int(in_band.sum()))

    # Step 4: k-means on the flags against the pooled classes; lowest inertia across seeds.
    runs = [kmeans(X, k, n_starts=kmeans_starts, seed=seed + i) for i in range(kmeans_seeds)]
    best = min(runs, key=lambda run: run.inertia)
    out["second method (k-means)"] = (adjusted_rand_index(pooled.labels, best.labels), int(X.shape[0]))
    return out


def real_values(results_dir: Path) -> dict[str, float | None]:
    """The real checks' ARIs from the step outputs in ``results_dir``; None where a step has not run."""
    values = {}
    for check, (file, keys) in REAL_SOURCES.items():
        path = results_dir / file
        value = None
        if path.exists():
            value = json.loads(path.read_text())
            for key in keys:
                value = value.get(key) if isinstance(value, dict) else None
        values[check] = value
    return values


def main() -> None:
    parser = make_parser("The replication checks on data simulated from a continuum and from the class model")
    parser.add_argument("--k", type=int, default=None, help="number of classes (default: step 13's choice)")
    parser.add_argument("--replicates", type=int, default=5, help="simulated datasets per kind")
    parser.add_argument("--n-starts", type=int, default=20)
    parser.add_argument("--kmeans-starts", type=int, default=200, help="k-means starts per seed, as in step 4")
    parser.add_argument("--kmeans-seeds", type=int, default=5, help="k-means seeds per dataset; the lowest-inertia solution is scored, as in step 4")
    args = parser.parse_args()

    backend = backend_or_exit(args.backend, "lca", "kmeans")
    k = resolve_k(args)
    table, flags = load_analysis(MAIN_YEARS)
    table_2025, _ = load_analysis([2025])
    X, X_2025 = to_flag_matrix(table, flags), to_flag_matrix(table_2025, flags)
    class_model = load_lca_fit(args.results, k, table, flags)
    log(f"{X.shape[0]} children 2016-2024 and {X_2025.shape[0]} in 2025; K={k}; {args.replicates} replicates per kind")

    log("fitting the two-dimensional latent trait model to the real data")
    trait = fit_latent_trait(X, n_dims=2, seed=args.seed)

    rows = []
    for r in range(args.replicates):
        simulated = {
            "continuum": (
                simulate_latent_trait(trait, X.shape[0], seed=args.seed + 500 + r),
                simulate_latent_trait(trait, X_2025.shape[0], seed=args.seed + 600 + r),
            ),
            f"{k} classes": (
                simulate_classes(class_model.class_sizes, class_model.condition_rates, X.shape[0], seed=args.seed + 700 + r),
                simulate_classes(class_model.class_sizes, class_model.condition_rates, X_2025.shape[0], seed=args.seed + 800 + r),
            ),
        }
        for name, (fake, fake_2025) in simulated.items():
            scores = run_checks(fake, fake_2025, table, k, backend, args.n_starts, args.kmeans_starts, args.kmeans_seeds, args.seed)
            for check, (ari, n) in scores.items():
                rows.append({"data": name, "replicate": r, "check": check, "ari": ari, "n": n})
            log(f"   {name} {r}: " + ", ".join(f"{check} {ari:.3f}" for check, (ari, _) in scores.items()))

    results = pd.DataFrame(rows)
    real = real_values(args.results)
    for check, value in real.items():
        if value is not None:
            rows.append({"data": "real", "replicate": 0, "check": check, "ari": value, "n": None})
    table_out = pd.DataFrame(rows)
    table_out.to_csv(args.results / "replication_calibration.csv", index=False)

    ranges = {}
    for (name, check), group in results.groupby(["data", "check"], sort=False):
        ranges.setdefault(name, {})[check] = {"min": float(group["ari"].min()), "median": float(group["ari"].median()), "max": float(group["ari"].max())}
    summary = results.pivot_table(index="check", columns="data", values="ari", aggfunc=["min", "max"], sort=False)
    summary.insert(0, "real", pd.Series(real))
    log("ARI by check, real against simulated ranges:\n" + summary.round(3).to_string())

    reading = []
    for check, value in real.items():
        if value is None or "continuum" not in ranges:
            continue
        lo, hi = ranges["continuum"][check]["min"], ranges["continuum"][check]["max"]
        v, lo3, hi3 = (round(x, 3) for x in (value, lo, hi))  # compare at the precision the tables report
        where = "inside" if lo3 <= v <= hi3 else ("below" if v < lo3 else "above")
        reading.append(f"{check}: real {value:.3f} is {where} the continuum's range {lo:.3f} to {hi:.3f}")
    for line in reading:
        log("   " + line)

    save_json(
        {
            "k": k,
            "real": real,
            "simulated_ranges": ranges,
            "reading": reading,
            "pass_line": ARI_PASS,
            "settings": {
                "replicates": args.replicates,
                "n_starts": args.n_starts,
                "kmeans_starts": args.kmeans_starts,
                "kmeans_seeds": args.kmeans_seeds,
                "seed": args.seed,
            },
            "backend": args.backend,
        },
        args.results / "replication_calibration.json",
    )
    log(f"wrote replication_calibration.csv and .json to {args.results}")


if __name__ == "__main__":
    main()
