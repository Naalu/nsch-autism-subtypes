"""Step 14 of 14. Do children clump in the two-dimensional space, and what does stability mean here?

Why this step exists
--------------------
Two of our arguments need a yardstick.

Stability (step 6) chose K = 4 as the largest K whose classes all come back
under resampling. But a smooth continuum with no groups can also be cut into
pieces that come back every time: if children are spread evenly over a
disc, "cut the disc into four quarters" gives the same quarters in every
resample. So we need to know how stable classes are when the data have no
classes, and how stable they are when the classes are real.

The latent trait model assumes children's positions follow a smooth bell
curve. If children really fell into groups, their positions would pile up in
a few places. We estimate the distribution of positions from the data
instead of assuming it (``fit_latent_density``), but that estimate is lumpy
even when the truth is smooth, so it too needs a yardstick.

The yardstick in both cases is simulation. We make fake datasets of the
same size from two models fitted to our data: the two-dimensional latent
trait model, which has no groups, and a latent class model, which is
nothing but groups. Each fake dataset goes through exactly the same
analysis as the real one.

What happens here
-----------------
1. Fit the two-dimensional latent trait model to the 2016 to 2024 children.
2. Shape of the cloud: estimate the distribution of positions for the real
   data and for ``--replicates`` datasets simulated from the latent trait
   model ("continuum") and from the latent class model at the chosen K
   ("classes"), refitting the trait model to each. The summary is how much
   better the estimated distribution explains the data than the bell curve
   (log-likelihood units): small for a continuum, large for groups.
3. Stability under known truth: one dataset from the latent trait model,
   bootstrapped at K = 2 to ``--k-max``, and one from the ``--control-k``
   class model, bootstrapped at K = control - 1, control and control + 1.
   These sit beside the real stability curve (step 13's
   ``stability_by_k.csv``) in the results figures.
4. Positions for plotting: one random draw of each real child's position
   from its posterior ("plausible value"), saved with the child's
   identifiers so the figures can color children by class.

Bootstraps here use fewer resamples and starts than step 6 (``--n-boot``,
``--n-starts``) to keep the run to about an hour; the comparison is between
curves, not individual values.

Outputs, under ``--results``::

    latent_density.npz            grid nodes and estimated weights for every dataset
    latent_density.csv            improvement over the bell curve, per dataset
    stability_calibration.csv     least and mean class Jaccard at each K, simulated data
    positions_2d.npz              one posterior draw per child, with year and hhid
    shape_and_stability.json      summary

Usage::

    python scripts/14_shape_and_stability.py --backend reference
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import MAIN_YEARS, backend_or_exit, load_analysis, load_lca_fit, log, make_parser, resolve_k, save_json
from nsch_subtypes.data import to_flag_matrix
from nsch_subtypes.latent_trait import draw_positions, fit_latent_density, fit_latent_trait, simulate_latent_trait
from nsch_subtypes.validate import JACCARD_STABLE, bootstrap_stability


def simulate_classes(class_sizes: np.ndarray, condition_rates: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Fake children from a latent class model: draw a class, then each condition independently at that class's rate."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(len(class_sizes), size=n, p=class_sizes / class_sizes.sum())
    return (rng.random((n, condition_rates.shape[1])) < condition_rates[labels]).astype(np.uint8)


def density_row(name: str, replicate: int, X: np.ndarray, trait, seed: int) -> tuple[dict, np.ndarray]:
    """Estimate the distribution of positions for one dataset, refitting the trait model unless one is given."""
    fit = trait if trait is not None else fit_latent_trait(X, n_dims=2, seed=seed)
    density = fit_latent_density(fit, X)
    row = {"data": name, "replicate": replicate, "improvement": density.improvement, "n_iter": density.n_iter, "converged": density.converged}
    log(f"   {name} {replicate}: estimated distribution beats the bell curve by {density.improvement:.1f} log-likelihood units")
    return row, density


def stability_rows(name: str, X: np.ndarray, ks, fit_lca, n_boot: int, n_starts: int, seed: int) -> list[dict]:
    """Bootstrap class stability at each K for one dataset."""
    rows = []
    for k in ks:
        reference = fit_lca(X, k, n_starts=n_starts, seed=seed)

        def refit(resampled: np.ndarray, n_classes: int) -> np.ndarray:
            return fit_lca(resampled, n_classes, n_starts=n_starts, seed=seed).labels

        result = bootstrap_stability(X, reference.labels, refit, k, n_boot=n_boot, seed=seed + 1)
        sizes = np.bincount(reference.labels, minlength=k) / len(reference.labels)
        rows.append(
            {
                "data": name,
                "k": k,
                "min_jaccard": float(result.mean.min()),
                "mean_jaccard": float(result.mean.mean()),
                "all_stable": bool((result.mean >= JACCARD_STABLE).all()),
                "smallest_class_share": float(sizes.min()),
                "jaccard_by_class": np.round(result.mean, 3).tolist(),
            }
        )
        log(f"   {name} K={k}: least stable class {result.mean.min():.3f}; class shares {np.round(sizes, 2).tolist()}")
    return rows


def main() -> None:
    parser = make_parser("Shape of the latent distribution and stability under known truth, by simulation")
    parser.add_argument("--k", type=int, default=None, help="class model used to simulate 'classes' data (default: step 13's choice)")
    parser.add_argument("--replicates", type=int, default=5, help="simulated datasets per kind for the shape check")
    parser.add_argument("--k-max", type=int, default=7, help="largest K bootstrapped on the simulated continuum")
    parser.add_argument("--control-k", type=int, default=5, help="class model used for the stability control")
    parser.add_argument("--n-boot", type=int, default=50)
    parser.add_argument("--n-starts", type=int, default=10)
    parser.add_argument("--skip-stability", action="store_true", help="only the shape check and the positions")
    args = parser.parse_args()

    fit_lca = backend_or_exit(args.backend, "lca")["lca"]
    table, flags = load_analysis(MAIN_YEARS)
    X = to_flag_matrix(table, flags)
    n = X.shape[0]
    k = resolve_k(args)
    class_model = load_lca_fit(args.results, k, table, flags)

    log("fitting the two-dimensional latent trait model")
    trait = fit_latent_trait(X, n_dims=2, seed=args.seed)

    # ---- 1. shape of the cloud ------------------------------------------------------------------
    log(f"shape: real data, then {args.replicates} simulated continua and {args.replicates} simulated {k}-class datasets")
    rows, weights = [], {}
    row, density = density_row("real", 0, X, trait, args.seed)
    rows.append(row)
    weights["real"] = density.weights
    nodes = density.nodes
    for r in range(args.replicates):
        fake = simulate_latent_trait(trait, n, seed=args.seed + 100 + r)
        row, density = density_row("continuum", r, fake, None, args.seed)
        rows.append(row)
        weights[f"continuum_{r}"] = density.weights
        fake = simulate_classes(class_model.class_sizes, class_model.condition_rates, n, seed=args.seed + 200 + r)
        row, density = density_row("classes", r, fake, None, args.seed)
        rows.append(row)
        weights[f"classes_{r}"] = density.weights
    shape = pd.DataFrame(rows)
    shape.to_csv(args.results / "latent_density.csv", index=False)
    np.savez_compressed(args.results / "latent_density.npz", nodes=nodes, class_model_k=k, **weights)
    summary = shape.groupby("data")["improvement"].agg(["min", "median", "max"])
    log("improvement over the bell curve (log-likelihood units):\n" + summary.round(1).to_string())

    # ---- 2. stability under known truth ------------------------------------------------------------
    stability = []
    if not args.skip_stability:
        log(f"stability on a simulated continuum, K = 2 to {args.k_max} ({args.n_boot} resamples, {args.n_starts} starts)")
        fake = simulate_latent_trait(trait, n, seed=args.seed + 300)
        stability += stability_rows("continuum", fake, range(2, args.k_max + 1), fit_lca, args.n_boot, args.n_starts, args.seed)
        control = load_lca_fit(args.results, args.control_k, table, flags)
        log(f"stability on data simulated from the {args.control_k}-class model")
        fake = simulate_classes(control.class_sizes, control.condition_rates, n, seed=args.seed + 400)
        ks = range(max(2, args.control_k - 1), args.control_k + 2)
        stability += stability_rows(f"{args.control_k} classes", fake, ks, fit_lca, args.n_boot, args.n_starts, args.seed)
        pd.DataFrame(stability).to_csv(args.results / "stability_calibration.csv", index=False)

    # ---- 3. positions for plotting -------------------------------------------------------------------
    draws = draw_positions(trait, X, n_draws=1, seed=args.seed)[:, 0, :]
    np.savez_compressed(
        args.results / "positions_2d.npz",
        positions=draws,
        slopes=trait.slopes,
        intercepts=trait.intercepts,
        flags=np.array(flags),
        year=table["year"].to_numpy(),
        hhid=table["hhid"].to_numpy(),
    )

    save_json(
        {
            "class_model_k": k,
            "replicates": args.replicates,
            "improvement_over_bell_curve": {name: summary.loc[name].to_dict() for name in summary.index},
            "stability_calibration": stability,
            "settings": {"n_boot": args.n_boot, "n_starts": args.n_starts, "control_k": args.control_k},
            "backend": args.backend,
        },
        args.results / "shape_and_stability.json",
    )
    log(f"wrote latent_density.*, stability_calibration.csv, positions_2d.npz and shape_and_stability.json to {args.results}")


if __name__ == "__main__":
    main()
