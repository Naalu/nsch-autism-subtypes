"""Step 12 of 14. Classes or dimensions? Continuous models against the latent classes.

Why this step exists
--------------------
A latent class model describes children as belonging to one of K groups. A
latent trait model (see ``src/nsch_subtypes/latent_trait.py`` and
``docs/background.md``) describes each child by a position on D continuous
scales. Both can be fit to the same 0/1 table and both give a likelihood on
the same scale, so we can ask which description the data support better,
and whether a model fit to data that have no groups at all would still
"find" groups (Bauer and Curran 2004).

What happens here
-----------------
1. Fit latent trait models with D = 1 to 4 dimensions on the pooled
   2016 to 2024 children and put them in one table with the latent class
   models from step 3, by log-likelihood and BIC. Trait likelihoods are
   recomputed on a finer quadrature grid than the one used to fit, so grid
   error does not flatter them.
2. Report the two- and three-dimensional loadings: which conditions depend
   on which axis.
3. Local dependence: for each model, compare the observed and model-implied
   rate of every pair of conditions (bivariate residuals). Both families
   assume conditions are independent given the class or position; pairs
   that violate this push either family toward extra classes or dimensions.
4. Age and period: how strongly the two scores track age and survey year,
   and whether the two-dimensional loadings are the same when the model is
   fit separately to ages 6 to 11 and 12 to 17 (Tucker's congruence after
   aligning the axes; 0.95 and above is conventionally "the same factor").
5. Simulation in both directions, several replicates each: data drawn from
   the fitted two-dimensional continuum (no groups), and data drawn from the
   fitted K-class model (groups, no continuum). On each, fit both families
   and record how much BIC prefers the continuous model. The real data are
   placed against both reference distributions.
6. A picture of the children in the two-dimensional space colored by class.
   Note that posterior-mean positions are shrunk toward the center and take
   one value per distinct row, so the picture would look clumped whether or
   not groups exist; it shows how the classes sit in the space, not whether
   the space is continuous.

Outputs, under ``--results``::

    dimensions.json                 all of the above in one file
    model_comparison.csv            latent class K=1..8 and latent trait D=1..4
    loadings_2d.csv, loadings_3d.csv
    bivariate_residuals.csv         every pair, for the chosen K and for D=2
    factor_scores.npy               (n, 2) positions of the children
    figures/factor_space.png
    figures/simulation_bic_gap.png

Usage::

    python scripts/12_dimensions.py --results results_k4 --k 4 --backend reference
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.linalg import orthogonal_procrustes  # noqa: E402

from _common import MAIN_YEARS, RESULTS_DIR, backend_or_exit, load_analysis, load_lca_fit, log, make_parser, resolve_k, save_json  # noqa: E402
from nsch_subtypes.data import to_flag_matrix  # noqa: E402
from nsch_subtypes.latent_trait import (  # noqa: E402
    fit_latent_trait,
    latent_trait_log_likelihood,
    latent_trait_pair_probabilities,
    simulate_latent_trait,
)
from nsch_subtypes.plots import CLASS_COLORS, GRAY, INK, RUST, tidy  # noqa: E402
from nsch_subtypes.validate import adjusted_rand_index, bivariate_residuals, class_model_pair_probabilities  # noqa: E402

CHECK_GRID = {1: 41, 2: 31, 3: 19, 4: 11}
CHI_SQUARE_5PCT = 3.84


def fit_trait_checked(X: np.ndarray, n_dims: int, seed: int):
    """Fit a latent trait model and recompute its log-likelihood on the finer check grid."""
    fit = fit_latent_trait(X, n_dims=n_dims, seed=seed)
    checked = dataclasses.replace(fit, n_quadrature=CHECK_GRID.get(n_dims, 11))
    return dataclasses.replace(checked, log_likelihood=latent_trait_log_likelihood(checked, X))


def simulate_classes(class_sizes: np.ndarray, rates: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Draw ``n`` children from a latent class model: pick a class, then flip each condition's coin."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(len(class_sizes), size=n, p=class_sizes / class_sizes.sum())
    return (rng.random((n, rates.shape[1])) < rates[labels]).astype(np.uint8)


def bic_gap_curve(fit_lca, X: np.ndarray, k_max: int, n_starts: int, seed: int) -> dict:
    """Latent class BIC for K = 1..k_max minus the 2-D latent trait BIC, on one dataset (positive = trait preferred)."""
    trait_bic = fit_trait_checked(X, 2, seed).bic()
    class_bic = [fit_lca(X, k, n_starts=n_starts, seed=seed).bic() for k in range(1, k_max + 1)]
    return {
        "trait_2d_bic": trait_bic,
        "class_bic": class_bic,
        "gap": [b - trait_bic for b in class_bic],
        "class_bic_turns_up": bool(np.argmin(class_bic) < k_max - 1),
    }


def summarize_residuals(name: str, table: pd.DataFrame) -> dict:
    worst = table.head(3)
    return {
        "model": name,
        "pairs_above_3.84": int((table["chi_square"] > CHI_SQUARE_5PCT).sum()),
        "total_pairs": len(table),
        "sum_chi_square": float(table["chi_square"].sum()),
        "worst_pairs": [f"{r.item_a}/{r.item_b} ({r.chi_square:.0f})" for r in worst.itertuples()],
    }


def congruence(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Tucker's congruence coefficient for each column pair of two loading matrices."""
    return (a * b).sum(axis=0) / np.sqrt((a**2).sum(axis=0) * (b**2).sum(axis=0))


def main() -> None:
    parser = make_parser("Latent trait (continuous) models against the latent classes")
    parser.add_argument("--k", type=int, default=None, help="the latent class solution to compare with (default: step 13's choice)")
    parser.add_argument("--d-max", type=int, default=4)
    parser.add_argument("--sim-k-max", type=int, default=6, help="largest K fit to each simulated dataset")
    parser.add_argument("--sim-replicates", type=int, default=3, help="simulated datasets per direction")
    parser.add_argument("--n-starts", type=int, default=10, help="starts per K on simulated data")
    parser.add_argument("--fits", type=Path, default=RESULTS_DIR, help="folder with step 3's lca_bic.csv and lca_k8.npz (default: results/)")
    args = parser.parse_args()

    fit_lca = backend_or_exit(args.backend, "lca")["lca"]
    table, flags = load_analysis(MAIN_YEARS)
    X = to_flag_matrix(table, flags)
    n = X.shape[0]
    k = resolve_k(args)
    lca = load_lca_fit(args.results, k, table, flags)
    labels = lca.labels
    (args.results / "figures").mkdir(parents=True, exist_ok=True)
    out: dict = {"k": k, "backend": args.backend, "n": n}

    # ---- 1. one table of models -----------------------------------------------------------
    traits = {}
    rows = []
    for d in range(1, args.d_max + 1):
        traits[d] = fit_trait_checked(X, d, args.seed)
        t = traits[d]
        rows.append(
            {"family": "latent trait", "size": d, "n_parameters": t.n_parameters, "log_likelihood": t.log_likelihood, "bic": t.bic(n), "converged": t.converged}
        )
        log(f"latent trait D={d}: log-likelihood {t.log_likelihood:.1f}, BIC {t.bic(n):.1f}, converged={t.converged}")
    bic_path = args.fits / "lca_bic.csv"
    if bic_path.exists():
        for _, r in pd.read_csv(bic_path).iterrows():
            rows.append(
                {
                    "family": "latent class",
                    "size": int(r["k"]),
                    "n_parameters": int(r["n_parameters"]),
                    "log_likelihood": r["log_likelihood"],
                    "bic": r["bic"],
                    "converged": bool(r["converged"]),
                }
            )
    comparison = pd.DataFrame(rows).sort_values("bic").reset_index(drop=True)
    comparison.to_csv(args.results / "model_comparison.csv", index=False)
    out["model_comparison"] = comparison.to_dict(orient="records")
    log(comparison.round(1).to_string(index=False))

    # ---- 2. loadings ----------------------------------------------------------------------
    for d in (2, 3):
        if d not in traits:
            continue
        loadings = pd.DataFrame(traits[d].slopes, columns=[f"dim{i + 1}_slope" for i in range(d)])
        loadings.insert(0, "condition", flags)
        loadings["intercept"] = traits[d].intercepts
        loadings["prevalence"] = X.mean(axis=0)
        loadings.to_csv(args.results / f"loadings_{d}d.csv", index=False)
        log(f"{d}-dimensional loadings:\n{loadings.round(2).to_string(index=False)}")

    # ---- 3. local dependence -----------------------------------------------------------------
    residual_tables = {f"latent class K={k}": bivariate_residuals(X, *class_model_pair_probabilities(lca.class_sizes, lca.condition_rates), flags)}
    k8 = args.fits / "lca_k8.npz"
    if k8.exists() and k != 8:
        fit8 = load_lca_fit(args.fits, 8, table, flags)
        residual_tables["latent class K=8"] = bivariate_residuals(X, *class_model_pair_probabilities(fit8.class_sizes, fit8.condition_rates), flags)
    for d in (2, 3):
        if d in traits:
            residual_tables[f"latent trait D={d}"] = bivariate_residuals(X, *latent_trait_pair_probabilities(traits[d]), flags)
    out["bivariate_residuals"] = [summarize_residuals(name, t) for name, t in residual_tables.items()]
    detail = pd.concat([t.assign(model=name) for name, t in residual_tables.items()], ignore_index=True)
    detail.to_csv(args.results / "bivariate_residuals.csv", index=False)
    for s in out["bivariate_residuals"]:
        log(f"{s['model']}: {s['pairs_above_3.84']} of {s['total_pairs']} pairs above 3.84; worst {s['worst_pairs']}")

    # ---- 4. age and period ----------------------------------------------------------------------
    scores = traits[2].scores
    np.save(args.results / "factor_scores.npy", scores)
    age_period = {
        "corr_age": [float(np.corrcoef(scores[:, i], table["age"])[0, 1]) for i in range(2)],
        "corr_year": [float(np.corrcoef(scores[:, i], table["year"])[0, 1]) for i in range(2)],
        "bands": {},
    }
    for name, (lo, hi) in {"6_11": (6, 11), "12_17": (12, 17)}.items():
        band = table["age"].between(lo, hi).to_numpy()
        band_fit = fit_trait_checked(X[band], 2, args.seed)
        rotation, _ = orthogonal_procrustes(band_fit.slopes, traits[2].slopes)
        aligned = band_fit.slopes @ rotation
        age_period["bands"][name] = {"n": int(band.sum()), "congruence_per_dimension": congruence(aligned, traits[2].slopes)}
    out["age_and_period"] = age_period
    log(f"score correlations with age {np.round(age_period['corr_age'], 3).tolist()}, with year {np.round(age_period['corr_year'], 3).tolist()}")
    log("loading congruence by age band: " + "; ".join(f"{b}: {np.round(v['congruence_per_dimension'], 3).tolist()}" for b, v in age_period["bands"].items()))

    # Quadrant rule: two cut points on the scores, placed at the median of the
    # class means. The cut points use the class labels, so this measures how
    # well the classes line up with the axes, not an independent clustering.
    class_means = np.vstack([scores[labels == c].mean(axis=0) for c in range(k)])
    cut_points = np.median(class_means, axis=0)
    quadrant = (scores[:, 0] > cut_points[0]).astype(int) * 2 + (scores[:, 1] > cut_points[1]).astype(int)
    out["quadrants"] = {
        "cut_points": cut_points,
        "ari_class_vs_quadrant": adjusted_rand_index(labels, quadrant),
        "note": "cut points placed using the class means",
    }
    log(f"ARI, classes vs two-cut quadrant rule: {out['quadrants']['ari_class_vs_quadrant']:.3f}")

    # ---- 5. simulation in both directions -----------------------------------------------------
    log(f"simulation: {args.sim_replicates} replicates per direction, K up to {args.sim_k_max}")
    real = bic_gap_curve(fit_lca, X, args.sim_k_max, args.n_starts, args.seed)
    from_continuum, from_classes = [], []
    for r in range(args.sim_replicates):
        from_continuum.append(bic_gap_curve(fit_lca, simulate_latent_trait(traits[2], n, seed=100 + r), args.sim_k_max, args.n_starts, args.seed))
        from_classes.append(
            bic_gap_curve(fit_lca, simulate_classes(lca.class_sizes, lca.condition_rates, n, seed=200 + r), args.sim_k_max, args.n_starts, args.seed)
        )
        log(f"   replicate {r + 1}: continuum gap at K={k} {from_continuum[-1]['gap'][k - 1]:.0f}; classes gap at K={k} {from_classes[-1]['gap'][k - 1]:.0f}")
    out["simulation"] = {
        "note": "gap = latent class BIC minus 2-D latent trait BIC on the same dataset; positive means the continuous model is preferred",
        "real": real,
        "from_continuum": from_continuum,
        "from_classes": from_classes,
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), sharey=True)
    ks = np.arange(1, args.sim_k_max + 1)
    for ax, sims, title in (
        (axes[0], from_continuum, "simulated from the 2-D continuum (no groups)"),
        (axes[1], from_classes, f"simulated from the K={k} class model (no continuum)"),
    ):
        for i, s in enumerate(sims):
            ax.plot(ks, s["gap"], "--", color=GRAY, lw=1, label="simulated datasets" if i == 0 else None)
        ax.plot(ks, real["gap"], "o-", color=RUST, lw=1.8, label="real data")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xlabel("number of classes K")
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xticks(ks)
        tidy(ax)
    axes[0].set_ylabel("class BIC minus 2-D trait BIC\n(above 0: continuous model preferred)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.results / "figures" / "simulation_bic_gap.png", dpi=160)
    plt.close(fig)

    # ---- 6. the children in the plane ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    jitter = np.random.default_rng(0).normal(0, 0.04, scores.shape)
    for c in range(k):
        m = labels == c
        ax.scatter(
            scores[m, 0] + jitter[m, 0],
            scores[m, 1] + jitter[m, 1],
            s=4,
            alpha=0.35,
            color=CLASS_COLORS[c % len(CLASS_COLORS)],
            label=f"class {c} (n={m.sum()})",
            rasterized=True,
        )
    load2 = pd.read_csv(args.results / "loadings_2d.csv")
    top1 = load2.sort_values("dim1_slope", ascending=False)["condition"].head(3).tolist()
    top2 = load2.sort_values("dim2_slope", ascending=False)["condition"].head(3).tolist()
    ax.set_xlabel(f"dimension 1 score  (highest slopes: {', '.join(top1)})")
    ax.set_ylabel(f"dimension 2 score  (highest slopes: {', '.join(top2)})")
    ax.set_title(f"Posterior-mean positions in the 2-D trait model, colored by K={k} class", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7, markerscale=3)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(args.results / "figures" / "factor_space.png", dpi=160)
    plt.close(fig)

    out["flags"] = flags
    save_json(out, args.results / "dimensions.json")


if __name__ == "__main__":
    main()
