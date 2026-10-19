"""Check 12. Types or dimensions? A continuous model against the latent classes.

Why this check exists
---------------------
The K=4 classes look like the four corners of two dimensions (developmental-
language and psychiatric), and no distance-based method finds cluster
geometry in the data. Bauer and Curran (2004) showed that a latent class
model will produce tidy-looking classes from data that are continuous. So
the right question is not "how many classes" but "classes or dimensions?",
and the way to ask it is to fit a continuous model to the same data and
compare.

What happens here
-----------------
1. Fit one- and two-dimensional latent trait models (binary factor
   analysis) to the pooled children. Report their log-likelihoods and BIC
   next to the latent class models for K = 1 to the largest K fitted.
2. Show the two-dimensional loadings: which conditions load on which axis.
3. Place every child in the two-dimensional space and color by K-class;
   report how well the class labels can be read off the two scores (a
   simple quadrant rule), and the share of each class in each quadrant.
4. Bauer-Curran simulation: generate a dataset of the same size from the
   fitted two-dimensional model (so by construction it has no classes), run
   the latent class BIC curve on it, and overlay on the real curve. If a
   pure continuum reproduces the real curve and the real entropy, the data
   cannot tell classes from dimensions.

Outputs, under ``--results``::

    dimensions.json           fit statistics for all models, quadrant table, simulation summary
    loadings_2d.csv           slopes and intercepts per condition
    factor_scores.npy         (n, 2) positions of the children
    figures/factor_space.png  children in the two-dimensional space, colored by class
    figures/bic_real_vs_simulated.png

Usage::

    python scripts/12_dimensions.py --results results_k4 --k 4 --backend reference
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import MAIN_YEARS, backend_or_exit, load_analysis, load_lca_fit, log, make_parser, read_best_k, save_json  # noqa: E402
from nsch_subtypes.data import to_flag_matrix  # noqa: E402
from nsch_subtypes.latent_trait import fit_latent_trait, simulate_latent_trait  # noqa: E402
from nsch_subtypes.plots import GRAY, INK, RUST  # noqa: E402
from nsch_subtypes.validate import adjusted_rand_index  # noqa: E402

CLASS_COLORS = ["#b8582a", "#0e7490", "#6b46c1", "#2f855a", "#b7791f", "#9b2c2c", "#2b6cb0", "#4a5568"]


def main() -> None:
    parser = make_parser("Latent trait (continuous) models against the latent classes")
    parser.add_argument("--k", type=int, default=None, help="the latent class solution to compare with")
    parser.add_argument("--k-max", type=int, default=6, help="largest K for the simulated BIC curve")
    parser.add_argument("--n-starts", type=int, default=10, help="starts per K on the simulated data")
    args = parser.parse_args()

    fit_lca = backend_or_exit(args.backend, "lca")["lca"]
    k = args.k or read_best_k(args.results)
    table, flags = load_analysis(MAIN_YEARS)
    X = to_flag_matrix(table, flags)
    n = X.shape[0]
    lca = load_lca_fit(args.results, k)
    labels = lca["responsibilities"].argmax(axis=1)
    (args.results / "figures").mkdir(exist_ok=True)
    out: dict = {"k": k, "backend": args.backend, "n": int(n)}

    # ---- 1. continuous models ----------------------------------------------------------
    trait_1d = fit_latent_trait(X, n_dims=1, seed=args.seed)
    trait_2d = fit_latent_trait(X, n_dims=2, seed=args.seed)
    bic_table = pd.read_csv(args.results / "lca_bic.csv") if (args.results / "lca_bic.csv").exists() else None
    rows = [
        {"model": "latent trait, 1 dimension", "n_parameters": trait_1d.n_parameters, "log_likelihood": trait_1d.log_likelihood, "bic": trait_1d.bic(n)},
        {"model": "latent trait, 2 dimensions", "n_parameters": trait_2d.n_parameters, "log_likelihood": trait_2d.log_likelihood, "bic": trait_2d.bic(n)},
    ]
    if bic_table is not None:
        for _, r in bic_table.iterrows():
            rows.append(
                {"model": f"latent class, K={int(r['k'])}", "n_parameters": int(r["n_parameters"]), "log_likelihood": r["log_likelihood"], "bic": r["bic"]}
            )
    comparison = pd.DataFrame(rows).sort_values("bic").reset_index(drop=True)
    out["model_comparison"] = comparison.to_dict(orient="records")
    log("models on the same data, sorted by BIC (lower is better):")
    log(comparison.round(1).to_string(index=False))

    # ---- 2. loadings ------------------------------------------------------------------------
    loadings = pd.DataFrame(trait_2d.slopes, columns=["dim1_slope", "dim2_slope"])
    loadings.insert(0, "condition", flags)
    loadings["intercept"] = trait_2d.intercepts
    loadings["prevalence"] = X.mean(axis=0)
    loadings.to_csv(args.results / "loadings_2d.csv", index=False)
    log("two-dimensional loadings (slopes):")
    log(loadings.round(2).to_string(index=False))

    # ---- 3. children in the plane, by class ------------------------------------------------------
    scores = trait_2d.scores
    np.save(args.results / "factor_scores.npy", scores)
    # Quadrant rule: split each dimension at the score that best separates high from low
    # by taking the median of class means (so it is not tuned child by child).
    class_means = np.vstack([scores[labels == c].mean(axis=0) for c in range(k)])
    cut = np.median(class_means, axis=0)
    quadrant = (scores[:, 0] > cut[0]).astype(int) * 2 + (scores[:, 1] > cut[1]).astype(int)
    quadrant_table = pd.crosstab(pd.Series(labels, name="class"), pd.Series(quadrant, name="quadrant"))
    out["quadrants"] = {
        "cut_points": cut.tolist(),
        "ari_class_vs_quadrant": adjusted_rand_index(labels, quadrant),
        "class_by_quadrant": quadrant_table.to_dict(),
        "class_mean_scores": class_means.tolist(),
    }
    log(f"ARI, latent classes vs a two-cut quadrant rule on the factor scores: {out['quadrants']['ari_class_vs_quadrant']:.3f}")
    log(quadrant_table.to_string())

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    jitter = np.random.default_rng(0).normal(0, 0.04, scores.shape)  # many children share identical rows
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
    ax.axvline(cut[0], color=GRAY, ls=":", lw=1)
    ax.axhline(cut[1], color=GRAY, ls=":", lw=1)
    top1 = loadings.sort_values("dim1_slope", ascending=False)["condition"].head(3).tolist()
    top2 = loadings.sort_values("dim2_slope", ascending=False)["condition"].head(3).tolist()
    ax.set_xlabel(f"dimension 1 score  ({', '.join(top1)})")
    ax.set_ylabel(f"dimension 2 score  ({', '.join(top2)})")
    ax.set_title(f"Children in the two-dimensional latent trait space, colored by K={k} latent class", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7, markerscale=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.results / "figures" / "factor_space.png", dpi=160)
    plt.close(fig)

    # ---- 4. Bauer-Curran simulation ----------------------------------------------------------------------
    X_sim = simulate_latent_trait(trait_2d, n, seed=args.seed + 1)
    ks = list(range(1, args.k_max + 1))
    sim_rows = []
    for kk in ks:
        fit = fit_lca(X_sim, kk, n_starts=args.n_starts, seed=args.seed)
        sim_rows.append(
            {"k": kk, "bic": fit.bic(), "entropy": fit.entropy(), "smallest_class_share": float(fit.class_sizes.min()), "log_likelihood": fit.log_likelihood}
        )
        if kk == k:
            sim_profiles = pd.DataFrame(fit.condition_rates, columns=flags)
            sim_profiles.insert(0, "class_size", fit.class_sizes)
            sim_profiles.insert(0, "class", range(kk))
            sim_profiles.to_csv(args.results / f"profiles_simulated_continuum_k{k}.csv", index=False)
    simulated = pd.DataFrame(sim_rows)
    out["simulation"] = {
        "note": "data drawn from the fitted 2-dimensional latent trait model, no classes by construction",
        "lca_on_simulated": simulated.to_dict(orient="records"),
    }
    log("latent class fits on data simulated from the two-dimensional continuum:")
    log(simulated.round(3).to_string(index=False))

    if bic_table is not None:
        real = bic_table[bic_table["k"].isin(ks)]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        axes[0].plot(real["k"], real["bic"] - real["bic"].iloc[0], "o-", color=RUST, label="real data")
        axes[0].plot(simulated["k"], simulated["bic"] - simulated["bic"].iloc[0], "s--", color=INK, label="simulated continuum")
        axes[0].set_xlabel("number of classes K")
        axes[0].set_ylabel("BIC relative to K=1")
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].plot(real["k"], real["entropy"], "o-", color=RUST, label="real data")
        axes[1].plot(simulated["k"], simulated["entropy"], "s--", color=INK, label="simulated continuum")
        axes[1].set_xlabel("number of classes K")
        axes[1].set_ylabel("relative entropy")
        axes[1].set_ylim(0.5, 1.0)
        for ax in axes:
            ax.set_xticks(ks)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.suptitle("Does a continuum with no classes look like the real data to a latent class model?", fontsize=9, x=0.02, ha="left")
        fig.tight_layout()
        fig.savefig(args.results / "figures" / "bic_real_vs_simulated.png", dpi=160)
        plt.close(fig)

    out["flags"] = flags
    save_json(out, args.results / "dimensions.json")


if __name__ == "__main__":
    main()
