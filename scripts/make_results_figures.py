"""The results figures for the checkpoint and the paper.

Reads the main run (``results/``, where BIC chose K) and the K-specific runs
(``results_k4/``, ``results_k5/``, ...) and writes to ``reports/figures/results/``:

    classes_2x2.png          the K=4 class profiles arranged as a developmental x psychiatric grid
    severity_by_class.png    weighted severity distribution per class, K=4
    loadings.png             two-dimensional latent trait loadings per condition
    shares_pooled_vs_2025.png  class shares 2016-24 and 2025 with 95% intervals
    checks_by_k.png          every pass/fail check against K
    bic_curves.png           real data, shuffled columns, simulated continuum
    method_comparison.csv    alternative clustering methods against the K=4 classes

Also names the K=4 classes from their profiles (see ``name_classes``) and
writes ``reports/tables/class_names_k4.csv`` so the paper uses one naming.

Usage::

    python scripts/make_results_figures.py --k 4
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import MAIN_YEARS, REPO, load_analysis, load_lca_fit, log  # noqa: E402
from nsch_subtypes.data import to_flag_matrix  # noqa: E402
from nsch_subtypes.validate import adjusted_rand_index  # noqa: E402

OUT = REPO / "reports" / "figures" / "results"
TABLES = REPO / "reports" / "tables"
RUST, CYAN, PURPLE, GREEN, INK, GRAY = "#b8582a", "#0e7490", "#6b46c1", "#2f855a", "#1f2330", "#7a8094"
DEVELOPMENTAL = ["dev_delay", "learning", "speech", "intellectual"]
PSYCHIATRIC = ["anxiety", "adhd", "depression", "behavior"]


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def name_classes(rates: pd.DataFrame) -> pd.DataFrame:
    """Name each class by where it sits on the two axes (mean rate of the four marker conditions on each).

    Returns one row per class with ``dev``, ``psych`` scores and a ``name``.
    With K=4 this gives the two-by-two; with other K the names still describe
    the corners and any extra class is named by its most distinctive items.
    """
    dev = rates[DEVELOPMENTAL].mean(axis=1)
    psych = rates[PSYCHIATRIC].mean(axis=1)
    hi_dev = dev > dev.median()
    hi_psych = psych > psych.median()
    names = []
    for i in rates.index:
        if hi_dev[i] and hi_psych[i]:
            names.append("developmental and psychiatric")
        elif hi_dev[i]:
            names.append("developmental-language")
        elif hi_psych[i]:
            names.append("psychiatric")
        else:
            names.append("few other conditions")
    out = pd.DataFrame({"class": rates.index, "dev_score": dev.round(3), "psych_score": psych.round(3), "name": names})
    # If two classes got the same name, disambiguate by their most distinctive items.
    for name, group in out.groupby("name"):
        if len(group) > 1:
            for i in group.index:
                top = rates.loc[i].sort_values(ascending=False).index[:2]
                out.loc[i, "name"] = f"{name} ({', '.join(top)})"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Results figures")
    parser.add_argument("--k", type=int, default=4, help="the K whose classes are the main result")
    parser.add_argument("--skip-methods", action="store_true", help="skip the slow alternative-methods table")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    k = args.k
    kdir = REPO / f"results_k{k}"
    table, flags = load_analysis(MAIN_YEARS)
    X = to_flag_matrix(table, flags)
    fit = load_lca_fit(kdir, k)
    labels = fit["responsibilities"].argmax(axis=1)
    rates = pd.DataFrame(fit["condition_rates"], columns=flags)
    names = name_classes(rates)
    names["n"] = np.bincount(labels, minlength=k)
    names["share_unweighted"] = (names["n"] / len(labels)).round(3)
    names.to_csv(TABLES / f"class_names_k{k}.csv", index=False)
    label_of = dict(zip(names["class"], names["name"], strict=True))
    log("class names:\n" + names.to_string(index=False))

    # ---- 1. the two-by-two ---------------------------------------------------------------------
    order = [c for c in sorted(rates.index, key=lambda c: (names.loc[c, "dev_score"] > names["dev_score"].median(), names.loc[c, "psych_score"]))]
    shares = pd.read_csv(kdir / "shares.csv").set_index("class")
    fig, ax = plt.subplots(figsize=(0.5 * len(flags) + 2.5, 0.55 * k + 1.8))
    data = rates.loc[order].to_numpy()
    im = ax.imshow(data, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(flags)))
    ax.set_xticklabels(flags, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"{label_of[c]}\n{shares.loc[c, 'share']:.0%} of children" for c in order], fontsize=8)
    for i in range(k):
        for j in range(len(flags)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if data[i, j] > 0.55 else INK)
    # Bracket the two axes' marker conditions.
    for group, color in ((DEVELOPMENTAL, CYAN), (PSYCHIATRIC, PURPLE)):
        xs = [flags.index(f) for f in group]
        ax.plot([min(xs) - 0.45, max(xs) + 0.45], [-0.75, -0.75], color=color, lw=3, clip_on=False)
    ax.set_title(
        f"Condition rates by class, K={k}. Rows are the four corners of two axes: developmental-language (blue bracket) and psychiatric (purple).",
        fontsize=8.5,
        loc="left",
        pad=14,
    )
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="share of class with the condition")
    fig.tight_layout()
    fig.savefig(OUT / "classes_2x2.png", dpi=170)
    plt.close(fig)

    # ---- 2. severity by class ---------------------------------------------------------------------
    sev = pd.read_csv(kdir / "severity_by_class.csv").set_index("class").loc[order]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    left = np.zeros(k)
    for level, color in (("mild", "#f6c89f"), ("moderate", RUST), ("severe", "#5a2a0e")):
        vals = sev[level].to_numpy()
        ax.barh(range(k), vals, left=left, color=color, label=level)
        for i, (v, start) in enumerate(zip(vals, left, strict=True)):
            if v > 0.06:
                ax.text(start + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white" if level != "mild" else INK)
        left += vals
    ax.set_yticks(range(k))
    ax.set_yticklabels([label_of[c] for c in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("weighted share of class, parent-rated autism severity")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0))
    ax.set_title("Severity follows the developmental-language axis, not the psychiatric one", fontsize=9, loc="left", pad=18)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "severity_by_class.png", dpi=170)
    plt.close(fig)

    # ---- 3. loadings ------------------------------------------------------------------------------------
    if (kdir / "loadings_2d.csv").exists():
        load = pd.read_csv(kdir / "loadings_2d.csv").sort_values("dim1_slope")
        fig, ax = plt.subplots(figsize=(6.5, 5))
        y = np.arange(len(load))
        ax.barh(y - 0.2, load["dim1_slope"], height=0.4, color=CYAN, label="dimension 1 (developmental-language)")
        ax.barh(y + 0.2, load["dim2_slope"], height=0.4, color=PURPLE, label="dimension 2 (psychiatric)")
        ax.set_yticks(y)
        ax.set_yticklabels(load["condition"], fontsize=8)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_xlabel("slope in the two-dimensional latent trait model (larger = condition depends more on that axis)")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.set_title("Each condition loads on one axis or the other, not both", fontsize=9, loc="left")
        tidy(ax)
        fig.tight_layout()
        fig.savefig(OUT / "loadings.png", dpi=170)
        plt.close(fig)

    # ---- 4. shares pooled vs 2025 ------------------------------------------------------------------------
    s25 = pd.read_csv(kdir / "shares_2025.csv").set_index("class") if (kdir / "shares_2025.csv").exists() else None
    if s25 is not None:
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        xs = np.arange(k)
        for off, df, color, lab in ((-0.17, shares, RUST, "2016-2024 pooled"), (0.17, s25, CYAN, "2025 (frozen model)")):
            d = df.loc[order]
            ax.bar(xs + off, d["share"], width=0.34, color=color, label=lab)
            ax.errorbar(xs + off, d["share"], yerr=[d["share"] - d["ci_low"], d["ci_high"] - d["share"]], fmt="none", ecolor=INK, capsize=3, lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([label_of[c].replace(" and ", "\nand ") for c in order], fontsize=8)
        ax.set_ylabel("weighted share of US autistic children 6-17")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("Class shares in a survey year released after the design was fixed", fontsize=9, loc="left")
        tidy(ax)
        fig.tight_layout()
        fig.savefig(OUT / "shares_pooled_vs_2025.png", dpi=170)
        plt.close(fig)

    # ---- 5. every check against K --------------------------------------------------------------------------
    rows = []
    for d in ["results"] + sorted(glob.glob(str(REPO / "results_k*")), key=lambda s: int(re.sub(r"\D", "", Path(s).name))):
        d = Path(d)

        def load(name, folder=d):
            try:
                return json.load(open(folder / name))
            except FileNotFoundError:
                return {}

        km, rep, fwd, age, par = (
            load("crosscheck_kmeans.json"),
            load("replicate.json"),
            load("forward_2025.json"),
            load("age_bands.json"),
            load("parallel_ever.json"),
        )
        kk = km.get("k") or rep.get("k")
        if kk is None:
            continue
        try:
            stab = pd.read_csv(d / "stability.csv")["mean_jaccard"].min()
        except FileNotFoundError:
            stab = np.nan
        rows.append(
            {
                "K": kk,
                "second method (k-means)": km.get("ari", np.nan),
                "held-out years": rep.get("ari", np.nan),
                "2025 forward test": fwd.get("ari", np.nan),
                "age 6-11 refit": age.get("bands", {}).get("6_11", {}).get("ari_pooled_vs_band_fit", np.nan),
                "age 12-17 refit": age.get("bands", {}).get("12_17", {}).get("ari_pooled_vs_band_fit", np.nan),
                "'ever told' items": par.get("ari_current_vs_ever", np.nan),
                "least stable class (Jaccard)": stab,
            }
        )
    checks = pd.DataFrame(rows).drop_duplicates("K").sort_values("K").set_index("K")
    checks.round(3).to_csv(TABLES / "checks_by_k.csv")
    log("checks by K:\n" + checks.round(2).to_string())
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ari_cols = [c for c in checks.columns if "Jaccard" not in c]
    colors = [RUST, CYAN, PURPLE, GREEN, "#9b2c2c", "#b7791f"]
    for col, color in zip(ari_cols, colors, strict=False):
        ax.plot(checks.index, checks[col], "o-", color=color, label=col, lw=1.4)
    ax.plot(checks.index, checks["least stable class (Jaccard)"], "s--", color=GRAY, label="least stable class (Jaccard, line at 0.75)")
    ax.axhline(0.65, color=INK, lw=1, ls=":")
    ax.text(checks.index.max() + 0.05, 0.655, "ARI pass line 0.65", fontsize=7, va="bottom")
    ax.set_xlabel("number of classes K")
    ax.set_ylabel("adjusted Rand index / Jaccard")
    ax.set_xticks(checks.index)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
    ax.set_title("Every check passes at K=4 except the second method; none pass at K=8", fontsize=9, loc="left")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "checks_by_k.png", dpi=170)
    plt.close(fig)

    # ---- 6. BIC curves -----------------------------------------------------------------------------------------
    bic = pd.read_csv(REPO / "results" / "lca_bic.csv")
    dims = json.load(open(kdir / "dimensions.json")) if (kdir / "dimensions.json").exists() else None
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(bic["k"], bic["bic"], "o-", color=RUST, label="latent class, real data")
    if "bic_shuffled" in bic:
        ax.plot(bic["k"], bic["bic_shuffled"], "^:", color=GRAY, label="latent class, columns shuffled")
    if dims:
        sim = pd.DataFrame(dims["simulation"]["lca_on_simulated"])
        ax.plot(sim["k"], sim["bic"], "s--", color=INK, label="latent class, simulated 2-D continuum")
        trait = [m for m in dims["model_comparison"] if m["model"].startswith("latent trait, 2")][0]
        ax.axhline(trait["bic"], color=CYAN, lw=1.5)
        ax.text(
            1,
            trait["bic"] - 250,
            f"2-dimensional latent trait model, {trait['n_parameters']} parameters, BIC {trait['bic']:.0f}",
            color=CYAN,
            fontsize=7.5,
            ha="left",
            va="top",
        )
    ax.set_xlabel("number of classes K")
    ax.set_ylabel("BIC (lower is better)")
    ax.set_xticks(bic["k"])
    ax.set_ylim(bic["bic"].min() - 1800, bic["bic"].max() + 500)
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    ax.set_title("BIC never turns up for classes; a two-dimensional continuum beats every K", fontsize=9, loc="left")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "bic_curves.png", dpi=170)
    plt.close(fig)

    # ---- 7. alternative methods table -----------------------------------------------------------------------------
    if not args.skip_methods:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist, squareform
        from sklearn.cluster import KMeans, SpectralClustering
        from sklearn.decomposition import PCA
        from sklearn.neighbors import kneighbors_graph

        Xf = X.astype(float)
        U, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()

        def hier(D, method):
            return fcluster(linkage(D, method), k, "maxclust")[inv] - 1

        dj = np.nan_to_num(pdist(X.astype(bool), "jaccard"))
        Dj = np.nan_to_num(pdist(U.astype(bool), "jaccard"))
        w = -np.log(X.mean(0))
        inter = (U * w) @ U.T
        tot = (U * w).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            Dw = np.nan_to_num(1 - inter / (tot[:, None] + tot[None, :] - inter))
        A = kneighbors_graph(squareform(dj), n_neighbors=30, metric="precomputed", include_self=False)
        A = 0.5 * (A + A.T)
        methods = {
            "k-means, Euclidean on flags": KMeans(k, n_init=20, random_state=0).fit(Xf).labels_,
            "k-means on first two principal components": KMeans(k, n_init=20, random_state=0).fit(PCA(2).fit_transform(Xf)).labels_,
            "Ward linkage, Euclidean": hier(pdist(U), "ward"),
            "average linkage, Jaccard": hier(Dj, "average"),
            "complete linkage, Jaccard": hier(Dj, "complete"),
            "average linkage, inverse-prevalence weighted Jaccard": hier(squareform(Dw, checks=False), "average"),
            "spectral clustering, 30-NN Jaccard graph": SpectralClustering(k, affinity="precomputed", random_state=0, assign_labels="discretize")
            .fit(A)
            .labels_,
        }
        mrows = []
        for name, lab in methods.items():
            sizes = np.bincount(lab, minlength=k)
            mrows.append(
                {
                    "method": name,
                    "ari_vs_latent_classes": round(adjusted_rand_index(labels, lab), 3),
                    "largest_cluster_share": round(sizes.max() / len(lab), 3),
                    "cluster_sizes": sorted(sizes.tolist(), reverse=True),
                }
            )
        mrows.append(
            {
                "method": "distance facts",
                "ari_vs_latent_classes": "",
                "largest_cluster_share": "",
                "cluster_sizes": (
                    f"{len(np.unique(dj.round(6)))} distinct Jaccard values; {(dj == 1).mean():.0%} of pairs at distance 1; "
                    f"{len(U)} distinct rows; {int((X.sum(1) == 0).sum())} all-zero rows"
                ),
            }
        )
        pd.DataFrame(mrows).to_csv(OUT / "method_comparison.csv", index=False)
        log(pd.DataFrame(mrows).to_string(index=False))

    log(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
