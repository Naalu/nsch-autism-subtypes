"""The results figures and summary tables for the checkpoint and the paper.

Reads the outputs of steps 3 to 14 and draws the figures. No analysis happens
here beyond naming the classes and arranging numbers that the steps already
computed. Figure titles describe what is plotted; conclusions belong in the
text, so that a rerun with different numbers cannot produce a figure whose
title contradicts its contents.

Reads the main results folder (``results/`` by default: step 3 fits, step 13
held-out likelihood and the K choice, step 14 simulations) and the per-K folders beside it
(``results_k{K}/``: steps 4 to 12 at each K). Writes to
``reports/figures/results/`` and ``reports/tables/`` (``--reports`` changes
the parent folder)::

    classes.png                  condition rates by class at the chosen K
    severity_by_class.png        weighted severity distribution per class
    loadings.png                 slopes in the two-dimensional latent trait model
    shares_pooled_vs_2025.png    class shares, 2016-24 average and 2025
    checks_by_k.png              every agreement and stability check against K, with intervals
    model_fit.png                in-sample BIC and held-out likelihood against number of parameters
    map.png                      children in the two-dimensional space, with condition directions and classes
    latent_density.png           estimated distribution of positions: real data against simulated continuum and classes
    stability_vs_simulation.png  least stable class at each K, real data against simulated continuum and classes
    agreement_by_confidence.png  each check's ARI as children near a class boundary are set aside
    tables/class_names_k{K}.csv, tables/checks_by_k.csv, tables/method_comparison.csv

Usage::

    python scripts/make_results_figures.py            # the K chosen in step 13
    python scripts/make_results_figures.py --k 5      # a sensitivity analysis
    python scripts/make_results_figures.py --results results_ours --reports reports_ours
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import MAIN_YEARS, REPO, RESULTS_DIR, load_analysis, load_lca_fit, log, resolve_k  # noqa: E402
from nsch_subtypes.plots import CLASS_COLORS, CYAN, GRAY, GREEN, INK, PURPLE, RUST, tidy  # noqa: E402
from nsch_subtypes.validate import ARI_PASS, JACCARD_STABLE  # noqa: E402

DEVELOPMENTAL = ["dev_delay", "learning", "speech", "intellectual"]
PSYCHIATRIC = ["anxiety", "adhd", "depression", "behavior"]


def name_classes(rates: pd.DataFrame) -> pd.DataFrame:
    """Name each class by the average rate of four marker conditions on each axis.

    A class is "high" on an axis when the mean rate of its four marker
    conditions is at least 0.5. This is a fixed threshold, not a median
    split, so it does not force two classes onto each side; with other K or
    other data the names may not form a two-by-two. Classes that would share
    a name get their two most common conditions appended.
    """
    dev = rates[DEVELOPMENTAL].mean(axis=1)
    psych = rates[PSYCHIATRIC].mean(axis=1)
    names = []
    for i in rates.index:
        if dev[i] >= 0.5 and psych[i] >= 0.5:
            names.append("developmental and psychiatric")
        elif dev[i] >= 0.5:
            names.append("developmental-language")
        elif psych[i] >= 0.5:
            names.append("psychiatric")
        else:
            names.append("fewer co-occurring conditions")
    out = pd.DataFrame({"class": rates.index, "dev_marker_rate": dev.round(3), "psych_marker_rate": psych.round(3), "name": names})
    for _, group in out.groupby("name"):
        if len(group) > 1:
            for i in group.index:
                top = rates.loc[i].sort_values(ascending=False).index[:2]
                out.loc[i, "name"] = f"{out.loc[i, 'name']} ({', '.join(top)})"
    return out


def load_json(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def k_folders(main: Path) -> dict[int, Path]:
    """The per-K folders that sit beside the main results folder: ``results_k4``, ``results_k5``, ... -> {4: ..., 5: ...}."""
    found = {}
    for folder in main.parent.glob(f"{main.name}_k*"):
        suffix = folder.name[len(main.name) + 2 :]
        if folder.is_dir() and suffix.isdigit():
            found[int(suffix)] = folder
    return dict(sorted(found.items()))


def checks_table(main: Path) -> pd.DataFrame:
    """One row per K with every check's ARI and interval, and the least stable class."""
    rows = []
    for k, folder in k_folders(main).items():
        row = {"K": k}
        km = load_json(folder / "crosscheck_kmeans.json")
        row["second method (k-means)"] = km.get("ari")
        for name, file in (("held-out years", "replicate.json"), ("2025 forward test", "forward_2025.json"), ("'ever told' items", "parallel_ever.json")):
            d = load_json(folder / file)
            row[name] = d.get("ari", d.get("ari_current_vs_ever"))
            row[f"{name} low"], row[f"{name} high"] = d.get("ci_low"), d.get("ci_high")
        bands = load_json(folder / "age_bands.json").get("bands", {})
        for band, label in (("6_11", "age 6-11 refit"), ("12_17", "age 12-17 refit")):
            b = bands.get(band, {})
            row[label] = b.get("ari", b.get("ari_pooled_vs_band_fit"))
            row[f"{label} low"], row[f"{label} high"] = b.get("ci_low"), b.get("ci_high")
        stability = folder / "stability.csv"
        row["least stable class (Jaccard)"] = pd.read_csv(stability)["mean_jaccard"].min() if stability.exists() else None
        rows.append(row)
    # K=8 lives in the main results folder (steps 4 to 8 at BIC's minimum).
    if (main / "stability.csv").exists() and 8 not in [r["K"] for r in rows]:
        row = {"K": 8, "least stable class (Jaccard)": pd.read_csv(main / "stability.csv")["mean_jaccard"].min()}
        row["second method (k-means)"] = load_json(main / "crosscheck_kmeans.json").get("ari")
        row["held-out years"] = load_json(main / "replicate.json").get("ari")
        row["2025 forward test"] = load_json(main / "forward_2025.json").get("ari")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("K").set_index("K").astype(float)


def axis_names(slopes: np.ndarray, flags: list[str]) -> list[str]:
    """Label each latent dimension by the three conditions with the steepest slopes on it."""
    names = []
    for d in range(slopes.shape[1]):
        top = np.argsort(-np.abs(slopes[:, d]))[:3]
        names.append(f"dimension {d + 1} ({', '.join(flags[j] for j in top)})")
    return names


def draw_map(positions_path: Path, table: pd.DataFrame, fit, order: list[int], label_of: dict, path: Path) -> None:
    """Children in the two-dimensional space: density with condition directions, and the same children colored by class."""
    archive = np.load(positions_path)
    if not (np.array_equal(archive["hhid"], table["hhid"].to_numpy()) and np.array_equal(archive["year"], table["year"].to_numpy())):
        log(f"{positions_path} does not match the analysis table; rerun step 14. Skipping the map.")
        return
    xy, slopes, flags = archive["positions"], archive["slopes"], archive["flags"].tolist()
    names = axis_names(slopes, flags)
    labels, confidence = fit.labels, fit.responsibilities.max(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, sharey=True)

    ax = axes[0]
    ax.hexbin(xy[:, 0], xy[:, 1], gridsize=38, extent=(-3.5, 3.5, -3.5, 3.5), cmap="Greys", mincnt=1, linewidths=0)
    scale = 2.8 / np.linalg.norm(slopes, axis=1).max()
    for condition in DEVELOPMENTAL + PSYCHIATRIC:  # the marker conditions used to name the classes
        end = slopes[flags.index(condition)] * scale
        ax.annotate("", xy=end, xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": RUST, "lw": 1.2})
        ax.text(*(end + 0.18 * end / np.linalg.norm(end)), condition, fontsize=7.5, color=RUST, ha="center", va="center")
    ax.set_title("Children (one posterior draw each) and\nthe direction of each marker condition", fontsize=8.5, loc="left")

    ax = axes[1]
    unsure = confidence < 0.7
    ax.scatter(xy[unsure, 0], xy[unsure, 1], s=3, color="#c9ccd6", lw=0, label="largest class probability < 0.7")
    for rank, c in enumerate(order):
        member = (labels == c) & ~unsure
        ax.scatter(xy[member, 0], xy[member, 1], s=3, color=CLASS_COLORS[rank % len(CLASS_COLORS)], lw=0, alpha=0.6, label=label_of[c])
    ax.legend(frameon=False, fontsize=7, markerscale=3, loc="lower left")
    ax.set_title(f"The same children by most likely class (K={len(order)});\ngray: no class above 0.7", fontsize=8.5, loc="left")
    for ax in axes:
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_xlabel(names[0], fontsize=8)
        ax.set_aspect("equal")
        tidy(ax)
    axes[0].set_ylabel(names[1], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def draw_latent_density(density_npz: Path, density_csv: Path, flags_from: Path, path: Path) -> None:
    """Estimated distribution of positions for the real data and one simulated dataset of each kind."""
    from scipy.ndimage import gaussian_filter

    archive, summary = np.load(density_npz), pd.read_csv(density_csv)
    nodes = archive["nodes"]
    side = round(np.sqrt(len(nodes)))
    extent = (nodes[:, 0].min(), nodes[:, 0].max(), nodes[:, 1].min(), nodes[:, 1].max())
    positions = np.load(flags_from) if flags_from.exists() else None
    names = axis_names(positions["slopes"], positions["flags"].tolist()) if positions is not None else ["dimension 1", "dimension 2"]
    k = int(archive["class_model_k"])
    panels = [("real", "real", "2016-24 children"), ("continuum_0", "continuum", "simulated, no classes"), ("classes_0", "classes", f"simulated, {k} classes")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.2), sharex=True, sharey=True)
    for ax, (key, kind, title) in zip(axes, panels, strict=True):
        image = gaussian_filter(archive[key].reshape(side, side), 0.7).T  # light blur for display only
        ax.imshow(image, origin="lower", extent=extent, cmap="Greys", aspect="equal")
        values = summary.loc[summary["data"] == kind, "improvement"]
        if len(values) == 1:
            spread = f"{values.iloc[0]:.0f}"
        else:
            spread = f"{values.median():.0f} (median of {len(values)}; {values.min():.0f} to {values.max():.0f})"
        ax.set_title(f"{title}\ngain over the bell curve: {spread}", fontsize=8.5, loc="left")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel(names[0], fontsize=7.5)
    axes[0].set_ylabel(names[1], fontsize=7.5)
    fig.suptitle("Estimated distribution of children's positions in the two-dimensional model (gain in log-likelihood units)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def draw_stability_vs_simulation(real_csv: Path, simulated_csv: Path, path: Path) -> None:
    """Least stable class at each K for the real data and for simulated data with known structure."""
    real, simulated = pd.read_csv(real_csv), pd.read_csv(simulated_csv)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(real["k"], real["min_jaccard"], "o-", color=RUST, lw=2, label="real data (200 resamples; 100 at K=7)")
    styles = [(CYAN, "s--"), (PURPLE, "^--")]
    for (name, group), (color, marker) in zip(simulated.groupby("data", sort=False), styles, strict=False):
        label = "simulated, no classes (50 resamples)" if name == "continuum" else f"simulated, {name} (50 resamples)"
        ax.plot(group["k"], group["min_jaccard"], marker, color=color, label=label)
    ax.axhline(JACCARD_STABLE, color=INK, lw=1, ls=":")
    ax.text(ax.get_xlim()[1], JACCARD_STABLE, f" {JACCARD_STABLE}", fontsize=7, va="center")
    ax.set_xlabel("number of classes K")
    ax.set_ylabel("least stable class, mean bootstrap Jaccard")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_title("Class stability at each K, real and simulated data", fontsize=9, loc="left")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def draw_agreement_by_confidence(kdir: Path, path: Path) -> None:
    """Each check's ARI as children with lower class probability are set aside, against the share of children kept."""
    series = []
    km = load_json(kdir / "crosscheck_kmeans.json")
    if km:
        series.append(("second method (k-means)", km.get("ari"), km.get("ari_by_confidence", [])))
    for name, file in (("held-out years", "replicate.json"), ("2025 forward test", "forward_2025.json"), ("'ever told' items", "parallel_ever.json")):
        d = load_json(kdir / file)
        if d:
            series.append((name, d.get("ari"), d.get("ari_by_confidence", [])))
    for band, label in (("6_11", "age 6-11 refit"), ("12_17", "age 12-17 refit")):
        b = load_json(kdir / "age_bands.json").get("bands", {}).get(band)
        if b:
            series.append((label, b.get("ari"), b.get("ari_by_confidence", [])))
    weighted = load_json(kdir / "robustness.json").get("weighted_fit")
    if weighted:
        series.append(("survey-weighted fit", weighted.get("ari_vs_unweighted"), weighted.get("ari_by_confidence", [])))
    colors = [RUST, CYAN, PURPLE, "#b7791f", GREEN, "#9b2c2c", GRAY]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for (name, ari_all, rows), color in zip(series, colors, strict=False):
        if ari_all is None or not rows:
            continue
        shares = [1.0] + [r["share_of_children"] for r in rows]
        aris = [ari_all] + [r["ari"] for r in rows]
        ax.plot(shares, aris, "o-", color=color, label=name, lw=1.4, ms=4)
    ax.axhline(ARI_PASS, color=INK, lw=1, ls=":")
    ax.text(0.31, ARI_PASS + 0.01, f"pass line {ARI_PASS}", fontsize=7, ha="right")
    ax.set_xlim(1.02, 0.3)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("share of children kept (all, then largest class probability at least 0.6, 0.7, 0.8, 0.9)")
    ax.set_ylabel("adjusted Rand index")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title("Agreement as children near a class boundary are set aside", fontsize=9, loc="left")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Results figures")
    parser.add_argument("--k", type=int, default=None, help="the K to draw (default: the choice from step 13)")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR, help="main results folder; the per-K folders are found beside it")
    parser.add_argument("--reports", type=Path, default=REPO / "reports", help="where figures/results/ and tables/ are written")
    args = parser.parse_args()
    main_dir = args.results
    out_dir, tables_dir = args.reports / "figures" / "results", args.reports / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    k = resolve_k(args, main_dir)
    kdir = main_dir.parent / f"{main_dir.name}_k{k}"
    table, flags = load_analysis(MAIN_YEARS)
    fit = load_lca_fit(kdir, k, table, flags)
    rates = pd.DataFrame(fit.condition_rates, columns=flags)
    shares = pd.read_csv(kdir / "shares.csv").set_index("class")
    names = name_classes(rates)
    names["n_children_modal"] = np.bincount(fit.labels, minlength=k)
    names["weighted_share"] = shares["share"].round(3).to_numpy()
    names.to_csv(tables_dir / f"class_names_k{k}.csv", index=False)
    label_of = dict(zip(names["class"], names["name"], strict=True))
    log("class names:\n" + names.to_string(index=False))
    order = list(names.sort_values(["dev_marker_rate", "psych_marker_rate"]).index)

    # ---- 1. condition rates by class ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(0.5 * len(flags) + 2.8, 0.55 * k + 1.9))
    data = rates.loc[order].to_numpy()
    image = ax.imshow(data, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(flags)))
    ax.set_xticklabels(flags, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"{label_of[c]}\n{shares.loc[c, 'share']:.0%} (weighted)" for c in order], fontsize=8)
    for i in range(k):
        for j in range(len(flags)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if data[i, j] > 0.55 else INK)
    for group, color in ((DEVELOPMENTAL, CYAN), (PSYCHIATRIC, PURPLE)):
        xs = [flags.index(f) for f in group]
        ax.plot([min(xs) - 0.45, max(xs) + 0.45], [-0.75, -0.75], color=color, lw=3, clip_on=False)
    ax.set_title(
        f"Condition rates by latent class, K={k}. Brackets: psychiatric (purple) and developmental-language (blue) marker conditions used to name the classes.",
        fontsize=8,
        loc="left",
        pad=14,
    )
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="share of class with the condition")
    fig.tight_layout()
    fig.savefig(out_dir / "classes.png", dpi=170)
    plt.close(fig)

    # ---- 2. severity by class ----------------------------------------------------------------------
    severity = pd.read_csv(kdir / "severity_by_class.csv").set_index("class").loc[order]
    fig, ax = plt.subplots(figsize=(7, 0.55 * k + 1.4))
    left = np.zeros(k)
    for level, color in (("mild", "#f6c89f"), ("moderate", RUST), ("severe", "#5a2a0e")):
        values = severity[level].to_numpy()
        ax.barh(range(k), values, left=left, color=color, label=level)
        for i, (v, start) in enumerate(zip(values, left, strict=True)):
            if v > 0.06:
                ax.text(start + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white" if level != "mild" else INK)
        left += values
    ax.set_yticks(range(k))
    ax.set_yticklabels([label_of[c] for c in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("weighted share of class")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0))
    ax.set_title(f"Parent-rated autism severity by latent class, K={k}", fontsize=9, loc="left", pad=18)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "severity_by_class.png", dpi=170)
    plt.close(fig)

    # ---- 3. loadings --------------------------------------------------------------------------------
    loadings_path = kdir / "loadings_2d.csv"
    if loadings_path.exists():
        loadings = pd.read_csv(loadings_path).sort_values("dim1_slope")
        fig, ax = plt.subplots(figsize=(6.5, 5))
        y = np.arange(len(loadings))
        ax.barh(y - 0.2, loadings["dim1_slope"], height=0.4, color=CYAN, label="dimension 1")
        ax.barh(y + 0.2, loadings["dim2_slope"], height=0.4, color=PURPLE, label="dimension 2")
        ax.set_yticks(y)
        ax.set_yticklabels(loadings["condition"], fontsize=8)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_xlabel("slope (how strongly the condition depends on each dimension)")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.set_title("Slopes in the two-dimensional latent trait model, after varimax rotation", fontsize=9, loc="left")
        tidy(ax)
        fig.tight_layout()
        fig.savefig(out_dir / "loadings.png", dpi=170)
        plt.close(fig)

    # ---- 4. shares, pooled vs 2025 ------------------------------------------------------------------
    shares_2025_path = kdir / "shares_2025.csv"
    if shares_2025_path.exists():
        shares_2025 = pd.read_csv(shares_2025_path).set_index("class")
        fig, ax = plt.subplots(figsize=(6.8, 3.3))
        xs = np.arange(k)
        for offset, frame, color, label in ((-0.17, shares, RUST, "2016-2024 average"), (0.17, shares_2025, CYAN, "2025, frozen 2016-24 model")):
            d = frame.loc[order]
            ax.bar(xs + offset, d["share"], width=0.34, color=color, label=label)
            ax.errorbar(xs + offset, d["share"], yerr=[d["share"] - d["ci_low"], d["ci_high"] - d["share"]], fmt="none", ecolor=INK, capsize=3, lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([label_of[c].replace(" and ", "\nand ").replace(" co-occurring", "\nco-occurring") for c in order], fontsize=8)
        ax.set_ylabel("weighted share, with 95% CI")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"Class shares among US autistic children aged 6-17, K={k}", fontsize=9, loc="left")
        tidy(ax)
        fig.tight_layout()
        fig.savefig(out_dir / "shares_pooled_vs_2025.png", dpi=170)
        plt.close(fig)

    # ---- 5. checks by K -------------------------------------------------------------------------------
    checks = checks_table(main_dir)
    checks.round(3).to_csv(tables_dir / "checks_by_k.csv")
    log("checks by K:\n" + checks.round(3).to_string())
    fig, ax = plt.subplots(figsize=(7.4, 4))
    series = [
        ("second method (k-means)", RUST),
        ("held-out years", CYAN),
        ("2025 forward test", PURPLE),
        ("age 6-11 refit", GREEN),
        ("age 12-17 refit", "#9b2c2c"),
        ("'ever told' items", "#b7791f"),
    ]
    for i, (col, color) in enumerate(series):
        if col not in checks or checks[col].isna().all():
            continue
        x = checks.index + (i - 2.5) * 0.04
        ax.plot(x, checks[col], "o-", color=color, label=col, lw=1.3)
        if f"{col} low" in checks:
            ax.vlines(x, checks[f"{col} low"], checks[f"{col} high"], color=color, lw=1)
    ax.plot(checks.index, checks["least stable class (Jaccard)"], "s--", color=GRAY, label="least stable class, bootstrap Jaccard")
    ax.axhline(ARI_PASS, color=INK, lw=1, ls=":")
    ax.axhline(JACCARD_STABLE, color=GRAY, lw=1, ls=":")
    ax.text(checks.index.max() + 0.1, ARI_PASS, f"ARI line {ARI_PASS}", fontsize=7, va="center")
    ax.text(checks.index.max() + 0.1, JACCARD_STABLE, f"Jaccard line {JACCARD_STABLE}", fontsize=7, va="center", color=GRAY)
    ax.set_xlabel("number of classes K")
    ax.set_ylabel("adjusted Rand index (with 95% interval) / Jaccard")
    ax.set_xticks(checks.index)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
    ax.set_title("Agreement and stability checks at each K", fontsize=9, loc="left")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "checks_by_k.png", dpi=170)
    plt.close(fig)

    # ---- 6. model fit: in-sample BIC and held-out likelihood ----------------------------------------
    comparison_path = kdir / "model_comparison.csv"
    heldout_path = main_dir / "heldout_selection.csv"
    if comparison_path.exists() and heldout_path.exists():
        comparison = pd.read_csv(comparison_path)
        heldout = pd.read_csv(heldout_path)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        panels = [
            (axes[0], comparison, "bic", "BIC on 2016-24 (lower is better)", "in-sample"),
            (axes[1], heldout[heldout["split"] == "2022_24"], "diff_from_best", "held-out log-likelihood minus best", "fit 2016-21, scored on 2022-24"),
            (axes[2], heldout[heldout["split"] == "2025"], "diff_from_best", "held-out log-likelihood minus best", "fit 2016-24, scored on 2025"),
        ]
        for ax, frame, column, ylabel, title in panels:
            for family, color, marker in (("latent class", RUST, "o-"), ("latent trait", PURPLE, "s-")):
                f = frame[frame["family"] == family].sort_values("n_parameters")
                ax.plot(f["n_parameters"], f[column], marker, color=color, label=family)
                prefix = "K" if family == "latent class" else "D"
                for _, r in f.iterrows():
                    point = (r["n_parameters"], r[column])
                    ax.annotate(f"{prefix}{int(r['size'])}", point, fontsize=6.5, xytext=(3, 3), textcoords="offset points")
            if column == "diff_from_best":
                ax.axhline(0, color=INK, lw=0.8)
                ax.set_ylim(-700, 60)
            else:
                ax.set_ylim(frame[column].min() - 300, frame[column].min() + 4000)
            ax.set_xlabel("number of parameters")
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=9, loc="left")
            tidy(ax)
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "model_fit.png", dpi=170)
        plt.close(fig)

    # ---- 7. the two-dimensional picture and the simulations (step 14) ---------------------------------
    if (main_dir / "positions_2d.npz").exists():
        draw_map(main_dir / "positions_2d.npz", table, fit, order, label_of, out_dir / "map.png")
    if (main_dir / "latent_density.npz").exists():
        draw_latent_density(main_dir / "latent_density.npz", main_dir / "latent_density.csv", main_dir / "positions_2d.npz", out_dir / "latent_density.png")
    if (main_dir / "stability_calibration.csv").exists() and (main_dir / "stability_by_k.csv").exists():
        draw_stability_vs_simulation(main_dir / "stability_by_k.csv", main_dir / "stability_calibration.csv", out_dir / "stability_vs_simulation.png")
    draw_agreement_by_confidence(kdir, out_dir / "agreement_by_confidence.png")

    # ---- 8. tables carried over from the steps ------------------------------------------------------------
    for source in (
        kdir / "method_comparison.csv",
        main_dir / "stability_by_k.csv",
        main_dir / "heldout_selection.csv",
        main_dir / "latent_density.csv",
        main_dir / "stability_calibration.csv",
    ):
        if source.exists():
            shutil.copy(source, tables_dir / source.name)
    log(f"wrote figures to {out_dir} and tables to {tables_dir}")


if __name__ == "__main__":
    main()
