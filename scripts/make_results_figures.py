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
    shares_by_year.png           class shares in each survey year under the pooled model
    checks_by_k.png              every agreement and stability check against K, with intervals
    model_fit.png                in-sample BIC and held-out likelihood against number of parameters
    map.png                      children in the two-dimensional space, with condition directions and classes
    figure1_map.png              the paper's Figure 1: one cloud on the two named axes, classes as regions, cut lines
    severity_map.png             share rated severe in each region of the map
    checks_k{K}.png              every agreement check at the chosen K with intervals and simulated ranges
    latent_density.png           estimated distribution of positions: real data against simulated continuum and classes
    stability_vs_simulation.png  least stable class at each K, real data against simulated continuum and classes
    replication_vs_simulation.png each replication check's real ARI against its range on simulated continuum and class data
    agreement_by_confidence.png  each check's ARI as children near a class boundary are set aside
    tables/class_names_k{K}.csv, tables/checks_by_k.csv, tables/method_comparison.csv

Usage::

    python scripts/make_results_figures.py            # the K chosen in step 13
    python scripts/make_results_figures.py --k 5      # a sensitivity analysis
    python scripts/make_results_figures.py --results results_reference
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

# One color per axis and per class, used by every figure. The developmental-
# language axis is blue and the psychiatric axis purple; a class high on one
# takes that axis's color, the class high on both takes rust, and the class
# low on both gray. Names outside the two-by-two (other K) fall back to the
# general palette in class order.
AXIS_COLORS = {"developmental-language": CYAN, "psychiatric": PURPLE}
NAME_COLORS = {
    "fewer co-occurring conditions": "#8a8f9e",
    "developmental-language": CYAN,
    "psychiatric": PURPLE,
    "developmental and psychiatric": RUST,
}
SEVERITY_COLORS = {"mild": "#f6c89f", "moderate": RUST, "severe": "#5a2a0e"}


def class_colors(order: list[int], label_of: dict) -> dict:
    """Class index -> color: the name's color when the name is one of the four, otherwise the general palette."""
    colors, spare = {}, iter(c for c in CLASS_COLORS if c not in NAME_COLORS.values())
    for c in order:
        colors[c] = NAME_COLORS.get(label_of[c]) or next(spare, GRAY)
    return colors


def axis_roles(slopes: np.ndarray, flags: list[str]) -> list[str]:
    """Which axis each dimension is: the marker group with the larger mean absolute slope on it."""
    roles = []
    for d in range(slopes.shape[1]):
        dev = np.abs(slopes[[flags.index(f) for f in DEVELOPMENTAL], d]).mean()
        psych = np.abs(slopes[[flags.index(f) for f in PSYCHIATRIC], d]).mean()
        roles.append("developmental-language" if dev > psych else "psychiatric")
    return roles


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
    return pd.DataFrame(rows).sort_values("K").set_index("K").astype(float)


SHORT_LABELS = {
    "adhd": "ADHD",
    "anxiety": "anxiety",
    "depression": "depression",
    "behavior": "behavior problems",
    "dev_delay": "developmental delay",
    "intellectual": "intellectual disability",
    "speech": "speech or language",
    "learning": "learning disability",
}


def axis_names(slopes: np.ndarray, flags: list[str]) -> list[str]:
    """Label each latent dimension by its role and the three marker conditions with the steepest slopes on it."""
    names = []
    for d, role in enumerate(axis_roles(slopes, flags)):
        markers = DEVELOPMENTAL if role == "developmental-language" else PSYCHIATRIC
        top = sorted(markers, key=lambda f: -abs(slopes[flags.index(f), d]))[:3]
        names.append(f"{role} axis ({', '.join(SHORT_LABELS.get(f, f) for f in top)})")
    return names


def _load_positions(positions_path: Path, table: pd.DataFrame):
    archive = np.load(positions_path)
    if not (np.array_equal(archive["hhid"], table["hhid"].to_numpy()) and np.array_equal(archive["year"], table["year"].to_numpy())):
        log(f"{positions_path} does not match the analysis table; rerun step 14. Skipping the map.")
        return None
    return archive["positions"], archive["slopes"], archive["flags"].tolist()


def draw_map(positions_path: Path, table: pd.DataFrame, fit, order: list[int], label_of: dict, colors: dict, path: Path) -> None:
    """Children in the two-dimensional space: density with condition directions, and the same children colored by class."""
    loaded = _load_positions(positions_path, table)
    if loaded is None:
        return
    xy, slopes, flags = loaded
    names = axis_names(slopes, flags)
    labels, confidence = fit.labels, fit.responsibilities.max(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, sharey=True)

    ax = axes[0]
    ax.hexbin(xy[:, 0], xy[:, 1], gridsize=38, extent=(-3.5, 3.5, -3.5, 3.5), cmap="Greys", mincnt=1, linewidths=0)
    scale = 2.8 / np.linalg.norm(slopes, axis=1).max()
    for group, color in ((DEVELOPMENTAL, AXIS_COLORS["developmental-language"]), (PSYCHIATRIC, AXIS_COLORS["psychiatric"])):
        for condition in group:  # the marker conditions used to name the classes
            end = slopes[flags.index(condition)] * scale
            ax.annotate("", xy=end, xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2})
            ax.text(*(end + 0.18 * end / np.linalg.norm(end)), condition, fontsize=7.5, color=color, ha="center", va="center")
    ax.set_title("Children (one posterior draw each) and\nthe direction of each marker condition", fontsize=8.5, loc="left")

    ax = axes[1]
    unsure = confidence < 0.7
    ax.scatter(xy[unsure, 0], xy[unsure, 1], s=3, color="#c9ccd6", lw=0, label="largest class probability < 0.7")
    for c in order:
        member = (labels == c) & ~unsure
        ax.scatter(xy[member, 0], xy[member, 1], s=3, color=colors[c], lw=0, alpha=0.6, label=label_of[c])
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


def draw_figure1(
    positions_path: Path, table: pd.DataFrame, fit, order: list[int], label_of: dict, colors: dict, shares: pd.DataFrame, cut_points, path: Path
) -> None:
    """Figure 1: one cloud of children on the two axes, the classes as its regions, uncertain children faded.

    The developmental-language axis goes across and the psychiatric axis up,
    whichever order the trait model returned them in. ``cut_points`` (from
    step 12, on posterior-mean scores) are drawn as the two lines that best
    split the classes into quadrants.
    """
    loaded = _load_positions(positions_path, table)
    if loaded is None:
        return
    xy, slopes, flags = loaded
    roles = axis_roles(slopes, flags)
    if roles == ["psychiatric", "developmental-language"]:
        xy, slopes = xy[:, ::-1], slopes[:, ::-1]
        cut_points = None if cut_points is None else np.asarray(cut_points)[::-1]
    names = axis_names(slopes, flags)
    labels, confidence = fit.labels, fit.responsibilities.max(axis=1)
    sure = confidence >= 0.8
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    for c in order:
        member = labels == c
        ax.scatter(xy[member & ~sure, 0], xy[member & ~sure, 1], s=3, color=colors[c], lw=0, alpha=0.18)
        ax.scatter(xy[member & sure, 0], xy[member & sure, 1], s=3, color=colors[c], lw=0, alpha=0.65)
    if cut_points is not None:
        ax.axvline(cut_points[0], color=INK, lw=0.8, ls="--")
        ax.axhline(cut_points[1], color=INK, lw=0.8, ls="--")
    # Label each class in the corner of the quadrant its confident members sit in.
    centre = np.zeros(2) if cut_points is None else np.asarray(cut_points, dtype=float)
    for c in order:
        member = (labels == c) & sure
        if member.sum() == 0:
            continue
        side = np.sign(np.median(xy[member], axis=0) - centre)
        x, y = 3.35 * side[0], 3.35 * side[1]
        ax.text(
            x,
            y,
            f"{label_of[c]}\n{shares.loc[c, 'share']:.0%}",
            fontsize=8.5,
            color=colors[c],
            fontweight="bold",
            ha="right" if side[0] > 0 else "left",
            va="top" if side[1] > 0 else "bottom",
        )
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.set_xlabel(names[0], fontsize=8.5, color=AXIS_COLORS.get(axis_roles(slopes, flags)[0], INK))
    ax.set_ylabel(names[1], fontsize=8.5, color=AXIS_COLORS.get(axis_roles(slopes, flags)[1], INK))
    ax.set_title(
        f"{len(xy):,} autistic children aged 6-17 (NSCH 2016-24) on the two axes of the latent trait model.\n"
        "Color: most likely class (weighted share of US children); faded: probability below 0.8.\nDashed: the two cuts that best match the classes.",
        fontsize=7.5,
        loc="left",
    )
    tidy(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def draw_severity_map(positions_path: Path, table: pd.DataFrame, path: Path) -> None:
    """Share of children rated severe in each region of the map: severity follows position, not a class border."""
    loaded = _load_positions(positions_path, table)
    if loaded is None:
        return
    xy, slopes, flags = loaded
    roles = axis_roles(slopes, flags)
    if roles == ["psychiatric", "developmental-language"]:
        xy, slopes = xy[:, ::-1], slopes[:, ::-1]
    names = axis_names(slopes, flags)
    rated = table["severity"].notna().to_numpy()
    severe = (table["severity"] == "severe").to_numpy().astype(float)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    image = ax.hexbin(
        xy[rated, 0],
        xy[rated, 1],
        C=severe[rated],
        reduce_C_function=np.mean,
        gridsize=16,
        extent=(-3.5, 3.5, -3.5, 3.5),
        mincnt=15,
        cmap="Oranges",
        vmin=0,
        vmax=0.5,
        linewidths=0.2,
        edgecolors="white",
    )
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, label="share rated severe (cells with 15 or more children)")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.set_xlabel(names[0], fontsize=8.5)
    ax.set_ylabel(names[1], fontsize=8.5)
    ax.set_title(f"Parent-rated severity across the map ({rated.sum():,} children with a rating)", fontsize=8.5, loc="left")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def draw_checks_dot_plot(kdir: Path, k: int, path: Path) -> None:
    """Every check at the chosen K on one axis: the real value with its interval, and the simulated ranges where step 15 ran."""
    rows = []
    km = load_json(kdir / "crosscheck_kmeans.json")
    if km:
        rows.append(("second method (k-means)", km.get("ari"), None, None))
    for name, file in (("held-out years", "replicate.json"), ("2025 forward test", "forward_2025.json"), ("'ever told' items", "parallel_ever.json")):
        d = load_json(kdir / file)
        if d:
            rows.append((name, d.get("ari", d.get("ari_current_vs_ever")), d.get("ci_low"), d.get("ci_high")))
    bands = load_json(kdir / "age_bands.json").get("bands", {})
    for band, label in (("6_11", "age 6-11 refit"), ("12_17", "age 12-17 refit")):
        if band in bands:
            b = bands[band]
            rows.append((label, b.get("ari"), b.get("ci_low"), b.get("ci_high")))
    robust = load_json(kdir / "robustness.json")
    if robust.get("weighted_fit"):
        w = robust["weighted_fit"]
        rows.append(("survey-weighted fit", w.get("ari_vs_unweighted"), *(w.get("ari_interval") or [None, None])))
    if robust.get("missing_as_missing"):
        m = robust["missing_as_missing"]
        rows.append(("children with blanks kept", m.get("ari_vs_complete_case"), *(m.get("ari_interval") or [None, None])))
    rows = [r for r in rows if r[1] is not None]
    if not rows:
        return
    simulated = pd.read_csv(kdir / "replication_calibration.csv") if (kdir / "replication_calibration.csv").exists() else None
    stability = kdir / "stability.csv"
    fig, ax = plt.subplots(figsize=(6.8, 0.42 * len(rows) + 1.6))
    y = np.arange(len(rows))
    for i, (name, value, low, high) in enumerate(rows):
        if simulated is not None and name in set(simulated["check"]):
            for kind, color, offset in (("continuum", CYAN, -0.22), (f"{k} classes", PURPLE, 0.22)):
                g = simulated[(simulated["data"] == kind) & (simulated["check"] == name)]["ari"]
                if len(g):
                    ax.hlines(i + offset, g.min(), g.max(), color=color, lw=4, alpha=0.5)
        if low is not None and high is not None:
            ax.hlines(i, low, high, color=RUST, lw=1.4)
        ax.plot(value, i, "o", color=RUST, ms=6, zorder=3)
        ax.text(1.0, i, f"{value:.2f}", fontsize=7.5, va="center", ha="left")
    ax.axvline(ARI_PASS, color=INK, lw=1, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("adjusted Rand index against the main classes (95% interval)")
    handles = [plt.Line2D([], [], marker="o", color=RUST, ls="-", label="real data")]
    if simulated is not None:
        handles += [
            plt.Line2D([], [], color=CYAN, lw=4, alpha=0.5, label="simulated, no classes"),
            plt.Line2D([], [], color=PURPLE, lw=4, alpha=0.5, label=f"simulated, {k} classes"),
        ]
    handles.append(plt.Line2D([], [], color=INK, ls=":", label=f"pass line {ARI_PASS}"))
    ax.legend(handles=handles, frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    title = f"Every agreement check at K={k}"
    if stability.exists():
        title += f"; least stable class under resampling: Jaccard {pd.read_csv(stability)['mean_jaccard'].min():.2f}"
    ax.set_title(title, fontsize=8.5, loc="left")
    tidy(ax)
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
    ax.plot(real["k"], real["min_jaccard"], "o-", color=RUST, lw=2, label="real data (200 resamples)")
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


def draw_replication_vs_simulation(calibration_csv: Path, path: Path) -> None:
    """Each replication check's real ARI against the range the same check produces on simulated continuum and class data."""
    table = pd.read_csv(calibration_csv)
    checks = [c for c in table["check"].unique()]
    simulated = table[table["data"] != "real"]
    real = table[table["data"] == "real"].set_index("check")["ari"]
    kinds = list(simulated["data"].unique())
    colors = {kinds[0]: CYAN} if kinds else {}
    if len(kinds) > 1:
        colors[kinds[1]] = PURPLE
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    y = np.arange(len(checks))
    offsets = np.linspace(-0.18, 0.18, max(len(kinds), 1))
    for kind, offset in zip(kinds, offsets, strict=False):
        group = simulated[simulated["data"] == kind]
        lo = group.groupby("check")["ari"].min().reindex(checks)
        hi = group.groupby("check")["ari"].max().reindex(checks)
        ax.hlines(y + offset, lo, hi, color=colors[kind], lw=5, alpha=0.55, label=f"simulated, {'no classes' if kind == 'continuum' else kind}")
    present = [c for c in checks if c in real.index]
    ax.plot(real.reindex(present).to_numpy(), [checks.index(c) for c in present], "o", color=RUST, ms=7, label="real data", zorder=3)
    ax.axvline(ARI_PASS, color=INK, lw=1, ls=":")
    ax.text(ARI_PASS, -0.55, f" pass line {ARI_PASS}", fontsize=7, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(checks)
    ax.set_ylim(len(checks) - 0.5, -0.8)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("adjusted Rand index")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.set_title("Replication checks on the real data and on data with known structure", fontsize=9, loc="left")
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


def report_output_directories(main_results: Path, reports_root: Path) -> tuple[Path, Path]:
    """Keep the reference (library) outputs separate from the main, from-scratch ones.

    This follows the same results_reference naming convention as sweep_k.sh.
    A custom --reports parent still works, including the smoke-test folders.
    """
    if main_results.name.startswith("results_reference"):
        return reports_root / "figures" / "results_reference", reports_root / "tables" / "results_reference"
    return reports_root / "figures" / "results", reports_root / "tables"


def main() -> None:
    parser = argparse.ArgumentParser(description="Results figures")
    parser.add_argument("--k", type=int, default=None, help="the K to draw (default: the choice from step 13)")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR, help="main results folder; the per-K folders are found beside it")
    parser.add_argument("--reports", type=Path, default=REPO / "reports", help="report root; results_reference inputs use separate figure and table subfolders")
    args = parser.parse_args()
    main_dir = args.results
    out_dir, tables_dir = report_output_directories(main_dir, args.reports)
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
    canonical = list(NAME_COLORS)  # low on both, developmental, psychiatric, high on both
    if all(label_of[c] in canonical for c in order):
        order = sorted(order, key=lambda c: canonical.index(label_of[c]))
    colors = class_colors(order, label_of)

    # ---- 1. condition rates by class ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(0.5 * len(flags) + 2.8, 0.55 * k + 1.9))
    data = rates.loc[order].to_numpy()
    image = ax.imshow(data, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(flags)))
    ax.set_xticklabels(flags, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"{label_of[c]}\n{shares.loc[c, 'share']:.0%} (weighted)" for c in order], fontsize=8)
    for tick, c in zip(ax.get_yticklabels(), order, strict=True):
        tick.set_color(colors[c])
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
    rated = table["severity"].notna().to_numpy()
    n_rated = {c: int(((fit.labels == c) & rated).sum()) for c in order}
    fig, ax = plt.subplots(figsize=(7, 0.55 * k + 1.4))
    left = np.zeros(k)
    for level, color in SEVERITY_COLORS.items():
        values = severity[level].to_numpy()
        ax.barh(range(k), values, left=left, color=color, label=level)
        for i, (v, start) in enumerate(zip(values, left, strict=True)):
            if 0 < v < 0.03 and level == "severe":  # too thin to hold its label: put it just past the bar
                ax.text(1.005, i, f"{v:.0%}", ha="left", va="center", fontsize=6.5, color=INK)
            if v >= 0.03:  # every segment wide enough to hold a label gets one
                ax.text(start + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=7.5 if v >= 0.06 else 6.5, color="white" if level != "mild" else INK)
        left += values
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"{label_of[c]}\n(n = {n_rated[c]:,})" for c in order], fontsize=8)
    for tick, c in zip(ax.get_yticklabels(), order, strict=True):
        tick.set_color(colors[c])
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
        roles = axis_roles(loadings[["dim1_slope", "dim2_slope"]].to_numpy(), loadings["condition"].tolist())
        fig, ax = plt.subplots(figsize=(6.5, 5))
        y = np.arange(len(loadings))
        for d, offset in ((1, -0.2), (2, 0.2)):
            role = roles[d - 1]
            se = loadings.get(f"dim{d}_se")
            ax.barh(
                y + offset,
                loadings[f"dim{d}_slope"],
                height=0.4,
                color=AXIS_COLORS[role],
                label=f"{role} axis",
                xerr=None if se is None else 1.96 * se,
                error_kw={"lw": 0.8, "ecolor": INK, "capsize": 1.5},
            )
        ax.set_yticks(y)
        ax.set_yticklabels(loadings["condition"], fontsize=8)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_xlabel("slope (how strongly the condition depends on each axis)")
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
        title = "Slopes in the two-dimensional latent trait model, after varimax rotation"
        if "dim1_se" in loadings:
            title += "\n(bars: 95% bootstrap intervals)"
        ax.set_title(title, fontsize=9, loc="left")
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
        for tick, c in zip(ax.get_xticklabels(), order, strict=True):
            tick.set_color(colors[c])
        ax.set_ylabel("weighted share, with 95% CI")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"Class shares among US autistic children aged 6-17, K={k}", fontsize=9, loc="left")
        tidy(ax)
        fig.tight_layout()
        fig.savefig(out_dir / "shares_pooled_vs_2025.png", dpi=170)
        plt.close(fig)

    # ---- 4b. shares by survey year ------------------------------------------------------------------
    by_year_path = kdir / "shares_by_year.csv"
    if by_year_path.exists():
        by_year = pd.read_csv(by_year_path)
        fig, ax = plt.subplots(figsize=(6.8, 3.4))
        for c in order:
            d = by_year[by_year["class"] == c].sort_values("year")
            color = colors[c]
            ax.plot(d["year"], d["share"], "o-", color=color, lw=1.6, ms=4, label=label_of[c])
            ax.fill_between(d["year"], d["ci_low"], d["ci_high"], color=color, alpha=0.12, lw=0)
        ax.set_xticks(sorted(by_year["year"].unique()))
        ax.set_ylim(0, max(0.5, float(by_year["ci_high"].max()) + 0.02))
        ax.set_ylabel("weighted share, with 95% CI")
        ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        ax.set_title(f"Class shares by survey year under the pooled K={k} model", fontsize=9, loc="left")
        tidy(ax)
        fig.tight_layout()
        fig.savefig(out_dir / "shares_by_year.png", dpi=170)
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
        # Held-out panels are measured from the best model whose fit is not at the slope bound,
        # when the table has that column (older tables only have the overall best).
        diff, se = ("diff_from_best_interior", "diff_interior_se") if "diff_from_best_interior" in heldout else ("diff_from_best", "diff_se")
        panels = [
            (axes[0], comparison, "bic", None, "BIC on 2016-24 (lower is better)", "in-sample"),
            (axes[1], heldout[heldout["split"] == "2022_24"], diff, se, "held-out log-likelihood minus best", "fit 2016-21, scored on 2022-24"),
            (axes[2], heldout[heldout["split"] == "2025"], diff, se, "held-out log-likelihood minus best", "fit 2016-24, scored on 2025"),
        ]
        for ax, frame, column, se_column, ylabel, title in panels:
            if column == "bic":
                low, high = frame[column].min() - 300, frame[column].min() + 4000
            else:
                low, high = -700, 120
            off_axis = []
            for family, color, marker in (("latent class", RUST, "o"), ("latent trait", PURPLE, "s")):
                f = frame[frame["family"] == family].sort_values("n_parameters")
                bound = f["at_bound"].astype(bool) if "at_bound" in f else pd.Series(False, index=f.index)
                ok = f[~bound]
                ax.plot(ok["n_parameters"], ok[column], marker + "-", color=color, label=family)
                if se_column and se_column in ok:
                    ax.errorbar(ok["n_parameters"], ok[column], yerr=1.96 * ok[se_column], fmt="none", ecolor=color, lw=0.8, capsize=2)
                if bound.any():
                    b = f[bound]
                    ax.plot(b["n_parameters"], b[column], marker, mfc="white", color=color, label=f"{family}, slope at bound (degenerate)")
                prefix = "K" if family == "latent class" else "D"
                for _, r in f.iterrows():
                    if not low <= r[column] <= high:
                        off_axis.append(f"{prefix}{int(r['size'])}")
                        continue
                    ax.annotate(f"{prefix}{int(r['size'])}", (r["n_parameters"], r[column]), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
            if column != "bic":
                ax.axhline(0, color=INK, lw=0.8)
            ax.set_ylim(low, high)
            if off_axis:
                ax.text(0.98, 0.03, "below the axis: " + ", ".join(off_axis), transform=ax.transAxes, ha="right", fontsize=6.5, color=GRAY)
            ax.set_xlabel("number of parameters")
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=9, loc="left")
            tidy(ax)
        axes[0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "model_fit.png", dpi=170)
        plt.close(fig)

    # ---- 7. the two-dimensional picture and the simulations (step 14) ---------------------------------
    if (main_dir / "positions_2d.npz").exists():
        draw_map(main_dir / "positions_2d.npz", table, fit, order, label_of, colors, out_dir / "map.png")
        cut_points = load_json(kdir / "dimensions.json").get("quadrants", {}).get("cut_points")
        draw_figure1(main_dir / "positions_2d.npz", table, fit, order, label_of, colors, shares, cut_points, out_dir / "figure1_map.png")
        draw_severity_map(main_dir / "positions_2d.npz", table, out_dir / "severity_map.png")
    if (main_dir / "latent_density.npz").exists():
        draw_latent_density(main_dir / "latent_density.npz", main_dir / "latent_density.csv", main_dir / "positions_2d.npz", out_dir / "latent_density.png")
    if (main_dir / "stability_calibration.csv").exists() and (main_dir / "stability_by_k.csv").exists():
        draw_stability_vs_simulation(main_dir / "stability_by_k.csv", main_dir / "stability_calibration.csv", out_dir / "stability_vs_simulation.png")
    draw_agreement_by_confidence(kdir, out_dir / "agreement_by_confidence.png")
    draw_checks_dot_plot(kdir, k, out_dir / f"checks_k{k}.png")
    if (kdir / "replication_calibration.csv").exists():
        draw_replication_vs_simulation(kdir / "replication_calibration.csv", out_dir / "replication_vs_simulation.png")

    # ---- 8. tables carried over from the steps ------------------------------------------------------------
    for source in (
        kdir / "method_comparison.csv",
        kdir / "replication_calibration.csv",
        kdir / "shares_by_year.csv",
        kdir / "loadings_2d.csv",
        kdir / "model_comparison.csv",
        main_dir / "stability_by_k.csv",
        main_dir / "k_rule_sensitivity.csv",
        main_dir / "heldout_selection.csv",
        main_dir / "latent_density.csv",
        main_dir / "stability_calibration.csv",
    ):
        if source.exists():
            shutil.copy(source, tables_dir / source.name)
    log(f"wrote figures to {out_dir} and tables to {tables_dir}")


if __name__ == "__main__":
    main()
