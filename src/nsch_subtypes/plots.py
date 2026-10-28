"""The figures the pipeline writes. Matplotlib only, saved as PNG.

Each function takes already-computed numbers and a path, draws one figure,
saves it, and closes it. No analysis happens here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# "Agg" draws to files without needing a display, so this works on a server or in CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Colors used throughout, so the figures look like one set. Every figure
# script imports these instead of defining its own.
RUST = "#b8582a"
PURPLE = "#6b46c1"
CYAN = "#0e7490"
GREEN = "#2f855a"
INK = "#1f2330"
GRAY = "#7a8094"
CLASS_COLORS = ["#b8582a", "#0e7490", "#6b46c1", "#2f855a", "#b7791f", "#9b2c2c", "#2b6cb0", "#4a5568"]


def tidy(ax: plt.Axes) -> None:
    """Hide the top and right frame lines, the house style for every figure."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def profile_heatmap(profiles: pd.DataFrame, flags: list[str], path: Path, title: str = "") -> None:
    """A grid of classes (rows) by conditions (columns), each cell colored and labeled by the share of the class with that condition.

    ``profiles`` is the table from :func:`nsch_subtypes.describe.class_profiles`.
    """
    rates = profiles[flags].to_numpy()
    n_classes = rates.shape[0]

    fig, ax = plt.subplots(figsize=(0.55 * len(flags) + 2, 0.5 * n_classes + 1.6))
    image = ax.imshow(rates, cmap="Oranges", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(flags)))
    ax.set_xticklabels(flags, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels([f"class {k}  (n={n})" for k, n in zip(profiles["class"], profiles["n"], strict=False)], fontsize=8)

    # Print the number in every cell; use white text on dark cells.
    for row in range(n_classes):
        for col in range(len(flags)):
            text_color = "white" if rates[row, col] > 0.55 else INK
            ax.text(col, row, f"{rates[row, col]:.2f}", ha="center", va="center", fontsize=6.5, color=text_color)

    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="share of class with the condition")
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def bic_curve(ks: list[int], bic: list[float], path: Path, null_bic: list[float] | None = None, chosen_k: int | None = None) -> None:
    """BIC against the number of classes K, with the shuffled-data curve if given, and ``chosen_k`` shaded if given."""
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(ks, bic, "o-", color=RUST, label="observed data")
    if null_bic is not None:
        ax.plot(ks, null_bic, "s--", color=GRAY, label="columns shuffled (no structure)")

    if chosen_k is not None:
        ax.axvline(chosen_k, color=RUST, alpha=0.3, lw=6)

    ax.set_xlabel("number of classes K")
    ax.set_ylabel("BIC (lower is better)")
    ax.set_xticks(ks)
    ax.legend(frameon=False, fontsize=8)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def dendrogram_plot(merges: np.ndarray, n_clusters: int, path: Path, title: str = "") -> None:
    """The hierarchical clustering tree, truncated to its last 40 merges, with the cut that gives ``n_clusters`` drawn as a dashed line."""
    from scipy.cluster.hierarchy import dendrogram

    fig, ax = plt.subplots(figsize=(7, 3.6))
    dendrogram(merges, truncate_mode="lastp", p=40, no_labels=True, color_threshold=0, above_threshold_color=GRAY, ax=ax)

    # The cut height that leaves n_clusters groups is the (n_clusters - 1)th largest merge distance.
    merge_distances = merges[:, 2]
    cut_height = np.sort(merge_distances)[-(n_clusters - 1)] if n_clusters > 1 else merge_distances.max()
    ax.axhline(cut_height, color=PURPLE, ls="--", lw=1)

    ax.set_ylabel("Jaccard distance at merge")
    ax.set_xlabel("children (tree truncated to the last 40 merges)")
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def stability_plot(mean_jaccard: np.ndarray, path: Path) -> None:
    """One bar per class showing its mean bootstrap Jaccard, with the 0.75 "stable" line and the 0.5 "dissolved" line."""
    n_classes = len(mean_jaccard)
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.bar(range(n_classes), mean_jaccard, color=RUST)

    ax.axhline(0.75, color=INK, ls="--", lw=1)
    ax.text(n_classes - 0.5, 0.76, "stable at 0.75", ha="right", fontsize=8)
    ax.axhline(0.5, color=GRAY, ls=":", lw=1)

    ax.set_ylim(0, 1)
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels([f"class {k}" for k in range(n_classes)])
    ax.set_ylabel("mean bootstrap Jaccard")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
