"""Figures shared by the pipeline scripts. Matplotlib only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUST = "#b8582a"
PURPLE = "#6b46c1"
INK = "#1f2330"


def profile_heatmap(profiles: pd.DataFrame, flags: list[str], path: Path, title: str = "") -> None:
    """Classes as rows, flags as columns, cells = rate of the flag in the class."""
    data = profiles[flags].to_numpy()
    k = data.shape[0]
    fig, ax = plt.subplots(figsize=(0.55 * len(flags) + 2, 0.5 * k + 1.6))
    im = ax.imshow(data, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(flags)))
    ax.set_xticklabels(flags, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"class {c}  (n={n})" for c, n in zip(profiles["class"], profiles["n"], strict=False)], fontsize=8)
    for i in range(k):
        for j in range(len(flags)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if data[i, j] > 0.55 else INK)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="share of class with the condition")
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def bic_curve(ks: list[int], bic: list[float], path: Path, null_bic: list[float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(ks, bic, "o-", color=RUST, label="observed data")
    if null_bic is not None:
        ax.plot(ks, null_bic, "s--", color="#7a8094", label="columns shuffled (no structure)")
    best = int(np.argmin(bic))
    ax.axvline(ks[best], color=RUST, alpha=0.3, lw=6)
    ax.set_xlabel("number of classes K")
    ax.set_ylabel("BIC (lower is better)")
    ax.set_xticks(ks)
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def dendrogram_plot(merges: np.ndarray, n_clusters: int, path: Path, title: str = "") -> None:
    from scipy.cluster.hierarchy import dendrogram

    fig, ax = plt.subplots(figsize=(7, 3.6))
    dendrogram(merges, truncate_mode="lastp", p=40, no_labels=True, color_threshold=0, above_threshold_color="#7a8094", ax=ax)
    cut = np.sort(merges[:, 2])[-(n_clusters - 1)] if n_clusters > 1 else merges[:, 2].max()
    ax.axhline(cut, color=PURPLE, ls="--", lw=1)
    ax.set_ylabel("Jaccard distance at merge")
    ax.set_xlabel("children (tree truncated to the last 40 merges)")
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def stability_plot(mean_jaccard: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.bar(range(len(mean_jaccard)), mean_jaccard, color=RUST)
    ax.axhline(0.75, color=INK, ls="--", lw=1)
    ax.text(len(mean_jaccard) - 0.5, 0.76, "stable at 0.75", ha="right", fontsize=8)
    ax.axhline(0.5, color="#7a8094", ls=":", lw=1)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(mean_jaccard)))
    ax.set_xticklabels([f"class {k}" for k in range(len(mean_jaccard))])
    ax.set_ylabel("mean bootstrap Jaccard")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
