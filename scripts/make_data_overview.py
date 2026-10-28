"""Figures and small tables that describe the data before any clustering is done.

These feed ``reports/data-overview.md``, the "what does the data look like"
document. Run it after steps 1 and 2. The report text itself is written by
hand from these outputs.

Writes to ``reports/figures/``::

    sample_by_year.png        how many children were selected each year
    prevalence.png            share of children with each condition, pooled vs 2025
    cooccurrence.png          of children with condition A, the share who also have B
    conditions_per_child.png  histogram of how many conditions each child has
    missingness.png           share of children with no answer, per condition

and to ``reports/tables/``::

    cooccurrence.csv, prevalence_by_year.csv, raw_rows_2024.csv,
    processed_rows_2024.csv, top_profiles.csv, summary.csv

Usage::

    python scripts/make_data_overview.py
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import ANALYSIS_PATH, INTERIM_DIR, MAIN_YEARS, PROCESSED_DIR, REPO, kept_flags, log  # noqa: E402
from nsch_subtypes.config import load_config  # noqa: E402

FIGURES_DIR = REPO / "reports" / "figures"
TABLES_DIR = REPO / "reports" / "tables"
RUST, CYAN, INK, GRAY = "#b8582a", "#0e7490", "#1f2330", "#7a8094"


def hide_top_and_right_spines(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()
    flags = kept_flags()
    label_of = {condition.name: condition.label for condition in config.conditions}

    table = pd.read_parquet(ANALYSIS_PATH)
    main_rows = table[table["year"].isin(list(MAIN_YEARS))]
    rows_2025 = table[table["year"] == 2025]
    selection_counts = pd.read_csv(PROCESSED_DIR / "selection_counts.csv")

    # Children with every flag answered, for the figures that need complete rows.
    complete_main = main_rows[flags].dropna()

    # ---- 1. children selected per year -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bar_colors = [RUST if year <= 2024 else CYAN for year in selection_counts["year"]]
    ax.bar(selection_counts["year"], selection_counts["n_selected"], color=bar_colors)
    for year, n_selected, n_total in zip(selection_counts["year"], selection_counts["n_selected"], selection_counts["n_children"], strict=False):
        ax.text(year, n_selected + 25, f"{n_selected:,}\nof {n_total / 1000:.0f}k", ha="center", fontsize=7, color=INK)
    ax.set_ylabel("autistic children aged 6-17")
    ax.set_xticks(selection_counts["year"])
    ax.set_ylim(0, selection_counts["n_selected"].max() * 1.25)
    ax.set_title("Selected children per survey year (rust = pooled analysis, blue = 2025 forward test)", fontsize=9, loc="left")
    hide_top_and_right_spines(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sample_by_year.png", dpi=160)
    plt.close(fig)

    # ---- 2. how common each condition is, pooled vs 2025 -----------------------------------
    prevalence_main = main_rows[flags].mean().sort_values()  # mean of 0/1 = share with the condition
    prevalence_2025 = rows_2025[flags].mean()
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    positions = np.arange(len(flags))
    ax.barh(positions, prevalence_main.values * 100, color=RUST, label="2016-2024 pooled")
    ax.plot(prevalence_2025[prevalence_main.index].values * 100, positions, "o", color=CYAN, label="2025")
    ax.set_yticks(positions)
    ax.set_yticklabels([label_of[flag] for flag in prevalence_main.index], fontsize=8)
    ax.set_xlabel("share of autistic children aged 6-17 with the condition (%)")
    ax.axvline(1, color=GRAY, ls=":", lw=1)
    ax.text(1.2, -0.8, "1% rule", fontsize=7, color=GRAY)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    hide_top_and_right_spines(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "prevalence.png", dpi=160)
    plt.close(fig)

    # ---- 3. co-occurrence: P(column condition | row condition) --------------------------------
    order = list(prevalence_main.index[::-1])  # most common first
    flags_01 = complete_main[order].to_numpy()
    n_with_both = flags_01.T @ flags_01  # [a, b] = children with both a and b
    n_with_row = np.maximum(flags_01.sum(axis=0)[:, None], 1)
    conditional_share = n_with_both / n_with_row
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(conditional_share, cmap="Oranges", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=65, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7.5)
    for row in range(len(order)):
        for col in range(len(order)):
            if row != col:
                text_color = "white" if conditional_share[row, col] > 0.55 else INK
                ax.text(col, row, f"{conditional_share[row, col]:.2f}", ha="center", va="center", fontsize=5.5, color=text_color)
    ax.set_title("Of children with the row condition, the share who also have the column condition", fontsize=9, loc="left")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cooccurrence.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(conditional_share, index=order, columns=order).round(3).to_csv(TABLES_DIR / "cooccurrence.csv")

    # ---- 4. how many conditions each child has --------------------------------------------------
    conditions_per_child = complete_main.sum(axis=1)
    histogram = conditions_per_child.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(histogram.index, histogram.values, color=RUST)
    ax.set_xlabel(f"number of co-occurring conditions (of the {len(flags)} flags)")
    ax.set_ylabel("children")
    ax.set_xticks(range(0, int(histogram.index.max()) + 1))
    n_with_none = int((conditions_per_child == 0).sum())
    ax.set_title(
        f"Mean {conditions_per_child.mean():.2f}, median {int(conditions_per_child.median())}; {n_with_none} children have none",
        fontsize=9,
        loc="left",
    )
    hide_top_and_right_spines(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "conditions_per_child.png", dpi=160)
    plt.close(fig)

    # ---- 5. missing answers per condition ----------------------------------------------------------
    share_missing = main_rows[flags].isna().mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(range(len(share_missing)), share_missing.values, color=GRAY)
    ax.set_yticks(range(len(share_missing)))
    ax.set_yticklabels([label_of[flag] for flag in share_missing.index], fontsize=8)
    ax.set_xlabel("share of selected children with no answer (%)")
    ax.invert_yaxis()
    hide_top_and_right_spines(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "missingness.png", dpi=160)
    plt.close(fig)

    # ---- 6. prevalence by year, as a table ------------------------------------------------------------
    prevalence_by_year = table.groupby("year")[flags].mean().T * 100
    prevalence_by_year.round(1).to_csv(TABLES_DIR / "prevalence_by_year.csv")

    # ---- 7. the same five children, raw and processed --------------------------------------------------
    interim_2024 = INTERIM_DIR / "selected_2024.parquet"
    if not interim_2024.exists():
        sys.exit(f"{interim_2024} not found. This figure needs the raw survey answers; run 'make download' (steps 1 and 2) first.")
    raw_2024 = pd.read_parquet(interim_2024)
    raw_columns_to_show = [
        "hhid",
        "sc_age_years",
        "sc_sex",
        "fwc",
        "stratum",
        "fipsst",
        "k2q35a",
        "k2q35b",
        "k2q35c",
        "k2q31a",
        "k2q31b",
        "k2q33a",
        "k2q33b",
        "k2q42a",
        "k2q42b",
        "k2q61a",
        "downsyn",
    ]
    sample_households = raw_2024["hhid"].iloc[[0, 7, 21, 40, 63]].tolist()
    raw_sample = raw_2024.set_index("hhid").loc[sample_households, [c for c in raw_columns_to_show if c != "hhid"]].reset_index()
    raw_sample.to_csv(TABLES_DIR / "raw_rows_2024.csv", index=False)
    processed_sample = table[(table["year"] == 2024) & table["hhid"].isin(sample_households)].set_index("hhid").loc[sample_households].reset_index()
    processed_sample.to_csv(TABLES_DIR / "processed_rows_2024.csv", index=False)

    # ---- 8. the most common condition profiles ----------------------------------------------------------
    n_unique_profiles = len(complete_main.drop_duplicates())
    top_profiles = complete_main.value_counts().head(10).reset_index()
    top_profiles.columns = flags + ["n_children"]
    top_profiles["profile"] = top_profiles[flags].apply(lambda row: ", ".join(flag for flag in flags if row[flag] == 1) or "(none)", axis=1)
    top_profiles[["profile", "n_children"]].to_csv(TABLES_DIR / "top_profiles.csv", index=False)

    # ---- 9. headline numbers ------------------------------------------------------------------------------
    summary = {
        "n_selected_all_years": int(len(table)),
        "n_main_2016_2024": int(len(main_rows)),
        "n_main_complete": int(len(complete_main)),
        "n_2025": int(len(rows_2025)),
        "n_2025_complete": int(rows_2025[flags].dropna().shape[0]),
        "n_flags": len(flags),
        "n_unique_profiles_main_complete": n_unique_profiles,
        "mean_conditions": float(conditions_per_child.mean()),
        "share_zero_conditions": float((conditions_per_child == 0).mean()),
        "severity_missing_main": int(main_rows["severity"].isna().sum()),
    }
    pd.Series(summary).to_csv(TABLES_DIR / "summary.csv", header=False)
    log("\n".join(f"{key}: {value}" for key, value in summary.items()))
    log(f"wrote figures to {FIGURES_DIR} and tables to {TABLES_DIR}")


if __name__ == "__main__":
    main()
