"""Figures and tables for reports/data-overview.md.

Run after steps 1 and 2. Writes PNGs to reports/figures/ and small tables to
reports/tables/. The report itself is written by hand from these outputs.

    python scripts/make_data_overview.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import ANALYSIS, CODEBOOK, INTERIM, MAIN_YEARS, PROCESSED, REPO, kept_flags, log  # noqa: E402
from nsch_subtypes.config import load_config  # noqa: E402

FIG = REPO / "reports" / "figures"
TAB = REPO / "reports" / "tables"
RUST, CYAN, INK, MUTE = "#b8582a", "#0e7490", "#1f2330", "#7a8094"


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    config = load_config()
    flags = kept_flags()
    table = pd.read_parquet(ANALYSIS)
    main_rows = table[table["year"].isin(list(MAIN_YEARS))]
    counts = pd.read_csv(PROCESSED / "selection_counts.csv")
    labels = {c.name: c.label for c in config.conditions}

    # 1. sample by year -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 3.2))
    cols = [RUST if y <= 2024 else CYAN for y in counts["year"]]
    ax.bar(counts["year"], counts["n_selected"], color=cols)
    for y, n, tot in zip(counts["year"], counts["n_selected"], counts["n_children"], strict=False):
        ax.text(y, n + 25, f"{n:,}\nof {tot / 1000:.0f}k", ha="center", fontsize=7, color=INK)
    ax.set_ylabel("autistic children aged 6-17")
    ax.set_xticks(counts["year"])
    ax.set_ylim(0, counts["n_selected"].max() * 1.25)
    ax.set_title("Selected children per survey year (rust = pooled analysis, blue = 2025 forward test)", fontsize=9, loc="left")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(FIG / "sample_by_year.png", dpi=160)
    plt.close(fig)

    # 2. prevalence per flag, pooled vs 2025 ----------------------------------
    prev_main = main_rows[flags].mean().sort_values()
    prev_2025 = table.loc[table["year"] == 2025, flags].mean()
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ypos = np.arange(len(flags))
    ax.barh(ypos, prev_main.values * 100, color=RUST, label="2016-2024 pooled")
    ax.plot(prev_2025[prev_main.index].values * 100, ypos, "o", color=CYAN, label="2025")
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels[f] for f in prev_main.index], fontsize=8)
    ax.set_xlabel("share of autistic children aged 6-17 with the condition (%)")
    ax.axvline(1, color=MUTE, ls=":", lw=1)
    ax.text(1.2, -0.8, "1% rule", fontsize=7, color=MUTE)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(FIG / "prevalence.png", dpi=160)
    plt.close(fig)

    # 3. co-occurrence: P(column | row) ---------------------------------------
    C = main_rows[flags].dropna()
    order = list(prev_main.index[::-1])
    M = C[order].to_numpy()
    both = M.T @ M
    cond = both / np.maximum(M.sum(axis=0)[:, None], 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cond, cmap="Oranges", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=65, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7.5)
    for i in range(len(order)):
        for j in range(len(order)):
            if i != j:
                ax.text(j, i, f"{cond[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if cond[i, j] > 0.55 else INK)
    ax.set_title("Of children with the row condition, the share who also have the column condition", fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIG / "cooccurrence.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(cond, index=order, columns=order).round(3).to_csv(TAB / "cooccurrence.csv")

    # 4. conditions per child --------------------------------------------------
    n_cond = C.sum(axis=1)
    fig, ax = plt.subplots(figsize=(6, 3))
    vc = n_cond.value_counts().sort_index()
    ax.bar(vc.index, vc.values, color=RUST)
    ax.set_xlabel("number of co-occurring conditions (of the 18 flags)")
    ax.set_ylabel("children")
    ax.set_xticks(range(0, int(vc.index.max()) + 1))
    ax.set_title(f"Mean {n_cond.mean():.2f}, median {int(n_cond.median())}; {int((n_cond == 0).sum())} children have none", fontsize=9, loc="left")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(FIG / "conditions_per_child.png", dpi=160)
    plt.close(fig)

    # 5. missingness ------------------------------------------------------------
    miss = main_rows[flags].isna().mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(range(len(miss)), miss.values, color=MUTE)
    ax.set_yticks(range(len(miss)))
    ax.set_yticklabels([labels[f] for f in miss.index], fontsize=8)
    ax.set_xlabel("share of selected children with no answer (%)")
    ax.invert_yaxis()
    tidy(ax)
    fig.tight_layout()
    fig.savefig(FIG / "missingness.png", dpi=160)
    plt.close(fig)

    # 6. prevalence by year table -------------------------------------------------
    by_year = table.groupby("year")[flags].mean().T * 100
    by_year.round(1).to_csv(TAB / "prevalence_by_year.csv")

    # 7. raw vs processed rows for the same children -----------------------------
    raw = pd.read_parquet(INTERIM / "selected_2024.parquet")
    raw_cols = [
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
    sample_ids = raw["hhid"].iloc[[0, 7, 21, 40, 63]].tolist()
    raw_sample = raw.set_index("hhid").loc[sample_ids, [c for c in raw_cols if c != "hhid"]].reset_index()
    raw_sample.to_csv(TAB / "raw_rows_2024.csv", index=False)
    proc_sample = table[(table["year"] == 2024) & table["hhid"].isin(sample_ids)].set_index("hhid").loc[sample_ids].reset_index()
    proc_sample.to_csv(TAB / "processed_rows_2024.csv", index=False)

    # 8. unique profiles ------------------------------------------------------------
    n_unique = len(C.drop_duplicates())
    top = C.value_counts().head(10).reset_index()
    top.columns = flags + ["n_children"]
    top["profile"] = top[flags].apply(lambda r: ", ".join(f for f in flags if r[f] == 1) or "(none)", axis=1)
    top[["profile", "n_children"]].to_csv(TAB / "top_profiles.csv", index=False)

    summary = {
        "n_selected_all_years": int(len(table)),
        "n_main_2016_2024": int(len(main_rows)),
        "n_main_complete": int(len(C)),
        "n_2025": int((table["year"] == 2025).sum()),
        "n_2025_complete": int(table[table["year"] == 2025][flags].dropna().shape[0]),
        "n_flags": len(flags),
        "n_unique_profiles_main_complete": n_unique,
        "mean_conditions": float(n_cond.mean()),
        "share_zero_conditions": float((n_cond == 0).mean()),
        "severity_missing_main": int(main_rows["severity"].isna().sum()),
    }
    pd.Series(summary).to_csv(TAB / "summary.csv", header=False)
    log("\n".join(f"{k}: {v}" for k, v in summary.items()))
    log(f"wrote figures to {FIG} and tables to {TAB}")
    _ = CODEBOOK


if __name__ == "__main__":
    main()
