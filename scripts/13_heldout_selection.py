"""Step 13 of 14. How much structure is there, and which K do we report?

Why this step exists
--------------------
BIC kept favoring more classes all the way to K=8, and also kept favoring
more dimensions in the continuous model, so it cannot settle either question
on data this large. Our first fallback was a rule picked after looking at
the checks ("the largest K at which every class is stable and replicates"),
which is circular: the checks then also served as the evidence.

This step separates two questions that are easy to blur.

How much structure is there, and which kind of model describes it better?
Answered by held-out likelihood: fit each candidate on one set of children
and measure how probable it finds a different set. Too little structure
misses real patterns; too much fits noise in the training children. No
parameter counting is involved, and the comparison works across model
families because both give a likelihood on the same scale. On these data
held-out likelihood keeps improving up to the largest K and D tried, and the
continuous models reach a given level with far fewer parameters, so it says
"more structure than a few groups capture", not "this K".

Which K should we report? Answered by stability, an established way to
choose the number of clusters (Ben-Hur et al. 2002; Lange et al. 2004): the
largest K at which every class comes back under resampling, mean bootstrap
Jaccard of at least 0.75 (Hennig 2007, step 6). This rule uses only the
bootstrap, so the other checks (replication across years, the 2025 file,
age bands, the "ever told" items, the second method) remain independent
tests of the K it picks.

Stability says how finely the children can be divided and get the same
division back. It does not say the pieces are real groups: data simulated
from a smooth continuum with no groups also give stable classes up to a
point (step 14 measures where). So K from this rule is "the finest
reproducible partition", and whether that partition reflects groups or cuts
through a continuum is a separate question for steps 12 and 14.

What happens here
-----------------
1. Held-out likelihood: fit latent class models (K = 1 to 8) and latent
   trait models (D = 1 to 4) on the 2016 to 2021 children and score the
   2022 to 2024 children; then refit on all 2016 to 2024 children and score
   the 2025 children. Differences come with a paired standard error from
   the spread of per-child differences. Trait likelihoods are computed on a
   finer quadrature grid than the one used for fitting.
2. Stability: read ``stability.csv`` from the results folder and from each
   sibling ``<results>_k{K}/`` folder (``results_k4/`` and so on for the
   default ``results/``), and apply the rule.

Run the held-out part once (about 15 minutes); ``--reuse-heldout`` skips it
and only re-applies the stability rule, for example after a new bootstrap.

Outputs, under ``--results``::

    heldout_selection.csv            every model, both held-out sets
    stability_by_k.csv               least and mean class Jaccard at each K with a bootstrap
    model_choice.json                chosen K, the rule, and the held-out summary
    figures/heldout_selection.png

Usage::

    python scripts/13_heldout_selection.py --backend reference
"""

from __future__ import annotations

import dataclasses

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import backend_or_exit, load_analysis, log, make_parser, save_json  # noqa: E402
from nsch_subtypes.data import to_flag_matrix  # noqa: E402
from nsch_subtypes.latent_trait import fit_latent_trait, latent_trait_log_likelihood  # noqa: E402
from nsch_subtypes.plots import INK, PURPLE, RUST, tidy  # noqa: E402
from nsch_subtypes.validate import JACCARD_STABLE, class_model_log_likelihood  # noqa: E402

SPLITS = {
    "2022_24": (range(2016, 2022), range(2022, 2025)),
    "2025": (range(2016, 2025), [2025]),
}
CHECK_GRID = {1: 41, 2: 31, 3: 19, 4: 11}  # quadrature points per dimension used for scoring


def score_split(fit_lca, train: np.ndarray, test: np.ndarray, k_max: int, d_max: int, n_starts: int, seed: int) -> list[dict]:
    """Fit every candidate on ``train`` and return per-child held-out log-likelihoods on ``test``."""
    rows = []
    for k in range(1, k_max + 1):
        fit = fit_lca(train, k, n_starts=n_starts, seed=seed)
        per_child = class_model_log_likelihood(test, fit.class_sizes, fit.condition_rates, per_child=True)
        rows.append({"family": "latent class", "size": k, "n_parameters": fit.n_parameters, "train_ll": fit.log_likelihood, "per_child": per_child})
        log(f"   latent class K={k}: held-out log-likelihood {per_child.sum():.1f}")
    for d in range(1, d_max + 1):
        fit = fit_latent_trait(train, n_dims=d, seed=seed)
        checked = dataclasses.replace(fit, n_quadrature=CHECK_GRID.get(d, 11))
        per_child = latent_trait_log_likelihood(checked, test, per_child=True)
        rows.append(
            {
                "family": "latent trait",
                "size": d,
                "n_parameters": fit.n_parameters,
                "train_ll": latent_trait_log_likelihood(checked, train),
                "per_child": per_child,
                "converged": fit.converged,
            }
        )
        log(f"   latent trait D={d}: held-out log-likelihood {per_child.sum():.1f} (converged={fit.converged})")
    return rows


def summarize(rows: list[dict], split: str) -> pd.DataFrame:
    """Totals, per-child means, and the paired difference (with SE) from the best model in the split."""
    best = max(rows, key=lambda r: r["per_child"].sum())
    out = []
    for r in rows:
        diff = r["per_child"] - best["per_child"]
        out.append(
            {
                "split": split,
                "family": r["family"],
                "size": r["size"],
                "n_parameters": r["n_parameters"],
                "train_ll": r["train_ll"],
                "heldout_ll": r["per_child"].sum(),
                "heldout_ll_per_child": r["per_child"].mean(),
                "diff_from_best": diff.sum(),
                "diff_se": diff.std(ddof=1) * np.sqrt(len(diff)) if r is not best else 0.0,
                "converged": r.get("converged", True),
            }
        )
    return pd.DataFrame(out)


def stability_table(results_dir) -> pd.DataFrame:
    """Least and mean class Jaccard for every K that has a bootstrap on file."""
    rows = []
    for folder in [results_dir, *sorted(results_dir.parent.glob(f"{results_dir.name}_k*"))]:
        path = folder / "stability.csv"
        if not path.exists():
            continue
        stab = pd.read_csv(path)
        rows.append(
            {
                "k": len(stab),
                "folder": folder.name,
                "min_jaccard": stab["mean_jaccard"].min(),
                "mean_jaccard": stab["mean_jaccard"].mean(),
                "all_stable": bool((stab["mean_jaccard"] >= JACCARD_STABLE).all()),
            }
        )
    return pd.DataFrame(rows).drop_duplicates("k").sort_values("k").reset_index(drop=True)


def heldout_summary(table: pd.DataFrame) -> dict:
    out = {}
    for split in SPLITS:
        t = table[table["split"] == split]
        best = t.loc[t["heldout_ll"].idxmax()]
        lca = t[t["family"] == "latent class"]
        trait = t[t["family"] == "latent trait"]
        out[split] = {
            "best_overall": f"{best['family']}, size {int(best['size'])}",
            "best_k_by_heldout": int(lca.loc[lca["heldout_ll"].idxmax(), "size"]),
            "best_k_is_largest_tried": bool(lca.loc[lca["heldout_ll"].idxmax(), "size"] == lca["size"].max()),
            "best_d_by_heldout": int(trait.loc[trait["heldout_ll"].idxmax(), "size"]),
            "best_class_minus_best_trait": float(lca["heldout_ll"].max() - trait["heldout_ll"].max()),
        }
    return out


def plot_heldout(table: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, split in zip(axes, SPLITS, strict=True):
        t = table[table["split"] == split]
        for family, color, marker in (("latent class", RUST, "o-"), ("latent trait", PURPLE, "s-")):
            f = t[t["family"] == family]
            ax.errorbar(f["n_parameters"], f["diff_from_best"], yerr=1.96 * f["diff_se"], fmt=marker, color=color, label=family, capsize=2)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xlabel("number of parameters")
        ax.set_ylabel("held-out log-likelihood minus best")
        ax.set_title(f"held-out children: {'2022 to 2024' if split == '2022_24' else '2025'}", fontsize=9, loc="left")
        ax.set_ylim(-1200, 60)
        ax.legend(frameon=False, fontsize=8)
        tidy(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = make_parser("Held-out likelihood across both model families, and the stability rule for K")
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--d-max", type=int, default=4)
    parser.add_argument("--n-starts", type=int, default=20)
    parser.add_argument("--reuse-heldout", action="store_true", help="skip the held-out fits and reuse heldout_selection.csv")
    args = parser.parse_args()

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "figures").mkdir(exist_ok=True)

    # ---- 1. held-out likelihood -----------------------------------------------------------
    heldout_path = args.results / "heldout_selection.csv"
    if args.reuse_heldout and heldout_path.exists():
        table = pd.read_csv(heldout_path)
    else:
        fit_lca = backend_or_exit(args.backend, "lca")["lca"]
        tables = []
        for split, (train_years, test_years) in SPLITS.items():
            train_table, flags = load_analysis(train_years)
            test_table, _ = load_analysis(test_years)
            train, test = to_flag_matrix(train_table, flags), to_flag_matrix(test_table, flags)
            log(f"split {split}: fit on {train.shape[0]} children, score {test.shape[0]}")
            tables.append(summarize(score_split(fit_lca, train, test, args.k_max, args.d_max, args.n_starts, args.seed), split))
        table = pd.concat(tables, ignore_index=True)
        table.to_csv(heldout_path, index=False)
    plot_heldout(table, args.results / "figures" / "heldout_selection.png")
    summary = heldout_summary(table)
    log(table.drop(columns=["train_ll"]).round(1).to_string(index=False))

    # ---- 2. stability rule for K ------------------------------------------------------------
    stability = stability_table(args.results)
    stability.to_csv(args.results / "stability_by_k.csv", index=False)
    stable = stability[stability["all_stable"]]
    if stable.empty:
        raise SystemExit("No K has every class stable; run the bootstrap (scripts/06_bootstrap.py) at more K values.")
    chosen_k = int(stable["k"].max())
    untested = sorted(set(range(chosen_k + 1, args.k_max + 1)) - set(stability["k"]))
    log(stability.round(3).to_string(index=False))
    if untested:
        log(f"warning: no bootstrap on file for K = {untested}; the rule can only be applied to the K that were tested")

    save_json(
        {
            "chosen_k": chosen_k,
            "rule": f"largest K at which every class has mean bootstrap Jaccard >= {JACCARD_STABLE}",
            "k_with_bootstrap": stability["k"].tolist(),
            "k_untested_above_choice": untested,
            "heldout": summary,
            "backend": args.backend,
        },
        args.results / "model_choice.json",
    )
    log(f"chosen K = {chosen_k} ({len(stability)} values of K have a bootstrap on file)")
    for split, s in summary.items():
        log(f"held out {split}: best overall {s['best_overall']}; within classes K={s['best_k_by_heldout']} (largest tried: {s['best_k_is_largest_tried']})")


if __name__ == "__main__":
    main()
