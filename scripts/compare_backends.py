"""Compare saved library and NumPy LCA fits, without refitting either model.

Writes summary tables, machine-readable comparison data, and three charts::

    python scripts/compare_backends.py --k 4

Inputs default to results/ (reference) and results_ours/ (from scratch).
Figures go to reports/figures/results_ours/; reference figures stay separate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import REPO, load_lca_fit, log, save_json  # noqa: E402
from nsch_subtypes.plots import CYAN, RUST, tidy  # noqa: E402
from nsch_subtypes.validate import adjusted_rand_index, match_classes  # noqa: E402


def compare_fit_pair(reference_dir: Path, ours_dir: Path, k: int) -> tuple[dict, pd.DataFrame]:
    """Match classes and compare fits on exactly the same children and flags."""
    reference = load_lca_fit(reference_dir, k)
    ours = load_lca_fit(ours_dir, k)

    # Scores and class assignments are comparable only when row identities
    # and condition order agree. Refuse a plausible-looking mismatched plot.
    with np.load(reference_dir / f"lca_k{k}.npz") as reference_data, np.load(ours_dir / f"lca_k{k}.npz") as ours_data:
        for field in ("year", "hhid", "flags"):
            if field not in reference_data or field not in ours_data:
                raise ValueError(f"K={k}: both saved fits must include {field}")
            if not np.array_equal(reference_data[field], ours_data[field]):
                raise ValueError(f"K={k}: saved fits have different {field} values or order")
        flags = reference_data["flags"].tolist()

    if reference.responsibilities.shape != ours.responsibilities.shape:
        raise ValueError(f"K={k}: saved fits have different responsibility shapes")
    if not np.isfinite([reference.log_likelihood, ours.log_likelihood]).all():
        raise ValueError(f"K={k}: both saved fits must have finite log-likelihoods")

    # Class numbers are arbitrary. Align profiles before measuring their gaps;
    # ARI itself already ignores arbitrary differences in class numbering.
    permutation = match_classes(reference.condition_rates, ours.condition_rates)
    matched_rates = ours.condition_rates[permutation]
    rate_difference = matched_rates - reference.condition_rates
    summary = {
        "k": k,
        "n_children": len(reference.labels),
        "reference_log_likelihood": reference.log_likelihood,
        "ours_log_likelihood": ours.log_likelihood,
        "log_likelihood_difference_ours_minus_reference": ours.log_likelihood - reference.log_likelihood,
        "reference_bic": float(reference.bic()),
        "ours_bic": float(ours.bic()),
        "bic_difference_ours_minus_reference": float(ours.bic() - reference.bic()),
        "ari": adjusted_rand_index(reference.labels, ours.labels),
        "max_absolute_rate_difference": float(np.abs(rate_difference).max()),
        "mean_absolute_rate_difference": float(np.abs(rate_difference).mean()),
        "reference_converged": reference.converged,
        "ours_converged": ours.converged,
        "reference_iterations": reference.n_iter,
        "ours_iterations": ours.n_iter,
    }

    # Keep the chart values in a long table, with the matching made explicit.
    profile_rows = []
    for reference_class in range(k):
        for condition_index, condition in enumerate(flags):
            profile_rows.append(
                {
                    "k": k,
                    "reference_class": reference_class,
                    "matched_ours_class": int(permutation[reference_class]),
                    "condition": condition,
                    "reference_rate": reference.condition_rates[reference_class, condition_index],
                    "ours_rate": matched_rates[reference_class, condition_index],
                    "difference_percentage_points": 100 * rate_difference[reference_class, condition_index],
                }
            )
    return summary, pd.DataFrame(profile_rows)


def draw_comparison_figures(summary: pd.DataFrame, profiles: pd.DataFrame, k: int, output_dir: Path) -> None:
    """Draw BIC, matched profiles, and signed condition-rate differences."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    axes[0].plot(summary["k"], summary["reference_bic"], "o-", color=CYAN, label="Reference (StepMix)")
    axes[0].plot(summary["k"], summary["ours_bic"], "x--", color=RUST, label="Ours (NumPy)")
    axes[0].set(xlabel="Number of classes K", ylabel="BIC (lower is better)", title="BIC from both implementations")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(summary["k"], summary["bic_difference_ours_minus_reference"], "o-", color=RUST)
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set(xlabel="Number of classes K", ylabel="BIC: ours minus reference", title="Difference at each K")
    for axis in axes:
        axis.set_xticks(summary["k"])
        tidy(axis)
    fig.savefig(output_dir / "backend_bic.png", dpi=170)
    plt.close(fig)

    # Preserve input condition order when converting the long table to grids.
    selected_profiles = profiles[profiles["k"] == k]
    conditions = selected_profiles["condition"].drop_duplicates().tolist()
    grids = {}
    for column in ("reference_rate", "ours_rate", "difference_percentage_points"):
        grids[column] = selected_profiles.pivot(index="reference_class", columns="condition", values=column)[conditions].to_numpy()
    labels = [condition.replace("_", " ") for condition in conditions]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), layout="constrained")
    for axis, column, title in zip(axes, ("reference_rate", "ours_rate"), ("Reference (StepMix)", "Ours (matched to reference)"), strict=True):
        image = axis.imshow(grids[column], vmin=0, vmax=1, cmap="Oranges", aspect="auto")
        axis.set_xticks(range(len(conditions)), labels, rotation=60, ha="right", fontsize=8)
        axis.set_yticks(range(k), [f"Class {index}" for index in range(k)])
        axis.set_title(title)
        for row in range(k):
            for col in range(len(conditions)):
                value = grids[column][row, col]
                axis.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=6, color="white" if value > 0.55 else "black")
    fig.colorbar(image, ax=axes, shrink=0.7, label="Fitted condition probability")
    fig.suptitle(f"Matched class profiles, K={k} (class numbers are arbitrary)")
    fig.savefig(output_dir / "backend_profiles.png", dpi=170)
    plt.close(fig)

    differences = grids["difference_percentage_points"]
    color_limit = max(float(np.abs(differences).max()), 0.1)
    fig, axis = plt.subplots(figsize=(12, 4.5), layout="constrained")
    image = axis.imshow(differences, vmin=-color_limit, vmax=color_limit, cmap="RdBu_r", aspect="auto")
    axis.set_xticks(range(len(conditions)), labels, rotation=60, ha="right", fontsize=8)
    axis.set_yticks(range(k), [f"Class {index}" for index in range(k)])
    axis.set_title(f"Condition-rate differences, K={k}: ours minus reference")
    for row in range(k):
        for col in range(len(conditions)):
            difference = differences[row, col]
            axis.text(col, row, f"{difference:+.2f}", ha="center", va="center", fontsize=7, color="white" if abs(difference) > 0.6 * color_limit else "black")
    fig.colorbar(image, ax=axis, label="Percentage points (signed difference)")
    fig.savefig(output_dir / "backend_rate_differences.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REPO / "results", help="saved StepMix fits")
    parser.add_argument("--ours", type=Path, default=REPO / "results_ours", help="saved NumPy fits")
    parser.add_argument("--reports", type=Path, default=REPO / "reports", help="output root for tables and figures")
    parser.add_argument("--k", type=int, default=4, help="K to show in the profile comparison")
    args = parser.parse_args()

    # Find shared K values, but do not silently omit a missing profile fit.
    reference_ks = {int(path.stem.removeprefix("lca_k")) for path in args.reference.glob("lca_k*.npz")}
    ours_ks = {int(path.stem.removeprefix("lca_k")) for path in args.ours.glob("lca_k*.npz")}
    common_ks = sorted(reference_ks & ours_ks)
    if args.k not in common_ks:
        parser.error(f"K={args.k} must be saved in both input folders; shared K values: {common_ks}")

    settings = {}
    for backend, directory in (("reference", args.reference), ("ours", args.ours)):
        settings[backend] = json.loads((directory / "lca_fits.json").read_text())
        if settings[backend].get("backend") != backend:
            raise ValueError(f"{directory} is not marked as backend {backend!r}")

    comparisons = [compare_fit_pair(args.reference, args.ours, k) for k in common_ks]
    summary = pd.DataFrame([comparison[0] for comparison in comparisons])
    profiles = pd.concat([comparison[1] for comparison in comparisons], ignore_index=True)
    tables_dir = args.reports / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "backend_comparison.csv", index=False)
    profiles.to_csv(tables_dir / "backend_condition_rates.csv", index=False)
    draw_comparison_figures(summary, profiles, args.k, args.reports / "figures" / "results_ours")
    # Save the numbers and settings needed to write a report. Narrative text
    # stays manually maintained; this command never creates or replaces it.
    save_json(
        {
            "reference": str(args.reference.resolve()),
            "ours": str(args.ours.resolve()),
            "settings": settings,
            "profile_k": args.k,
            "comparisons": summary.to_dict(orient="records"),
        },
        args.ours / "backend_comparison.json",
    )
    log(f"Wrote comparison tables to {tables_dir}")
    log(f"Wrote comparison figures to {args.reports / 'figures' / 'results_ours'}")
    log(f"Wrote comparison data to {args.ours / 'backend_comparison.json'}")
    log(summary.to_string(index=False))


if __name__ == "__main__":
    main()
