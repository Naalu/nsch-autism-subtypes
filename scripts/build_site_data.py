"""Write the data files the project website draws from.

The website (``site/``) shows the analysis as interactive figures: every
child as a dot on the two axes, recolored by group, by the number of groups
asked for, or by condition, and the groups year by year. This script gathers
what those figures need from the result folders and the committed tables
and writes it as a handful of small JSON files. It fits nothing new except
the two simulated clouds the site shows beside the real one, which go
through the same steps as the real children (the latent trait model, then
one draw of each child's position).

What is published about a child, and what is not, is set out at the top of
``src/nsch_subtypes/site_data.py``. In short: position and condition
profile, with the answers of profiles shared by fewer than 10 children
withheld; no identifiers, no weights, no survey year or age; years only as
counts over map cells, and outcomes only as rates by group.

Run it after the pipeline, on the machine that holds the result folders::

    python scripts/build_site_data.py            # or: make site-data

Files, under ``--out`` (default ``site/static/data``)::

    children.json   one entry per child: position (drawn two ways) and profile (shuffled)
    profiles.json   one entry per distinct condition profile: answers (withheld when rare),
                    count, number of conditions, group probabilities, and its group under
                    each K, under k-means and under the two-group factor mixture model
    model.json      what the page needs to place a set of answers: conditions, the four
                    groups' names, rates and shares, the latent trait slopes and
                    intercepts, and each K's groups with the color slot that keeps a
                    group's color as K changes
    clouds.json     2,500 children each from the real data, a simulated continuum and
                    simulated four classes, each drawn two ways, for the side-by-side comparison
    evidence.json   the checks: replication against simulation, stability by K, shape
                    of the cloud, model comparison, factor mixture
    outcomes.json   severity, missed care and emergency visits: rates by group, and what
                    the group label adds to the axes
    years.json      each survey year: children per group in each map cell, and the
                    weighted group shares with intervals
    context.json    the funnel, children per year, co-occurrence
    numbers.json    every number quoted in the site's text, with the table it came from
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from _common import MAIN_YEARS, REPO, load_analysis, load_lca_fit, log, make_parser
from nsch_subtypes.data import to_flag_matrix
from nsch_subtypes.latent_trait import LatentTraitResult, draw_positions, fit_latent_density, fit_latent_trait, simulate_latent_trait
from nsch_subtypes.lca import simulate_classes
from nsch_subtypes.partition import cluster_kmeans
from nsch_subtypes.site_data import (
    DEVELOPMENTAL,
    MIN_PROFILE,
    MIN_YEAR_CELL,
    PSYCHIATRIC,
    assert_no_identifiers,
    axis_alignment,
    binned_counts,
    children_columns,
    coarsened,
    color_slots,
    distinct_profiles,
    order_by_size,
    per_profile,
    profile_strings,
    ranges_by,
    rounded,
    share_with_interval,
)
from nsch_subtypes.validate import adjusted_rand_index

# The four groups in the order the site lists them, and how each is recognized:
# (high on the developmental-language markers, high on the mental-health markers).
# The names describe the pattern of diagnoses, as in the paper; the short names
# are for legends, and the keys for numbers.json.
CLASS_ORDER = {
    "fewer co-occurring diagnoses": (False, False),
    "mostly developmental and language diagnoses": (True, False),
    "mostly mental-health and behavioral diagnoses": (False, True),
    "both kinds of diagnoses": (True, True),
}
SHORT_NAMES = ["fewer diagnoses", "developmental and language", "mental health and behavior", "both"]
KEYS = ["fewer", "developmental", "mental_health", "both"]
OUTCOMES = {
    "severity": ("severity", lambda column: column == "severe", "autism rated severe by the parent"),
    "forgone_care": ("forgone_care", lambda column: column == "yes", "needed health care in the past year and did not get it"),
    "er_visits": ("er_visits", lambda column: column != "none", "one or more emergency room visits in the past year"),
}
CLOUD_SIZE = 2500


def name_four_classes(rates: np.ndarray, flags: list[str]) -> list[str]:
    """Name each of four classes by whether it is high or low on the two sets of marker conditions.

    "High" means above the midpoint between the lowest and highest class on
    that set of markers. The four classes must come out as the four
    combinations; anything else stops the script, because the site's text
    describes exactly those four.
    """
    developmental = rates[:, [flags.index(name) for name in DEVELOPMENTAL]].mean(axis=1)
    psychiatric = rates[:, [flags.index(name) for name in PSYCHIATRIC]].mean(axis=1)
    high_dev = developmental > (developmental.min() + developmental.max()) / 2
    high_psych = psychiatric > (psychiatric.min() + psychiatric.max()) / 2
    by_pattern = {pattern: name for name, pattern in CLASS_ORDER.items()}
    names = [by_pattern[(bool(d), bool(p))] for d, p in zip(high_dev, high_psych, strict=True)]
    if sorted(names) != sorted(CLASS_ORDER):
        raise SystemExit(f"the four classes are not the four high/low combinations of the two axes: {names}")
    return names


def match_labels(labels: np.ndarray, reference: np.ndarray, n_classes: int) -> np.ndarray:
    """Renumber ``labels`` so each cluster takes the number of the reference class it overlaps most."""
    overlap = np.zeros((n_classes, n_classes))
    np.add.at(overlap, (labels, reference), 1)
    rows, columns = linear_sum_assignment(-overlap)
    renumber = np.empty(n_classes, dtype=int)
    renumber[rows] = columns
    return renumber[labels]


def both_draws(trait, X: np.ndarray, flags: list[str], seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Each child's position drawn two ways, on the site's axes, and how lumpy the data are.

    A child's 18 answers only locate it roughly, so where its dot goes
    depends on what is assumed about where children in general sit. The
    first draw assumes the bell curve of the latent trait model; drawn that
    way, any dataset looks like one round cloud, even one simulated from
    separate classes. The second draw uses the distribution of positions
    estimated from the data (``fit_latent_density``), which lets real
    bunching show, and also shows some bunching that is only noise. The site
    offers both and says so. The third value is step 14's summary: how many
    log-likelihood units the estimated distribution beats the bell curve by.
    """
    density = fit_latent_density(trait, X)
    n_points, limit = int(round(np.sqrt(len(density.nodes)))), float(density.nodes.max())
    alignment = axis_alignment(trait.slopes, flags)
    smooth = draw_positions(trait, X, n_draws=1, seed=seed)[:, 0, :] @ alignment
    sharp = draw_positions(trait, X, n_draws=1, seed=seed, n_points=n_points, limit=limit, weights=density.weights)[:, 0, :] @ alignment
    return smooth, sharp, float(density.improvement)


def cloud(X: np.ndarray, flags: list[str], seed: int, size: int, trait=None, smooth: np.ndarray | None = None) -> dict:
    """``size`` children from one dataset, drawn both ways, for the side-by-side comparison.

    The latent trait model is fit to the dataset unless ``trait`` is given.
    ``smooth`` replaces the bell-curve draw (used for the real children, so
    the comparison shows the same positions as the main map).
    """
    trait = trait if trait is not None else fit_latent_trait(X, n_dims=2, seed=seed)
    drawn, sharp, improvement = both_draws(trait, X, flags, seed)
    drawn = drawn if smooth is None else smooth
    keep = np.random.default_rng(seed).choice(len(X), size=min(size, len(X)), replace=False)
    return {
        "x": np.round(drawn[keep, 0], 2).tolist(),
        "y": np.round(drawn[keep, 1], 2).tolist(),
        "sharp_x": np.round(sharp[keep, 0], 2).tolist(),
        "sharp_y": np.round(sharp[keep, 1], 2).tolist(),
        "gain_over_bell_curve": improvement,
    }


def records(path: Path, **read_options) -> list[dict]:
    """A committed table as a list of row dictionaries."""
    return pd.read_csv(path, **read_options).to_dict(orient="records")


def write(out: Path, name: str, payload: dict, digits: int = 4) -> None:
    """Check a payload for identifying fields, round it and write it as compact JSON."""
    assert_no_identifiers(payload)
    path = out / name
    path.write_text(json.dumps(rounded(payload, digits), separators=(",", ":"), allow_nan=False) + "\n")
    log(f"   {name}: {path.stat().st_size / 1024:.0f} KB")


def main() -> None:
    parser = make_parser("Write the JSON files the website draws from", backend=False)
    parser.add_argument("--tables", type=Path, default=REPO / "reports" / "tables", help="the committed summary tables")
    parser.add_argument("--out", type=Path, default=REPO / "site" / "static" / "data", help="where the JSON files go")
    parser.add_argument("--kmeans-starts", type=int, default=200, help="k-means starts per seed, as in step 4")
    parser.add_argument("--kmeans-seeds", type=int, default=5, help="k-means seeds, as in step 4")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tables = args.tables

    table, flags = load_analysis(MAIN_YEARS)
    X = to_flag_matrix(table, flags)
    n = len(table)
    codebook = pd.read_csv(REPO / "data" / "processed" / "codebook.csv").set_index("flag")
    profiles, profile_of_child, counts = distinct_profiles(X)
    log(f"{n} children, {len(profiles)} distinct profiles")

    # ---- positions on the site's axes ------------------------------------------------------------------
    positions_path = args.results / "positions_2d.npz"
    if not positions_path.exists():
        raise SystemExit(f"{positions_path} not found. Run scripts/14_shape_and_stability.py first.")
    saved = np.load(positions_path)
    if not (np.array_equal(saved["hhid"], table["hhid"].to_numpy()) and np.array_equal(saved["year"], table["year"].to_numpy())):
        raise SystemExit(f"{positions_path} was made from a different analysis table. Rerun step 14.")
    alignment = axis_alignment(saved["slopes"], flags)
    positions = saved["positions"] @ alignment
    # The same children drawn from the estimated distribution of positions, with the saved model.
    saved_trait = LatentTraitResult(2, saved["slopes"], saved["intercepts"], np.zeros((n, 2)), 0.0, 0, True)
    _, sharp_positions, real_gain = both_draws(saved_trait, X, flags, args.seed)
    log(f"estimated distribution of positions beats the bell curve by {real_gain:.1f} log-likelihood units")

    # ---- the class models at every K -------------------------------------------------------------------
    fits = {k: load_lca_fit(args.results, k, table, flags) for k in range(2, 9)}
    four = fits[4]
    names = name_four_classes(four.condition_rates, flags)
    to_site_order = np.array([list(CLASS_ORDER).index(name) for name in names])  # class number in the fit -> on the site
    from_site_order = np.argsort(to_site_order)
    labels_by_k = {k: order_by_size(fit.labels, k) for k, fit in fits.items()}
    labels_by_k[4] = to_site_order[four.labels]
    class_probabilities = four.responsibilities[:, from_site_order]

    # k-means at K = 4 with step 4's settings, numbered to match the classes.
    runs = [cluster_kmeans(X, 4, n_starts=args.kmeans_starts, seed=args.seed + i) for i in range(args.kmeans_seeds)]
    kmeans = min(runs, key=lambda run: run.inertia)
    kmeans_labels = match_labels(kmeans.labels, labels_by_k[4], 4)
    kmeans_ari = adjusted_rand_index(labels_by_k[4], kmeans_labels)
    log(f"k-means against the four classes: ARI {kmeans_ari:.3f}; {np.mean(kmeans_labels != labels_by_k[4]):.1%} of children placed differently")

    # The two-group factor mixture model (shared spread), on the site's axes.
    mixture = np.load(args.results / "factor_mixture_fits.npz")
    mixture_alignment = axis_alignment(mixture["slopes_shared_k2"], flags)
    mixture_groups = {
        "shares": mixture["shares_shared_k2"],
        "means": mixture["means_shared_k2"] @ mixture_alignment,
        "covariances": np.einsum("ba,kbc,cd->kad", mixture_alignment, mixture["covariances_shared_k2"], mixture_alignment),
    }

    slots = color_slots(labels_by_k, reference_k=4)
    n_profiles = len(profiles)
    children = children_columns(positions, sharp_positions, profile_of_child, seed=args.seed)
    write(args.out, "children.json", {"n": n, **children}, digits=2)
    write(
        args.out,
        "profiles.json",
        {
            "min_count_for_answers": MIN_PROFILE,
            "answers": profile_strings(profiles, counts),
            "count": counts,
            "n_conditions": profiles.sum(axis=1),
            "class_probabilities": coarsened(per_profile(class_probabilities, profile_of_child, n_profiles), counts),
            "label_by_k": {str(k): per_profile(labels, profile_of_child, n_profiles) for k, labels in labels_by_k.items()},
            "kmeans_label": per_profile(kmeans_labels, profile_of_child, n_profiles),
            "mixture_probability": coarsened(per_profile(mixture["probabilities_shared_k2"][:, 0], profile_of_child, n_profiles), counts),
        },
        digits=3,
    )

    # ---- what the page needs to place a profile the visitor builds -----------------------------------
    class_names = pd.read_csv(tables / "class_names_k4.csv").set_index("name")
    stability = pd.read_csv(tables / "stability_by_k.csv").set_index("k")
    write(
        args.out,
        "model.json",
        {
            "conditions": [{"flag": flag, "label": codebook.loc[flag, "label"], "prevalence": float(X[:, j].mean())} for j, flag in enumerate(flags)],
            "axes": {
                "x": {"name": "developmental-language", "markers": list(DEVELOPMENTAL)},
                "y": {"name": "mental-health", "markers": list(PSYCHIATRIC)},
            },
            "classes": [
                {
                    "name": name,
                    "short_name": SHORT_NAMES[i],
                    "share_of_children": float(four.class_sizes[from_site_order[i]]),
                    "weighted_share": float(class_names.loc[name, "weighted_share"]),
                    "rates": four.condition_rates[from_site_order[i]],
                    "center": positions[labels_by_k[4] == i].mean(axis=0),
                }
                for i, name in enumerate(CLASS_ORDER)
            ],
            "trait": {"slopes": saved["slopes"] @ alignment, "intercepts": saved["intercepts"]},
            "by_k": [
                {
                    "k": k,
                    "color_slot": slots[k],
                    "shares": np.bincount(labels_by_k[k], minlength=k) / n,
                    "centers": [positions[labels_by_k[k] == c].mean(axis=0) for c in range(k)],
                    "least_stable_jaccard": float(stability.loc[k, "min_jaccard"]) if k in stability.index else None,
                    "all_stable": bool(stability.loc[k, "all_stable"]) if k in stability.index else None,
                }
                for k in sorted(fits)
            ],
            "kmeans": {"ari": kmeans_ari, "share_placed_differently": float(np.mean(kmeans_labels != labels_by_k[4]))},
            "mixture": mixture_groups,
        },
    )

    # ---- three clouds for the side-by-side comparison ---------------------------------------------------
    # The simulated datasets are replicate 0 of step 15's: same models, same seeds.
    trait = fit_latent_trait(X, n_dims=2, seed=args.seed)
    write(
        args.out,
        "clouds.json",
        {
            "real": cloud(X, flags, args.seed, CLOUD_SIZE, trait=saved_trait, smooth=positions),
            "continuum": cloud(simulate_latent_trait(trait, n, seed=args.seed + 500), flags, args.seed, CLOUD_SIZE),
            "classes": cloud(simulate_classes(four.class_sizes, four.condition_rates, n, seed=args.seed + 700), flags, args.seed, CLOUD_SIZE),
        },
        digits=2,
    )

    # ---- the checks --------------------------------------------------------------------------------------
    calibration = records(tables / "replication_calibration.csv")
    mixture_calibration = records(tables / "factor_mixture_calibration.csv")
    write(
        args.out,
        "evidence.json",
        {
            "checks_by_k": records(tables / "checks_by_k.csv"),
            "replication": {
                "real": [row for row in calibration if row["data"] == "real"],
                "simulated": [row for row in calibration if row["data"] != "real"],
                "ranges": ranges_by([row for row in calibration if row["data"] != "real"], ("data", "check"), "ari"),
            },
            "stability": {"real": records(tables / "stability_by_k.csv"), "simulated": records(tables / "stability_calibration.csv")},
            "shape": records(tables / "latent_density.csv"),
            "model_comparison": records(tables / "model_comparison.csv"),
            "mixture": {
                "real": records(tables / "factor_mixture.csv"),
                "heldout": records(tables / "factor_mixture_heldout.csv"),
                "calibration": mixture_calibration,
                "ranges": ranges_by([row for row in mixture_calibration if row["data"] != "real"], ("data", "groups"), "log_likelihood_gain"),
            },
            "loadings": records(tables / "loadings_2d.csv"),
        },
    )

    # ---- outcomes ----------------------------------------------------------------------------------------
    outcomes = {}
    for key, (column, test, description) in OUTCOMES.items():
        answered = table[column].notna().to_numpy()
        has = test(table[column]).to_numpy() & answered
        outcomes[key] = {
            "description": description,
            "overall": share_with_interval(has[answered]),
            "by_class": [{"class": name, **share_with_interval(has[answered & (labels_by_k[4] == i)])} for i, name in enumerate(CLASS_ORDER)],
            "models": records(tables / f"outcome_models_{key}.csv"),
        }
    write(args.out, "outcomes.json", {"outcomes": outcomes, "heldout_2025": records(tables / "outcome_heldout_2025.csv")})

    # ---- the groups year by year ------------------------------------------------------------------------
    shares_by_year = pd.read_csv(tables / "shares_by_year.csv")
    # The table numbers classes as the run that made it did; its own names file says which is which.
    shares_by_year["class"] = shares_by_year["class"].map(pd.read_csv(tables / "class_names_k4.csv").set_index("class")["name"])
    years = table["year"].to_numpy()
    write(
        args.out,
        "years.json",
        {
            "cell": 0.75,
            "min_cell": MIN_YEAR_CELL,
            "groups": list(CLASS_ORDER),
            "maps": [
                {"year": int(year), "n": int((years == year).sum()), "cells": binned_counts(positions[years == year], labels_by_k[4][years == year], 4)}
                for year in sorted(set(years.tolist()))
            ],
            "weighted_shares": shares_by_year[["year", "n", "class", "share", "ci_low", "ci_high"]].to_dict(orient="records"),
        },
    )

    # ---- context -----------------------------------------------------------------------------------------
    summary = pd.read_csv(tables / "summary.csv", header=None, index_col=0)[1]
    write(
        args.out,
        "context.json",
        {
            "funnel": summary.to_dict(),
            "children_per_year": table.groupby("year").size().to_dict(),
            "conditions_per_child": {"mean": float(X.sum(axis=1).mean()), "share_with_any": float((X.sum(axis=1) > 0).mean())},
            "cooccurrence": {
                "conditions": pd.read_csv(tables / "cooccurrence.csv", index_col=0).columns.tolist(),
                "matrix": pd.read_csv(tables / "cooccurrence.csv", index_col=0).to_numpy(),
            },
        },
    )

    # ---- every number the site's text quotes -----------------------------------------------------------
    checks = pd.read_csv(tables / "checks_by_k.csv").set_index("K").loc[4]
    real_mixture = pd.read_csv(tables / "factor_mixture.csv")
    two_groups = real_mixture[(real_mixture["version"] == "shared spread") & (real_mixture["groups"] == 2)].iloc[0]
    comparison = pd.read_csv(tables / "model_comparison.csv")
    trait_bic = comparison[comparison["family"] == "latent trait"].set_index("size")["bic"]
    shape = pd.read_csv(tables / "latent_density.csv")
    numbers = {
        "children": (n, "summary.csv"),
        "children_2025": (int(summary["n_2025_complete"]) if "n_2025_complete" in summary else None, "summary.csv"),
        "conditions": (len(flags), "config/conditions.yaml"),
        "distinct_profiles": (len(profiles), "analysis table"),
        "share_with_any_condition": (float((X.sum(axis=1) > 0).mean()), "analysis table"),
        "mean_conditions": (float(X.sum(axis=1).mean()), "analysis table"),
        "ari_kmeans": (float(checks["second method (k-means)"]), "checks_by_k.csv"),
        "ari_heldout_years": (float(checks["held-out years"]), "checks_by_k.csv"),
        "ari_2025": (float(checks["2025 forward test"]), "checks_by_k.csv"),
        "ari_ever_told": (float(checks["'ever told' items"]), "checks_by_k.csv"),
        "ari_age_6_11": (float(checks["age 6-11 refit"]), "checks_by_k.csv"),
        "ari_age_12_17": (float(checks["age 12-17 refit"]), "checks_by_k.csv"),
        "least_stable_jaccard_k4": (float(stability.loc[4, "min_jaccard"]), "stability_by_k.csv"),
        "least_stable_jaccard_k5": (float(stability.loc[5, "min_jaccard"]), "stability_by_k.csv"),
        "mixture_gain_two_groups": (float(two_groups["log_likelihood_gain"]), "factor_mixture.csv"),
        "mixture_bic_two_groups": (float(two_groups["bic_vs_one_group"]), "factor_mixture.csv"),
        "third_axis_bic": (float(trait_bic[3] - trait_bic[2]), "model_comparison.csv"),
        "shape_gain_real": (float(shape.loc[shape["data"] == "real", "improvement"].iloc[0]), "latent_density.csv"),
    }
    for key, name in zip(KEYS, CLASS_ORDER, strict=True):
        numbers[f"weighted_share_{key}"] = (float(class_names.loc[name, "weighted_share"]), "class_names_k4.csv")
    write(args.out, "numbers.json", {key: {"value": value, "source": source} for key, (value, source) in numbers.items() if value is not None})
    log(f"wrote the site data to {args.out}")


if __name__ == "__main__":
    main()
