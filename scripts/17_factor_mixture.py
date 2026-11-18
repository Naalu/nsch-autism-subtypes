"""Step 17 of 17. Once the two axes are in the model, do groups still help?

Why this step exists
--------------------
Steps 12 to 15 compared two extremes: a model that is nothing but groups
(latent classes) and a model with no groups at all (the latent trait model,
children spread along two axes). The data came out closer to the second,
but not exactly on it: step 14 found the children somewhat lumpier than a
smooth bell curve. A factor mixture model sits between the extremes. It
keeps the two axes and lets children bunch around K places on them, so it
can test the remaining question directly. With one group it is the latent
trait model; if more groups do not improve it, the axes are the whole
story.

Two versions are fit (see ``src/nsch_subtypes/factor_mixture.py``). In the
main one all groups share one spread, so they differ only in where they
sit. In the sensitivity check each group has its own spread, which gives
groups a second way to show up.

A mixture can also "find" groups in data that has none, because a few
overlapping bell curves describe any lopsided cloud better than one bell
curve does (Bauer and Curran 2003). So, as in steps 14 and 15, the same
models are fit to simulated data where the truth is known.

What happens here
-----------------
1. Fit the two-dimensional latent trait model to the 2016 to 2024 children,
   then factor mixture models with 1 to ``--k-max`` groups (shared spread)
   and 2 to ``--k-max-separate`` groups (separate spreads). Compare by BIC.
2. Held-out years, with the splits of step 13: fit on 2016 to 2021 and
   score 2022 to 2024; fit on 2016 to 2024 and score 2025. Each model's
   held-out log-likelihood is compared with the one-group model's, child by
   child, which gives a standard error for the difference.
3. Known truth: ``--replicates`` datasets simulated from the latent trait
   model ("continuum", no groups) and from the latent class model at the
   chosen K ("classes", nothing but groups), the same datasets step 15
   uses. Each gets the shared-spread models with 1 to ``--k-max`` groups.
   The question for the real data is which of the two it resembles.

Outputs, under ``--results``::

    factor_mixture.csv               one row per model fit to the real data
    factor_mixture_heldout.csv       held-out log-likelihood against one group, per split
    factor_mixture_calibration.csv   improvement from adding groups: real and simulated data
    factor_mixture_fits.npz          group shares, means, covariances and child probabilities per model
    factor_mixture.json              summary and settings
    factor_mixture_work/             each dataset's fits, kept so an interrupted run can resume

Usage::

    python scripts/17_factor_mixture.py
    python scripts/17_factor_mixture.py --jobs 4      # fit several models at once
"""

from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from _common import MAIN_YEARS, load_analysis, load_lca_fit, log, make_parser, resolve_k, save_json
from nsch_subtypes.data import to_flag_matrix
from nsch_subtypes.factor_mixture import FactorMixtureResult, factor_mixture_log_likelihood, fit_factor_mixture
from nsch_subtypes.latent_trait import fit_latent_trait, simulate_latent_trait
from nsch_subtypes.lca import simulate_classes

# Train and test years for the held-out comparison, as in step 13.
SPLITS = {
    "2022_24": (range(2016, 2022), range(2022, 2025)),
    "2025": (range(2016, 2025), [2025]),
}
CONFIDENT = 0.9  # a child is "confidently placed" if one group has at least this probability


def version_name(shared_spread: bool) -> str:
    return "shared spread" if shared_spread else "separate spreads"


def fit_models(task: dict) -> dict:
    """Fit the latent trait model and every requested factor mixture model to one dataset.

    ``task`` holds the dataset (``X``), the models to fit (``models``: pairs
    of group count and shared-spread flag), the fitting settings, and
    optionally ``X_test`` to score. Kept as one function taking one argument
    so several datasets can be fit at once in separate processes.
    """
    X, settings = task["X"], task["settings"]
    trait = fit_latent_trait(X, n_dims=2, seed=settings["seed"])
    fits = {}
    for n_groups, shared_spread in task["models"]:
        fits[(n_groups, shared_spread)] = fit_factor_mixture(
            X,
            n_groups,
            shared_spread=shared_spread,
            n_starts=settings["n_starts"],
            short_iter=settings["short_iter"],
            max_iter=settings["max_iter"],
            seed=settings["seed"],
            trait=trait,
        )
    heldout = {}
    if task.get("X_test") is not None:
        heldout = {key: factor_mixture_log_likelihood(fit, task["X_test"], per_child=True) for key, fit in fits.items()}
    return {"label": task["label"], "fits": fits, "heldout": heldout}


def describe_fit(fit: FactorMixtureResult, one_group: FactorMixtureResult) -> dict:
    """One table row for a fitted model, with its improvement over the one-group model fit to the same data."""
    return {
        "version": version_name(fit.shared_spread),
        "groups": fit.n_groups,
        "n_parameters": fit.n_parameters,
        "log_likelihood": fit.log_likelihood,
        "bic": fit.bic(),
        "log_likelihood_gain": fit.log_likelihood - one_group.log_likelihood,
        "bic_vs_one_group": fit.bic() - one_group.bic(),  # negative: adding groups lowers BIC
        "converged": fit.converged,
        "at_bound": fit.at_bound,
        "group_shares": np.round(fit.group_shares, 3).tolist(),
        "group_means": np.round(fit.group_means, 2).tolist(),
        "smallest_group_sd": fit.smallest_group_sd,
        "confidently_placed": float((fit.group_probabilities.max(axis=1) >= CONFIDENT).mean()),
    }


def main() -> None:
    parser = make_parser("Factor mixture models: groups on top of the two axes, on real and simulated data", backend=False)
    parser.add_argument("--k", type=int, default=None, help="class model used to simulate 'classes' data (default: step 13's choice)")
    parser.add_argument("--k-max", type=int, default=4, help="largest number of groups, shared-spread version")
    parser.add_argument("--k-max-separate", type=int, default=3, help="largest number of groups, separate-spreads version")
    parser.add_argument("--replicates", type=int, default=5, help="simulated datasets per kind")
    parser.add_argument("--n-starts", type=int, default=10, help="starting points per model")
    parser.add_argument("--short-iter", type=int, default=200, help="EM iterations per starting point before the best are continued")
    parser.add_argument("--max-iter", type=int, default=5000, help="cap on further EM iterations for the continued runs")
    parser.add_argument("--jobs", type=int, default=1, help="datasets to fit at the same time")
    args = parser.parse_args()

    k_classes = resolve_k(args)
    table, flags = load_analysis(MAIN_YEARS)
    table_2025, _ = load_analysis([2025])
    X, X_2025 = to_flag_matrix(table, flags), to_flag_matrix(table_2025, flags)
    class_model = load_lca_fit(args.results, k_classes, table, flags)
    settings = {"n_starts": args.n_starts, "short_iter": args.short_iter, "max_iter": args.max_iter, "seed": args.seed}
    shared = [(n_groups, True) for n_groups in range(1, args.k_max + 1)]
    separate = [(n_groups, False) for n_groups in range(2, args.k_max_separate + 1)]
    log(f"{X.shape[0]} children 2016-2024 and {X_2025.shape[0]} in 2025; 'classes' data from the K={k_classes} class model")

    # Every dataset that needs fitting, as one list, so they can run side by side.
    tasks = [{"label": ("real", 0), "X": X, "models": shared + separate, "settings": settings}]
    years = table["year"].to_numpy()
    for split, (train_years, test_years) in SPLITS.items():
        if split == "2025":
            train, test = X, X_2025
        else:
            train, test = X[np.isin(years, list(train_years))], X[np.isin(years, list(test_years))]
        tasks.append({"label": ("heldout", split), "X": train, "X_test": test, "models": shared + separate, "settings": settings})
    trait = fit_latent_trait(X, n_dims=2, seed=args.seed)
    for r in range(args.replicates):  # the same simulated datasets as step 15
        continuum = simulate_latent_trait(trait, X.shape[0], seed=args.seed + 500 + r)
        classes = simulate_classes(class_model.class_sizes, class_model.condition_rates, X.shape[0], seed=args.seed + 700 + r)
        tasks.append({"label": ("continuum", r), "X": continuum, "models": shared, "settings": settings})
        tasks.append({"label": (f"{k_classes} classes", r), "X": classes, "models": shared, "settings": settings})

    # Each dataset's fits are saved as they finish, so an interrupted run picks up where it stopped.
    work = args.results / "factor_mixture_work"
    work.mkdir(parents=True, exist_ok=True)
    stamp = {**settings, "models": {str(task["label"]): task["models"] for task in tasks}, "replicates": args.replicates, "k": k_classes}
    stamp_path = work / "settings.pkl"
    if not (stamp_path.exists() and pickle.loads(stamp_path.read_bytes()) == stamp):
        for old in work.glob("*.pkl"):
            old.unlink()
        stamp_path.write_bytes(pickle.dumps(stamp))

    def saved_path(label: tuple) -> Path:
        return work / f"{label[0]}_{label[1]}.pkl".replace(" ", "_")

    results = {task["label"]: pickle.loads(saved_path(task["label"]).read_bytes()) for task in tasks if saved_path(task["label"]).exists()}
    todo = [task for task in tasks if task["label"] not in results]
    log(f"{len(tasks)} datasets, {len(results)} already on file; fitting {sum(len(task['models']) for task in todo)} models, {args.jobs} datasets at a time")

    def record(done: dict) -> None:
        results[done["label"]] = done
        saved_path(done["label"]).write_bytes(pickle.dumps(done))
        log(f"   finished {done['label'][0]} {done['label'][1]}")

    if args.jobs > 1 and todo:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for future in as_completed([pool.submit(fit_models, task) for task in todo]):
                record(future.result())
    else:
        for task in todo:
            record(fit_models(task))
    results = {task["label"]: results[task["label"]] for task in tasks}  # back in the order the tables expect

    # 1. The real data.
    real = results[("real", 0)]["fits"]
    one_group = real[(1, True)]
    real_table = pd.DataFrame([describe_fit(fit, one_group) for fit in real.values()])
    real_table.to_csv(args.results / "factor_mixture.csv", index=False)
    log("real data:\n" + real_table.drop(columns=["group_means", "log_likelihood", "at_bound"]).round(3).to_string(index=False))
    np.savez_compressed(
        args.results / "factor_mixture_fits.npz",
        year=table["year"].to_numpy(),
        hhid=table["hhid"].to_numpy(),
        slopes_one_group=one_group.slopes,
        **{
            f"{name}_{'shared' if shared_spread else 'separate'}_k{n_groups}": getattr(fit, attribute)
            for (n_groups, shared_spread), fit in real.items()
            for name, attribute in [
                ("shares", "group_shares"),
                ("means", "group_means"),
                ("covariances", "group_covariances"),
                ("probabilities", "group_probabilities"),
                ("slopes", "slopes"),
            ]
        },
    )

    # 2. Held-out years: each model against the one-group model, child by child.
    heldout_rows = []
    for split in SPLITS:
        done = results[("heldout", split)]
        base = done["heldout"][(1, True)]
        for (n_groups, shared_spread), per_child in done["heldout"].items():
            difference = per_child - base
            heldout_rows.append(
                {
                    "split": split,
                    "version": version_name(shared_spread),
                    "groups": n_groups,
                    "n_test": len(per_child),
                    "heldout_log_likelihood": float(per_child.sum()),
                    "gain_over_one_group": float(difference.sum()),
                    "gain_se": float(difference.std(ddof=1) * np.sqrt(len(difference))),
                    "train_bic_vs_one_group": done["fits"][(n_groups, shared_spread)].bic() - done["fits"][(1, True)].bic(),
                }
            )
    heldout_table = pd.DataFrame(heldout_rows)
    heldout_table.to_csv(args.results / "factor_mixture_heldout.csv", index=False)
    log("held-out years, gain over one group (log-likelihood units):\n" + heldout_table.round(2).to_string(index=False))

    # 3. Known truth: how much do groups add when there are none, and when there is nothing else?
    calibration_rows = []
    for (data, replicate), done in results.items():
        if data == "heldout":
            continue
        base = done["fits"][(1, True)]
        for (n_groups, shared_spread), fit in done["fits"].items():
            if not shared_spread or n_groups == 1:
                continue
            calibration_rows.append(
                {
                    "data": data,
                    "replicate": replicate,
                    "groups": n_groups,
                    "log_likelihood_gain": fit.log_likelihood - base.log_likelihood,
                    "bic_vs_one_group": fit.bic() - base.bic(),
                    "confidently_placed": float((fit.group_probabilities.max(axis=1) >= CONFIDENT).mean()),
                    "converged": fit.converged,
                }
            )
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(args.results / "factor_mixture_calibration.csv", index=False)
    ranges = calibration.groupby(["data", "groups"], sort=False)["bic_vs_one_group"].agg(["min", "max"])
    log("BIC against one group (negative: groups help), shared spread:\n" + ranges.round(1).to_string())

    best = real_table.loc[real_table["bic"].idxmin()]
    save_json(
        {
            "lowest_bic": {"version": best["version"], "groups": int(best["groups"]), "bic_vs_one_group": float(best["bic_vs_one_group"])},
            "real": real_table.to_dict(orient="records"),
            "heldout": heldout_rows,
            "calibration_bic_ranges": {
                f"{data}, {groups} groups": {"min": float(row["min"]), "max": float(row["max"])} for (data, groups), row in ranges.iterrows()
            },
            "classes_simulated_from_k": k_classes,
            "confident_threshold": CONFIDENT,
            "settings": {**settings, "k_max": args.k_max, "k_max_separate": args.k_max_separate, "replicates": args.replicates},
        },
        args.results / "factor_mixture.json",
    )
    log(f"wrote factor_mixture.csv, _heldout.csv, _calibration.csv, _fits.npz and factor_mixture.json to {args.results}")


if __name__ == "__main__":
    main()
