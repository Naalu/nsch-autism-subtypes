"""Step 6. Bootstrap stability of each class (Hennig 2007).

Refits the chosen model on resamples drawn with replacement and scores how
much of each original class comes back. Mean Jaccard >= 0.75 counts as
stable; <= 0.5 means the class dissolved.

    python scripts/06_bootstrap.py --backend reference --n-boot 200
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _common import MAIN_YEARS, backend_or_exit, load_analysis, load_lca, log, parser
from nsch_subtypes.data import flag_matrix
from nsch_subtypes.plots import stability_plot
from nsch_subtypes.validate import bootstrap_stability, match_classes, relabel


def main() -> None:
    p = parser("Bootstrap stability per class")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--n-starts", type=int, default=20)
    p.add_argument("--k", type=int, default=None)
    args = p.parse_args()

    fit_lca = backend_or_exit(args.backend, "lca")["lca"]
    k = args.k or json.loads((args.results / "lca_best.json").read_text())["best_k"]
    table, flags = load_analysis(MAIN_YEARS)
    X = flag_matrix(table, flags)
    ref = load_lca(args.results, k)
    labels_ref = ref["responsibilities"].argmax(axis=1)

    def fit(X_boot: np.ndarray, n_classes: int) -> np.ndarray:
        res = fit_lca(X_boot, n_classes, n_starts=args.n_starts, seed=args.seed)
        perm = match_classes(ref["condition_rates"], res.condition_rates)
        return relabel(res.labels, perm)

    log(f"{args.n_boot} bootstrap refits at K={k} with {args.n_starts} starts each")
    stability = bootstrap_stability(
        X, labels_ref, fit, k, n_boot=args.n_boot, seed=args.seed, progress=lambda b: log(f"  {b}/{args.n_boot}") if b % 20 == 0 else None
    )

    out = pd.DataFrame(
        {
            "class": range(k),
            "n": np.bincount(labels_ref, minlength=k),
            "mean_jaccard": stability.mean,
            "sd_jaccard": stability.jaccard.std(axis=1),
            "verdict": stability.verdicts(),
        }
    )
    out.to_csv(args.results / "stability.csv", index=False)
    np.save(args.results / "stability_jaccard.npy", stability.jaccard)
    stability_plot(stability.mean, args.results / "figures" / "stability.png")
    log(out.to_string(index=False))


if __name__ == "__main__":
    main()
