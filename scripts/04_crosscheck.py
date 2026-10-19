"""Step 4. Cross-check the latent classes with a second clustering method.

Clusters the same children with a method that has no probability model,
cuts to the K chosen in step 3, matches clusters to classes by profile, and
compares the two labelings with the adjusted Rand index. Pass line: ARI >= 0.65.

Two methods are available:

    --method hierarchical   Jaccard distance, average linkage (the proposal's choice)
    --method kmeans         k-means on the 0/1 flags (as in Vargason et al. 2019)

Average linkage turned out to peel off outliers on this data, leaving one
cluster with nearly every child at any K; k-means was added as the
alternative. Both can be run and the results compared.

With ``--backend ours --method hierarchical`` identical condition profiles are
collapsed first and passed with their counts, which is exact for average
linkage and much faster.

    python scripts/04_crosscheck.py --backend reference --method kmeans
"""

from __future__ import annotations

import json

import numpy as np

from _common import MAIN_YEARS, backend_or_exit, load_analysis, load_lca, log, parser, save_json
from nsch_subtypes.data import flag_matrix
from nsch_subtypes.describe import class_profiles
from nsch_subtypes.plots import dendrogram_plot, profile_heatmap
from nsch_subtypes.validate import ARI_PASS, ari, match_classes, relabel


def main() -> None:
    p = parser("Cross-check with a second clustering method")
    p.add_argument("--method", choices=["hierarchical", "kmeans"], default="hierarchical")
    p.add_argument("--k", type=int, default=None, help="override the K chosen in step 3")
    p.add_argument("--n-starts", type=int, default=20, help="k-means only")
    args = p.parse_args()

    backend = backend_or_exit(args.backend, args.method)
    table, flags = load_analysis(MAIN_YEARS)
    X = flag_matrix(table, flags)
    k = args.k or json.loads((args.results / "lca_best.json").read_text())["best_k"]
    lca = load_lca(args.results, k)
    lca_labels = lca["responsibilities"].argmax(axis=1)

    merges = None
    if args.method == "hierarchical":
        if args.backend == "ours":
            uniq, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
            log(f"{X.shape[0]} children collapse to {len(uniq)} distinct profiles")
            res = backend["hierarchical"](uniq, k, weights=counts)
            labels = res.labels[inverse.ravel()]
        else:
            res = backend["hierarchical"](X, k)
            labels = res.labels
        merges = res.merges
    else:
        res = backend["kmeans"](X, k, n_starts=args.n_starts, seed=args.seed)
        labels = res.labels

    profiles = class_profiles(X, labels, flags)
    perm = match_classes(lca["condition_rates"], profiles[flags].to_numpy())
    labels_matched = relabel(labels, perm)
    score = ari(lca_labels, labels_matched)
    crosstab = np.zeros((k, k), dtype=int)
    for a, b in zip(lca_labels, labels_matched, strict=False):
        crosstab[a, b] += 1

    tag = args.method
    save_json(
        {
            "method": args.method,
            "k": k,
            "backend": args.backend,
            "ari": score,
            "pass": score >= ARI_PASS,
            "pass_line": ARI_PASS,
            "cluster_sizes_matched": np.bincount(labels_matched, minlength=k).tolist(),
            "crosstab_lca_rows_method_cols": crosstab.tolist(),
        },
        args.results / f"crosscheck_{tag}.json",
    )
    class_profiles(X, labels_matched, flags).to_csv(args.results / f"profiles_{tag}_k{k}.csv", index=False)
    profile_heatmap(
        class_profiles(X, labels_matched, flags),
        flags,
        args.results / "figures" / f"profiles_{tag}_k{k}.png",
        title=f"{tag} clusters, K={k}, matched to the latent classes",
    )
    if merges is not None:
        dendrogram_plot(merges, k, args.results / "figures" / "dendrogram.png", title="Average-linkage tree on Jaccard distances")
    verdict = "pass" if score >= ARI_PASS else "below the pass line"
    sizes = np.bincount(labels_matched, minlength=k).tolist()
    log(f"ARI, latent classes vs {args.method}: {score:.3f} ({verdict}); cluster sizes {sizes}")


if __name__ == "__main__":
    main()
