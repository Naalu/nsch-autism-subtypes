"""Step 4 of 8. Does a second clustering method, with no probability model, find the same groups?

What happens here
-----------------
1. Load the same children and the latent classes chosen in step 3.
2. Cluster the children again with a different method, cut to the same K.
3. Match each new cluster to the latent class whose condition profile it
   most resembles (so "cluster 2" and "class 0" can be compared fairly).
4. Score the agreement with the adjusted Rand index. Pass line: ARI >= 0.65.

Two methods are available::

    --method hierarchical   Jaccard distance, average linkage (the proposal's choice)
    --method kmeans         k-means on the 0/1 flags (as in Vargason et al. 2019)

On this data, average linkage peels off outliers one at a time and leaves one
cluster holding nearly every child, at any K. That is why k-means was added.
Run both and compare; which one the paper reports is an open decision
(``docs/validation-checklist.md``).

With ``--backend ours --method hierarchical``, children with identical flag
rows are collapsed to one row with a count first. Average linkage gives the
same answer either way and it is much faster.

Outputs, under ``results/``::

    crosscheck_{method}.json          ARI, pass/fail, cluster sizes, a class-by-cluster crosstab
    profiles_{method}_k{K}.csv/.png   condition rates by cluster, in the latent classes' numbering
    figures/dendrogram.png            the tree, hierarchical only

Usage::

    python scripts/04_crosscheck.py --backend reference --method kmeans
"""

from __future__ import annotations

import numpy as np

from _common import MAIN_YEARS, backend_or_exit, load_analysis, load_lca_fit, log, make_parser, read_best_k, save_json
from nsch_subtypes.data import to_flag_matrix
from nsch_subtypes.describe import class_profiles
from nsch_subtypes.plots import dendrogram_plot, profile_heatmap
from nsch_subtypes.validate import ARI_PASS, adjusted_rand_index, match_classes, relabel_to_match


def main() -> None:
    parser = make_parser("Cross-check the latent classes with a second clustering method")
    parser.add_argument("--method", choices=["hierarchical", "kmeans"], default="hierarchical")
    parser.add_argument("--k", type=int, default=None, help="override the K chosen in step 3")
    parser.add_argument("--n-starts", type=int, default=20, help="k-means only")
    args = parser.parse_args()

    backend = backend_or_exit(args.backend, args.method)
    table, flags = load_analysis(MAIN_YEARS)
    flag_matrix = to_flag_matrix(table, flags)
    k = args.k or read_best_k(args.results)
    latent = load_lca_fit(args.results, k)
    latent_labels = latent["responsibilities"].argmax(axis=1)

    # ---- cluster with the second method ----------------------------------------
    merges = None
    if args.method == "hierarchical":
        if args.backend == "ours":
            # Collapse identical rows and pass their counts as weights.
            unique_rows, row_of_child, counts = np.unique(flag_matrix, axis=0, return_inverse=True, return_counts=True)
            log(f"{flag_matrix.shape[0]} children collapse to {len(unique_rows)} distinct profiles")
            result = backend["hierarchical"](unique_rows, k, weights=counts)
            method_labels = result.labels[row_of_child.ravel()]
        else:
            result = backend["hierarchical"](flag_matrix, k)
            method_labels = result.labels
        merges = result.merges
    else:
        result = backend["kmeans"](flag_matrix, k, n_starts=args.n_starts, seed=args.seed)
        method_labels = result.labels

    # ---- line the clusters up with the latent classes and score ------------------
    method_profiles = class_profiles(flag_matrix, method_labels, flags)
    perm = match_classes(latent["condition_rates"], method_profiles[flags].to_numpy())
    method_labels = relabel_to_match(method_labels, perm)
    score = adjusted_rand_index(latent_labels, method_labels)

    # How many children fall in each (latent class, method cluster) pair.
    crosstab = np.zeros((k, k), dtype=int)
    for latent_class, method_cluster in zip(latent_labels, method_labels, strict=False):
        crosstab[latent_class, method_cluster] += 1

    # ---- save ------------------------------------------------------------------------
    tag = args.method
    save_json(
        {
            "method": args.method,
            "k": k,
            "backend": args.backend,
            "ari": score,
            "pass": score >= ARI_PASS,
            "pass_line": ARI_PASS,
            "cluster_sizes_matched": np.bincount(method_labels, minlength=k).tolist(),
            "crosstab_lca_rows_method_cols": crosstab.tolist(),
        },
        args.results / f"crosscheck_{tag}.json",
    )
    matched_profiles = class_profiles(flag_matrix, method_labels, flags)
    matched_profiles.to_csv(args.results / f"profiles_{tag}_k{k}.csv", index=False)
    profile_heatmap(
        matched_profiles,
        flags,
        args.results / "figures" / f"profiles_{tag}_k{k}.png",
        title=f"{tag} clusters, K={k}, matched to the latent classes",
    )
    if merges is not None:
        dendrogram_plot(merges, k, args.results / "figures" / "dendrogram.png", title="Average-linkage tree on Jaccard distances")

    verdict = "pass" if score >= ARI_PASS else "below the pass line"
    cluster_sizes = np.bincount(method_labels, minlength=k).tolist()
    log(f"ARI, latent classes vs {args.method}: {score:.3f} ({verdict}); cluster sizes {cluster_sizes}")


if __name__ == "__main__":
    main()
