"""Step 4 of 14. Does a second clustering method, with no probability model, find the same groups?

What happens here
-----------------
1. Load the same children and the latent class model for the chosen K.
2. Cluster the children again with a different method, cut to the same K.
3. Match each new cluster to the latent class whose condition profile it
   most resembles, so "cluster 2" and "class 0" can be compared fairly.
4. Score agreement with the adjusted Rand index (pass line 0.65), on all
   children and again on only the children the latent class model is
   confident about.

Why the second score
--------------------
The latent class model gives each child a probability for each class. About
a quarter of children sit near a boundary (no class above 0.7), and for them
the "class" is close to a coin flip. A second method cannot be expected to
agree with a coin flip. Scoring agreement on confidently classified children
separates "the two methods find different groups" from "the two methods
draw the boundary in slightly different places".

k-means has many near-tied solutions
------------------------------------
On these flags k-means finds several solutions whose total distance to the
centers differs by well under 0.1%, but which split the children differently.
Which one a run returns depends on the random seed, and with 20 starts the
ARI against the latent classes ranged from 0.47 to 0.65 at K=4 across ten
seeds. So k-means is run with 200 starts from each of several seeds
(``--kmeans-seeds``); the solution with the lowest total distance is the one
reported, and the ARI of every seed's solution is saved alongside it so the
spread is visible. A spread like this is itself a sign that the data have no
clear cluster boundaries for k-means to find.

Methods
-------
::

    --method kmeans         k-means on the 0/1 flags (the reported second method)
    --method hierarchical   Jaccard distance, average linkage (the proposal's choice)
    --method alternatives   seven methods side by side, written to method_comparison.csv

Average linkage on this data does not produce K usable groups at any K: it
chains through the 517 children with no conditions and leaves one cluster
with nearly everyone. That is a known weakness of average linkage on sparse
binary data, not evidence about the children, and it is why k-means is the
reported second method. Hierarchical trees are cut with ``cut_tree`` rather
than SciPy's ``fcluster(..., "maxclust")``, which returns fewer clusters
than asked when several merges tie at the same height.

Outputs, under ``--results``::

    crosscheck_{method}.json          ARI on all and on confident children, sizes, crosstab
    profiles_{method}_k{K}.csv/.png   condition rates by cluster, in the latent classes' numbering
    figures/dendrogram.png            hierarchical only
    method_comparison.csv             alternatives only

Usage::

    python scripts/04_crosscheck.py --backend reference --method kmeans --k 4
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import MAIN_YEARS, backend_or_exit, load_analysis, load_lca_fit, log, make_parser, resolve_k, save_json
from nsch_subtypes.data import to_flag_matrix
from nsch_subtypes.describe import class_profiles
from nsch_subtypes.plots import dendrogram_plot, profile_heatmap
from nsch_subtypes.validate import ARI_PASS, CONFIDENCE_LEVELS, adjusted_rand_index, match_classes, relabel_to_match


def agreement(latent_labels: np.ndarray, max_posterior: np.ndarray, method_labels: np.ndarray) -> dict:
    """ARI on all children and on children whose largest class probability is at least each confidence level."""
    out = {"ari_all": adjusted_rand_index(latent_labels, method_labels), "by_confidence": []}
    for level in CONFIDENCE_LEVELS:
        keep = max_posterior >= level
        out["by_confidence"].append(
            {"min_posterior": level, "share_of_children": float(keep.mean()), "ari": adjusted_rand_index(latent_labels[keep], method_labels[keep])}
        )
    return out


def matched(method_labels: np.ndarray, X: np.ndarray, flags: list[str], latent_rates: np.ndarray, k: int) -> np.ndarray:
    """Renumber a method's clusters to the latent classes' numbering. Requires exactly K non-empty clusters."""
    if len(np.unique(method_labels)) != k:
        return method_labels  # cannot match; ARI does not need matching, profiles do
    profiles = class_profiles(X, method_labels, flags, n_classes=k)
    return relabel_to_match(method_labels, match_classes(latent_rates, profiles[flags].to_numpy()))


def cut(merges: np.ndarray, k: int) -> np.ndarray:
    """K groups from a SciPy linkage matrix, robust to tied merge heights."""
    from scipy.cluster.hierarchy import cut_tree

    return cut_tree(merges, n_clusters=k).ravel()


def weighted_jaccard_condensed(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Condensed Jaccard distances where each condition counts by its weight (two all-zero rows are distance 0)."""
    Xw = X * weights
    totals = Xw.sum(axis=1)
    pieces = []
    for i in range(X.shape[0] - 1):
        shared = Xw[i + 1 :] @ X[i]
        union = totals[i] + totals[i + 1 :] - shared
        with np.errstate(invalid="ignore", divide="ignore"):
            pieces.append(np.clip(np.where(union > 0, 1 - shared / union, 0.0), 0.0, 1.0))  # clip rounding error
    return np.concatenate(pieces)


def compare_alternatives(X: np.ndarray, k: int, latent_labels: np.ndarray, max_posterior: np.ndarray, seed: int) -> pd.DataFrame:
    """Seven clustering methods against the latent classes, every one on all 8,891 rows."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist, squareform
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.decomposition import PCA
    from sklearn.neighbors import kneighbors_graph

    Xf = X.astype(float)
    jaccard = np.nan_to_num(pdist(X.astype(bool), "jaccard"))  # two all-zero rows: SciPy gives NaN, we use 0
    labels = {
        "k-means on the flags": KMeans(k, n_init=200, random_state=seed).fit(Xf).labels_,
        "k-means on the first two principal components": KMeans(k, n_init=200, random_state=seed).fit(PCA(2).fit_transform(Xf)).labels_,
        "Ward linkage, Euclidean": cut(linkage(Xf, "ward"), k),
        "average linkage, Jaccard": cut(linkage(jaccard, "average"), k),
        "complete linkage, Jaccard": cut(linkage(jaccard, "complete"), k),
    }
    weights = -np.log(Xf.mean(axis=0))  # rarer conditions count more when shared
    labels["average linkage, prevalence-weighted Jaccard"] = cut(linkage(weighted_jaccard_condensed(Xf, weights), "average"), k)
    graph = kneighbors_graph(squareform(jaccard), n_neighbors=30, metric="precomputed", include_self=False)
    graph = 0.5 * (graph + graph.T)
    labels["spectral clustering, 30-nearest-neighbor Jaccard graph"] = (
        SpectralClustering(k, affinity="precomputed", random_state=seed, assign_labels="discretize").fit(graph).labels_
    )

    rows = []
    for name, lab in labels.items():
        sizes = np.bincount(lab, minlength=k)
        score = agreement(latent_labels, max_posterior, lab)
        rows.append(
            {
                "method": name,
                "ari_all": round(score["ari_all"], 3),
                **{f"ari_posterior_{c['min_posterior']}": round(c["ari"], 3) for c in score["by_confidence"]},
                "largest_cluster_share": round(sizes.max() / len(lab), 3),
                "cluster_sizes": sorted(sizes.tolist(), reverse=True),
            }
        )
        log(f"   {name}: ARI {score['ari_all']:.3f}; sizes {sorted(sizes.tolist(), reverse=True)}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = make_parser("Cross-check the latent classes with a second clustering method")
    parser.add_argument("--method", choices=["kmeans", "hierarchical", "alternatives"], default="kmeans")
    parser.add_argument("--k", type=int, default=None, help="number of classes (default: the held-out choice from step 13)")
    parser.add_argument("--n-starts", type=int, default=200, help="k-means only: random starts per seed")
    parser.add_argument("--kmeans-seeds", type=int, default=5, help="k-means only: independent seeds; the lowest-inertia solution is reported")
    args = parser.parse_args()

    table, flags = load_analysis(MAIN_YEARS)
    X = to_flag_matrix(table, flags)
    k = resolve_k(args)
    latent = load_lca_fit(args.results, k, table, flags)
    latent_labels = latent.labels
    max_posterior = latent.responsibilities.max(axis=1)

    if args.method == "alternatives":
        comparison = compare_alternatives(X, k, latent_labels, max_posterior, args.seed)
        comparison.to_csv(args.results / "method_comparison.csv", index=False)
        log(comparison.to_string(index=False))
        return

    backend = backend_or_exit(args.backend, args.method)
    merges = None
    if args.method == "hierarchical":
        if args.backend == "ours":
            # Collapse identical rows and pass their counts as weights (exact for average linkage).
            unique_rows, row_of_child, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
            log(f"{X.shape[0]} children collapse to {len(unique_rows)} distinct profiles")
            result = backend["hierarchical"](unique_rows, k, weights=counts)
            method_labels = result.labels[row_of_child.ravel()]
        else:
            result = backend["hierarchical"](X, k)
            method_labels = result.labels
        merges = result.merges
    else:
        runs = [backend["kmeans"](X, k, n_starts=args.n_starts, seed=args.seed + i) for i in range(args.kmeans_seeds)]
        result = min(runs, key=lambda run: run.inertia)
        method_labels = result.labels
        by_seed = [{"seed": args.seed + i, "inertia": run.inertia, "ari": adjusted_rand_index(latent_labels, run.labels)} for i, run in enumerate(runs)]
        log(f"k-means, {args.kmeans_seeds} seeds x {args.n_starts} starts: ARI vs classes by seed {[round(r['ari'], 3) for r in by_seed]}")
        log(f"   inertia by seed {[round(r['inertia'], 1) for r in by_seed]}; the lowest is reported")

    method_labels = matched(method_labels, X, flags, latent.condition_rates, k)
    score = agreement(latent_labels, max_posterior, method_labels)
    crosstab = pd.crosstab(pd.Categorical(latent_labels, categories=range(k)), pd.Categorical(method_labels, categories=range(k)), dropna=False)

    tag = args.method
    extra = {}
    if args.method == "kmeans":
        aris = [r["ari"] for r in by_seed]
        extra = {"n_starts_per_seed": args.n_starts, "by_seed": by_seed, "ari_range_across_seeds": [min(aris), max(aris)]}
    save_json(
        {
            "method": args.method,
            "k": k,
            "backend": args.backend,
            "ari": score["ari_all"],
            "pass": score["ari_all"] >= ARI_PASS,
            "pass_line": ARI_PASS,
            "ari_by_confidence": score["by_confidence"],
            "cluster_sizes_matched": np.bincount(method_labels, minlength=k),
            "crosstab_lca_rows_method_cols": crosstab.to_numpy(),
            **extra,
        },
        args.results / f"crosscheck_{tag}.json",
    )
    matched_profiles = class_profiles(X, method_labels, flags, n_classes=k)
    matched_profiles.to_csv(args.results / f"profiles_{tag}_k{k}.csv", index=False)
    (args.results / "figures").mkdir(parents=True, exist_ok=True)
    profile_heatmap(
        matched_profiles, flags, args.results / "figures" / f"profiles_{tag}_k{k}.png", title=f"{tag} clusters, K={k}, matched to the latent classes"
    )
    if merges is not None:
        dendrogram_plot(merges, k, args.results / "figures" / "dendrogram.png", title="Average-linkage tree on Jaccard distances")

    verdict = "pass" if score["ari_all"] >= ARI_PASS else "below the pass line"
    log(f"ARI, latent classes vs {tag}: {score['ari_all']:.3f} ({verdict}); cluster sizes {np.bincount(method_labels, minlength=k).tolist()}")
    for c in score["by_confidence"]:
        log(f"   children with largest class probability >= {c['min_posterior']}: {c['share_of_children']:.0%} of children, ARI {c['ari']:.3f}")


if __name__ == "__main__":
    main()
