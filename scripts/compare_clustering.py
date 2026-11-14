"""Compare our k-means and hierarchical clustering with scikit-learn and SciPy on the survey data.

``compare_backends.py`` checks our latent class model against StepMix using
saved fits. The two second methods are cheap enough to rerun, so this script
fits both versions of each on the same 8,891 children and compares them
directly::

    python scripts/compare_clustering.py

What is compared
----------------
k-means
    Our ``cluster_kmeans`` and scikit-learn's ``KMeans``, each with the same
    number of random starts and seeds, at K = 2 to 6. The two choose their
    random starting centers differently, so they need not land on the same
    solution, and on these flags k-means has several near-tied solutions (see
    step 4). Three numbers separate "different code" from "different luck":

    - the best total distance (inertia) each finds; equal or near-equal values
      mean both reach equally good solutions;
    - the ARI between the two best solutions;
    - a fixed-point check: assigning every child to the nearest of
      scikit-learn's final centers with our distance code must give back
      scikit-learn's clusters, and the means of those clusters must match its
      centers. scikit-learn stops once the centers move less than a small
      tolerance, so its stored centers can differ from the final cluster means
      in the fourth decimal place; the largest such difference is reported.
      This tests our arithmetic on the real data without depending on starts.

hierarchical
    Our ``cluster_hierarchical`` (on the distinct profiles, weighted by how
    many children share each) and SciPy's ``linkage(..., "average")`` on every
    child, cut at K = 2 to 6. Average linkage is deterministic apart from the
    order in which tied merges are taken, so we compare the cuts by ARI and the
    two trees by their merge heights. SciPy's tree first merges identical
    children at height 0; after dropping those merges, the two trees should
    have the same number of merges. Jaccard distances on 18 yes/no flags tie
    often (many pairs sit at exactly 1/2 or 2/3), and which tie is merged first
    changes the averages computed later. To see how much that alone matters,
    SciPy is also run on the same children in several shuffled orders and
    compared with itself; differences between our tree and SciPy's that fall
    within the range of SciPy's differences with itself come from tie order,
    not from our code.

Outputs::

    reports/tables/backend_clustering.csv     one row per method and K
    results/backend_clustering.json           the same numbers plus settings
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _common import MAIN_YEARS, REPO, load_analysis, log, save_json
from nsch_subtypes.data import to_flag_matrix
from nsch_subtypes.hierarchical import cluster_hierarchical
from nsch_subtypes.partition import _squared_distances, cluster_kmeans
from nsch_subtypes.reference import cluster_kmeans_sklearn, cluster_scipy
from nsch_subtypes.validate import adjusted_rand_index


def best_of_seeds(fit, X: np.ndarray, k: int, n_starts: int, seeds: list[int]):
    """Run a k-means function once per seed and keep the run with the lowest inertia."""
    runs = [fit(X, k, n_starts=n_starts, seed=seed) for seed in seeds]
    return min(runs, key=lambda run: run.inertia), runs


def fixed_point(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> tuple[bool, float]:
    """One round of our assign/update step started from a converged k-means solution.

    Returns whether assigning each child to the nearest center (with our
    distance code) reproduces ``labels`` exactly, and the largest difference
    between the given centers and the means of the reassigned clusters.
    """
    reassigned = _squared_distances(X, centers).argmin(axis=1)
    same_labels = bool((reassigned == labels).all())
    recomputed = np.vstack([X[reassigned == c].mean(axis=0) for c in range(len(centers))])
    return same_labels, float(np.abs(recomputed - centers).max())


def compare_kmeans(X: np.ndarray, ks: list[int], n_starts: int, seeds: list[int]) -> list[dict]:
    """Our k-means against scikit-learn's at each K."""
    rows = []
    for k in ks:
        start = time.perf_counter()
        ours, _ = best_of_seeds(cluster_kmeans, X, k, n_starts, seeds)
        reference, _ = best_of_seeds(cluster_kmeans_sklearn, X, k, n_starts, seeds)
        same_labels, center_gap = fixed_point(X, reference.labels, reference.centers)
        rows.append(
            {
                "method": "k-means",
                "k": k,
                "ari_ours_vs_library": adjusted_rand_index(ours.labels, reference.labels),
                "inertia_ours": ours.inertia,
                "inertia_library": reference.inertia,
                "inertia_gap_percent": 100 * (ours.inertia - reference.inertia) / reference.inertia,
                "reassignment_matches_library": same_labels,
                "max_center_gap": center_gap,
                "sizes_ours": sorted(np.bincount(ours.labels, minlength=k).tolist(), reverse=True),
                "sizes_library": sorted(np.bincount(reference.labels, minlength=k).tolist(), reverse=True),
            }
        )
        log(
            f"k-means K={k}: ARI {rows[-1]['ari_ours_vs_library']:.3f}, inertia {ours.inertia:.1f} vs {reference.inertia:.1f}, "
            f"reassignment {'matches' if same_labels else 'DIFFERS'}, center gap {center_gap:.1e} ({time.perf_counter() - start:.0f}s)"
        )
    return rows


def nonzero_heights(merges: np.ndarray) -> np.ndarray:
    """Sorted merge heights above zero (merges of identical children happen at height 0)."""
    heights = merges[:, 2]
    return np.sort(heights[heights > 1e-12])


def compare_hierarchical(X: np.ndarray, ks: list[int], n_shuffles: int = 5, seed: int = 0) -> list[dict]:
    """Our weighted average linkage on distinct profiles against SciPy's on every child."""
    unique_rows, row_of_child, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    row_of_child = row_of_child.ravel()
    log(f"hierarchical: {len(X)} children, {len(unique_rows)} distinct profiles")

    start = time.perf_counter()
    reference_tree = cluster_scipy(X, ks[0]).merges
    log(f"   SciPy tree on every child: {time.perf_counter() - start:.0f}s")
    start = time.perf_counter()
    ours_tree = cluster_hierarchical(unique_rows, ks[0], weights=counts).merges
    log(f"   our tree on distinct profiles: {time.perf_counter() - start:.0f}s")
    random_generator = np.random.default_rng(seed)
    shuffles = []  # (tree, position of each child in the shuffled order)
    for _ in range(n_shuffles):
        order = random_generator.permutation(len(X))
        shuffles.append((cluster_scipy(X[order], ks[0]).merges, np.argsort(order)))

    ours_heights, reference_heights = nonzero_heights(ours_tree), nonzero_heights(reference_tree)
    same_count = len(ours_heights) == len(reference_heights)
    height_gap = float(np.abs(ours_heights - reference_heights).max()) if same_count else float("nan")
    shuffled_gaps = [float(np.abs(nonzero_heights(tree) - reference_heights).max()) for tree, _ in shuffles]

    from scipy.cluster.hierarchy import cut_tree

    rows = []
    for k in ks:
        ours_labels = cluster_hierarchical(unique_rows, k, weights=counts).labels[row_of_child]
        reference_labels = cut_tree(reference_tree, n_clusters=k).ravel()
        shuffled_aris = [adjusted_rand_index(cut_tree(tree, n_clusters=k).ravel()[back], reference_labels) for tree, back in shuffles]
        rows.append(
            {
                "method": "hierarchical (average linkage, Jaccard)",
                "k": k,
                "ari_ours_vs_library": adjusted_rand_index(ours_labels, reference_labels),
                "ari_library_vs_shuffled_library_min": min(shuffled_aris),
                "ari_library_vs_shuffled_library_max": max(shuffled_aris),
                "nonzero_merges_ours": len(ours_heights),
                "nonzero_merges_library": len(reference_heights),
                "max_merge_height_gap": height_gap,
                "max_merge_height_gap_library_vs_shuffled_min": min(shuffled_gaps),
                "max_merge_height_gap_library_vs_shuffled_max": max(shuffled_gaps),
                "sizes_ours": sorted(np.bincount(ours_labels, minlength=k).tolist(), reverse=True),
                "sizes_library": sorted(np.bincount(reference_labels, minlength=k).tolist(), reverse=True),
            }
        )
        row = rows[-1]
        log(
            f"hierarchical K={k}: ARI {row['ari_ours_vs_library']:.3f} "
            f"(SciPy vs shuffled SciPy {min(shuffled_aris):.3f} to {max(shuffled_aris):.3f}); "
            f"sizes ours {row['sizes_ours']}, SciPy {row['sizes_library']}"
        )
    log(f"   merges above height 0: ours {len(ours_heights)}, SciPy {len(reference_heights)}")
    log(f"   largest height gap {height_gap:.3f} (SciPy vs shuffled SciPy {min(shuffled_gaps):.3f} to {max(shuffled_gaps):.3f})")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5, 6], help="K values to compare")
    parser.add_argument("--n-starts", type=int, default=200, help="k-means random starts per seed (step 4 uses 200)")
    parser.add_argument("--seeds", type=int, default=5, help="k-means seeds per implementation (step 4 uses 5)")
    parser.add_argument("--shuffles", type=int, default=5, help="hierarchical: shuffled-order SciPy runs used as the tie-order control")
    parser.add_argument("--methods", nargs="+", choices=["kmeans", "hierarchical"], default=["kmeans", "hierarchical"], help="which comparisons to run")
    parser.add_argument("--reports", type=Path, default=REPO / "reports", help="output root for the table")
    parser.add_argument("--results", type=Path, default=REPO / "results", help="where to save the JSON")
    args = parser.parse_args()

    table, flags = load_analysis(MAIN_YEARS)
    X = to_flag_matrix(table, flags)
    ks = sorted(args.ks)
    seeds = list(range(args.seeds))

    rows = []
    if "kmeans" in args.methods:
        rows += compare_kmeans(X, ks, args.n_starts, seeds)
    if "hierarchical" in args.methods:
        rows += compare_hierarchical(X, ks, n_shuffles=args.shuffles)
    summary = pd.DataFrame(rows)
    tables_dir = args.reports / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "backend_clustering.csv", index=False)
    args.results.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "children": len(X),
            "conditions": len(flags),
            "ks": ks,
            "kmeans_starts_per_seed": args.n_starts,
            "kmeans_seeds": seeds,
            "hierarchical_shuffles": args.shuffles,
            "comparisons": rows,
        },
        args.results / "backend_clustering.json",
    )
    log(f"Wrote {tables_dir / 'backend_clustering.csv'} and {args.results / 'backend_clustering.json'}")


if __name__ == "__main__":
    main()
