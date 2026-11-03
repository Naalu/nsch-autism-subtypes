"""Run every pipeline script once, end to end, at tiny settings.

The unit tests check the building blocks on made-up data. This test checks
the plumbing: that each numbered script still runs against the committed
analysis table, reads what the step before it wrote, and writes what the
step after it expects. It catches the bugs unit tests miss, such as a
renamed column or a JSON key that one script writes and another reads.

The settings are far too small for the numbers to mean anything (two
random starts, three bootstrap resamples). Only "did it run, and did it
write its files" is checked.

Both backends run separately and can take several minutes each, so this
is skipped by default. Run it with::

    uv run pytest -m slow        # or: make smoke

Everything is written under pytest's temporary folder; ``results/`` and
``reports/`` are not touched.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
K = 3  # small enough to be quick, large enough that "the classes" means something


def run(script: str, *args: str | Path) -> str:
    """Run one pipeline script and fail the test with its output if it errors."""
    command = [sys.executable, str(REPO / "scripts" / script), *map(str, args)]
    finished = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    if finished.returncode != 0:
        pytest.fail(f"{script} exited with {finished.returncode}\n--- stdout\n{finished.stdout[-3000:]}\n--- stderr\n{finished.stderr[-3000:]}")
    return finished.stdout


# Run the same pipeline contract for the library and NumPy implementations.
@pytest.mark.slow
@pytest.mark.parametrize("backend", ["reference", "ours"])
def test_every_script_runs_at_tiny_settings(tmp_path: Path, backend: str) -> None:
    main_results = tmp_path / "smoke"  # plays the part of results/
    results_for_k = tmp_path / f"smoke_k{K}"  # plays the part of results_k3/, beside it as the sweep arranges them
    report_directory = tmp_path / "reports"
    # Reuse these options so every fitting stage uses the chosen backend
    # and the same small restart budget.
    small_fit_arguments = ("--backend", backend, "--n-starts", "2")

    # Step 3: fit K = 1 to 4 into the main results folder.
    run("03_fit_lca.py", "--results", main_results, *small_fit_arguments, "--k-max", "4")
    assert (main_results / "lca_fits.json").exists() and (main_results / f"lca_k{K}.npz").exists()

    # Steps 4 to 12 against a copy of the K = 3 fit, as scripts/sweep_k.sh does.
    results_for_k.mkdir()
    (results_for_k / f"lca_k{K}.npz").write_bytes((main_results / f"lca_k{K}.npz").read_bytes())
    selected_k_arguments = ("--results", results_for_k, "--k", str(K))
    run("04_crosscheck.py", *selected_k_arguments, *small_fit_arguments, "--method", "kmeans", "--kmeans-seeds", "2")
    run("04_crosscheck.py", *selected_k_arguments, "--backend", backend, "--method", "hierarchical")
    run("04_crosscheck.py", *selected_k_arguments, *small_fit_arguments, "--method", "alternatives")
    run("05_replicate.py", *selected_k_arguments, *small_fit_arguments)
    run("06_bootstrap.py", *selected_k_arguments, *small_fit_arguments, "--n-boot", "3")
    run("07_describe.py", *selected_k_arguments)
    run("08_forward_2025.py", *selected_k_arguments, *small_fit_arguments)
    run("09_age_bands.py", *selected_k_arguments, *small_fit_arguments)
    run("10_parallel_ever.py", *selected_k_arguments, *small_fit_arguments)
    run("11_robustness.py", *selected_k_arguments, *small_fit_arguments, "--seeds", "1,2")
    # --sim-k-max below K on purpose: step 12 must still fit up to K (this crashed at K=7 and 8).
    run("12_dimensions.py", *selected_k_arguments, *small_fit_arguments, "--fits", main_results, "--d-max", "2", "--sim-replicates", "1", "--sim-k-max", "2")
    run("15_replication_calibration.py", *selected_k_arguments, *small_fit_arguments, "--replicates", "1", "--kmeans-starts", "2", "--kmeans-seeds", "1")

    for name in (
        "crosscheck_kmeans.json",
        "method_comparison.csv",
        "replicate.json",
        "stability.csv",
        "severity_test.json",
        "forward_2025.json",
        "age_bands.json",
        "parallel_ever.json",
        "robustness.json",
        "model_comparison.csv",
        "replication_calibration.csv",
    ):
        assert (results_for_k / name).exists(), f"{name} was not written"

    # Step 13 reads the bootstrap from the K folder and writes the K choice to the main results folder.
    run("13_heldout_selection.py", "--results", main_results, *small_fit_arguments, "--k-max", "3", "--d-max", "2")
    assert (main_results / "model_choice.json").exists() and (main_results / "heldout_selection.csv").exists()

    # Step 14: shape and stability by simulation, at the smallest settings that exercise every branch.
    run(
        "14_shape_and_stability.py",
        "--results",
        main_results,
        *small_fit_arguments,
        "--k",
        str(K),
        "--replicates",
        "1",
        "--k-max",
        "3",
        "--control-k",
        "3",
        "--n-boot",
        "2",
    )
    assert (
        (main_results / "latent_density.csv").exists()
        and (main_results / "stability_calibration.csv").exists()
        and (main_results / "positions_2d.npz").exists()
    )

    # Figures, drawn for K = 3 from the temporary folders into a temporary reports folder.
    run("make_results_figures.py", "--results", main_results, "--reports", report_directory, "--k", str(K))
    for figure in ("classes.png", "severity_by_class.png", "checks_by_k.png"):
        assert (report_directory / "figures" / "results" / figure).exists(), f"{figure} was not drawn"
