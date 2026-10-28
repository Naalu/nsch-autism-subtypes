"""Run every pipeline script once, end to end, at tiny settings.

The unit tests check the building blocks on made-up data. This test checks
the plumbing: that each numbered script still runs against the committed
analysis table, reads what the step before it wrote, and writes what the
step after it expects. It catches the bugs unit tests miss, such as a
renamed column or a JSON key that one script writes and another reads.

The settings are far too small for the numbers to mean anything (two
random starts, three bootstrap resamples). Only "did it run, and did it
write its files" is checked.

It takes about five minutes, so it is skipped by default. Run it with::

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
    """Run one script with the reference backend's libraries and fail the test with its output if it errors."""
    command = [sys.executable, str(REPO / "scripts" / script), *map(str, args)]
    finished = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    if finished.returncode != 0:
        pytest.fail(f"{script} exited with {finished.returncode}\n--- stdout\n{finished.stdout[-3000:]}\n--- stderr\n{finished.stderr[-3000:]}")
    return finished.stdout


@pytest.mark.slow
def test_every_script_runs_at_tiny_settings(tmp_path: Path) -> None:
    main = tmp_path / "smoke"  # plays the part of results/
    per_k = tmp_path / f"smoke_k{K}"  # plays the part of results_k3/, beside it as the sweep arranges them
    reports = tmp_path / "reports"
    tiny = ("--backend", "reference", "--n-starts", "2")

    # Step 3: fit K = 1 to 4 into the main folder.
    run("03_fit_lca.py", "--results", main, *tiny, "--k-max", "4")
    assert (main / "lca_fits.json").exists() and (main / f"lca_k{K}.npz").exists()

    # Steps 4 to 12 against a copy of the K = 3 fit, as scripts/sweep_k.sh does.
    per_k.mkdir()
    (per_k / f"lca_k{K}.npz").write_bytes((main / f"lca_k{K}.npz").read_bytes())
    at_k = ("--results", per_k, "--k", str(K))
    run("04_crosscheck.py", *at_k, *tiny, "--method", "kmeans", "--kmeans-seeds", "2")
    run("04_crosscheck.py", *at_k, "--backend", "reference", "--method", "hierarchical")
    run("04_crosscheck.py", *at_k, *tiny, "--method", "alternatives")
    run("05_replicate.py", *at_k, *tiny)
    run("06_bootstrap.py", *at_k, *tiny, "--n-boot", "3")
    run("07_describe.py", *at_k)
    run("08_forward_2025.py", *at_k, *tiny)
    run("09_age_bands.py", *at_k, *tiny)
    run("10_parallel_ever.py", *at_k, *tiny)
    run("11_robustness.py", *at_k, *tiny, "--seeds", "1,2")
    run("12_dimensions.py", *at_k, *tiny, "--fits", main, "--d-max", "2", "--sim-replicates", "1", "--sim-k-max", "3")

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
    ):
        assert (per_k / name).exists(), f"{name} was not written"

    # Step 13 reads the bootstrap from the K folder and writes the K choice to the main folder.
    run("13_heldout_selection.py", "--results", main, *tiny, "--k-max", "3", "--d-max", "2")
    assert (main / "model_choice.json").exists() and (main / "heldout_selection.csv").exists()

    # Step 14: shape and stability by simulation, at the smallest settings that exercise every branch.
    run("14_shape_and_stability.py", "--results", main, *tiny, "--k", str(K), "--replicates", "1", "--k-max", "3", "--control-k", "3", "--n-boot", "2")
    assert (main / "latent_density.csv").exists() and (main / "stability_calibration.csv").exists() and (main / "positions_2d.npz").exists()

    # Figures, drawn for K = 3 from the temporary folders into a temporary reports folder.
    run("make_results_figures.py", "--results", main, "--reports", reports, "--k", str(K))
    for figure in ("classes.png", "severity_by_class.png", "checks_by_k.png"):
        assert (reports / "figures" / "results" / figure).exists(), f"{figure} was not drawn"
