"""Checks for class matching, input identity, and separate report destinations."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _common import save_lca_fit
from compare_backends import compare_fit_pair, main
from make_results_figures import report_output_directories
from nsch_subtypes.lca import LCAResult


@pytest.fixture
def saved_pair(tmp_path):
    """Two identical models with their class columns deliberately swapped."""
    reference_dir, ours_dir = tmp_path / "reference", tmp_path / "ours"
    reference_dir.mkdir()
    ours_dir.mkdir()
    table = pd.DataFrame({"year": [2020, 2021, 2022], "hhid": [11, 12, 13]})
    reference = LCAResult(
        n_classes=2,
        class_sizes=np.array([0.4, 0.6]),
        condition_rates=np.array([[0.1, 0.8], [0.9, 0.2]]),
        responsibilities=np.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8]]),
        log_likelihood=-10.0,
        n_iter=5,
        converged=True,
    )
    ours = LCAResult(
        n_classes=2,
        class_sizes=reference.class_sizes[::-1],
        condition_rates=reference.condition_rates[::-1],
        responsibilities=reference.responsibilities[:, ::-1],
        log_likelihood=reference.log_likelihood,
        n_iter=6,
        converged=True,
    )
    save_lca_fit(reference_dir, reference, table, ["condition_a", "condition_b"])
    save_lca_fit(ours_dir, ours, table, ["condition_a", "condition_b"])
    return reference_dir, ours_dir


def test_comparison_matches_class_numbers_before_comparing_profiles(saved_pair):
    summary, profiles = compare_fit_pair(*saved_pair, 2)
    assert summary["ari"] == 1
    assert summary["max_absolute_rate_difference"] == 0
    assert summary["bic_difference_ours_minus_reference"] == 0
    assert profiles["matched_ours_class"].tolist() == [1, 1, 0, 0]
    assert np.all(profiles["difference_percentage_points"] == 0)


@pytest.mark.parametrize("field", ["hhid", "flags"])
def test_comparison_rejects_different_children_or_condition_order(saved_pair, field):
    reference_dir, ours_dir = saved_pair
    path = ours_dir / "lca_k2.npz"
    with np.load(path) as archive:
        contents = {name: archive[name] for name in archive.files}
    contents[field] = contents[field][::-1]
    np.savez(path, **contents)
    with pytest.raises(ValueError, match=f"different {field}"):
        compare_fit_pair(reference_dir, ours_dir, 2)


def test_reference_reports_do_not_overwrite_the_main_destinations(tmp_path):
    reports = tmp_path / "reports"
    assert report_output_directories(Path("results"), reports) == (reports / "figures" / "results", reports / "tables")
    assert report_output_directories(Path("results_reference"), reports) == (
        reports / "figures" / "results_reference",
        reports / "tables" / "results_reference",
    )


@pytest.mark.parametrize("existing_report", [False, True])
def test_comparison_outputs_data_and_charts_without_writing_a_report(saved_pair, tmp_path, monkeypatch, existing_report):
    reference_dir, ours_dir = saved_pair
    for backend, directory in (("reference", reference_dir), ("ours", ours_dir)):
        (directory / "lca_fits.json").write_text(json.dumps({"backend": backend, "n_starts": 1, "seed": 0}))

    reports = tmp_path / "reports"
    reports.mkdir()
    report_path = reports / "implementation-comparison.md"
    if existing_report:
        report_path.write_text("Manually written report. Keep this text.")

    monkeypatch.setattr(sys, "argv", ["compare_backends.py", "--reference", str(reference_dir), "--ours", str(ours_dir), "--reports", str(reports), "--k", "2"])
    main()

    # The same command must preserve an existing draft and avoid creating one.
    if existing_report:
        assert report_path.read_text() == "Manually written report. Keep this text."
    else:
        assert not report_path.exists()
    assert (reports / "tables" / "backend_comparison.csv").exists()
    assert (reports / "tables" / "backend_condition_rates.csv").exists()
    for name in ("backend_bic.png", "backend_profiles.png", "backend_rate_differences.png"):
        assert (reports / "figures" / "backend_comparison" / name).exists()
    comparison = json.loads((ours_dir / "backend_comparison.json").read_text())
    assert comparison["profile_k"] == 2
    assert comparison["comparisons"][0]["ari"] == 1
