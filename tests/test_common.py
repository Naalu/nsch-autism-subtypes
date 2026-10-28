"""The shared script helpers: reading arguments, saving JSON, and loading a saved fit safely."""

import argparse
import json

import numpy as np
import pandas as pd
import pytest

from _common import load_lca_fit, parse_years, resolve_k, save_json, save_lca_fit
from nsch_subtypes.lca import LCAResult


def test_parse_years_accepts_ranges_lists_and_mixtures():
    assert parse_years("2016-2018") == [2016, 2017, 2018]
    assert parse_years("2016,2020") == [2016, 2020]
    assert parse_years("2016-2017,2025") == [2016, 2017, 2025]


def test_save_json_handles_numpy_values_and_writes_nan_as_null(tmp_path):
    path = tmp_path / "out.json"
    save_json({"pass": np.bool_(True), "n": np.int64(3), "x": np.float32(0.5), "missing": float("nan"), np.int64(4): np.arange(2)}, path)
    loaded = json.loads(path.read_text())
    assert loaded == {"pass": True, "n": 3, "x": 0.5, "missing": None, "4": [0, 1]}


def test_resolve_k_prefers_the_command_line_then_the_file_on_record(tmp_path):
    assert resolve_k(argparse.Namespace(k=5), results_dir=tmp_path) == 5
    with pytest.raises(SystemExit):
        resolve_k(argparse.Namespace(k=None), results_dir=tmp_path)
    (tmp_path / "model_choice.json").write_text(json.dumps({"chosen_k": 4}))
    assert resolve_k(argparse.Namespace(k=None), results_dir=tmp_path) == 4


def _tiny_fit_and_table():
    table = pd.DataFrame({"year": [2016, 2016, 2017], "hhid": [11, 12, 13], "adhd": [1, 0, 1]})
    fit = LCAResult(
        n_classes=2,
        class_sizes=np.array([0.6, 0.4]),
        condition_rates=np.array([[0.9], [0.1]]),
        responsibilities=np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]),
        log_likelihood=-1.5,
        n_iter=7,
        converged=True,
    )
    return fit, table


def test_a_saved_fit_loads_back_unchanged(tmp_path):
    fit, table = _tiny_fit_and_table()
    save_lca_fit(tmp_path, fit, table, ["adhd"])
    loaded = load_lca_fit(tmp_path, 2, table)
    assert np.allclose(loaded.responsibilities, fit.responsibilities)
    assert loaded.log_likelihood == -1.5 and loaded.n_iter == 7 and loaded.converged


def test_loading_a_fit_against_different_children_stops(tmp_path):
    """If the analysis table changed since step 3, pairing old probabilities with new rows would be silently wrong."""
    fit, table = _tiny_fit_and_table()
    save_lca_fit(tmp_path, fit, table, ["adhd"])
    reordered = table.iloc[[1, 0, 2]].reset_index(drop=True)
    with pytest.raises(SystemExit):
        load_lca_fit(tmp_path, 2, reordered)
    with pytest.raises(SystemExit):
        load_lca_fit(tmp_path, 2, table, ["adhd", "anxiety"])  # a condition was added since the fit
    with pytest.raises(SystemExit):
        load_lca_fit(tmp_path, 3)  # no K=3 file
