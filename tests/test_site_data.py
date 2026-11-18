"""The website's data: the shaping functions, and the committed files against the committed tables."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nsch_subtypes.site_data import (
    IDENTIFYING_KEYS,
    MIN_CELL,
    assert_no_identifiers,
    axis_alignment,
    children_columns,
    distinct_profiles,
    order_by_size,
    outcome_surface,
    per_profile,
    profile_strings,
    ranges_by,
    rounded,
    share_with_interval,
)

REPO = Path(__file__).resolve().parents[1]
SITE_DATA = REPO / "site" / "static" / "data"
TABLES = REPO / "reports" / "tables"
FLAGS = ["adhd", "anxiety", "depression", "behavior", "dev_delay", "intellectual", "speech", "learning"]


def load(name: str) -> dict:
    return json.loads((SITE_DATA / name).read_text())


# ---------------------------------------------------------------------------- #
# The shaping functions
# ---------------------------------------------------------------------------- #
def test_identifying_fields_are_caught_at_any_depth():
    assert_no_identifiers({"x": [1, 2], "nested": [{"year": 2020}]})
    for key in IDENTIFYING_KEYS:
        with pytest.raises(ValueError, match=key):
            assert_no_identifiers({"rows": [{"fine": 1, key: 2}]})
    with pytest.raises(ValueError):
        assert_no_identifiers({"HHID": 1})


def test_rounded_gives_plain_json_values():
    out = rounded({"a": np.float64(1.23456789), "b": np.array([1, 2]), "c": float("nan"), "d": np.bool_(True), 3: [np.int64(4)]}, digits=3)
    assert out == {"a": 1.235, "b": [1, 2], "c": None, "d": True, "3": [4]}
    json.dumps(out, allow_nan=False)


def test_axis_alignment_orders_and_orients_the_axes():
    # Dimension 0 is psychiatric and reversed; dimension 1 is developmental.
    slopes = np.zeros((8, 2))
    slopes[:4, 0] = -2.0
    slopes[4:, 1] = 3.0
    alignment = axis_alignment(slopes, FLAGS)
    assert np.array_equal(alignment, [[0.0, -1.0], [1.0, 0.0]])
    high_psychiatric, high_developmental = np.array([[-1.0, 0.0]]), np.array([[0.0, 1.0]])
    assert np.array_equal(high_psychiatric @ alignment, [[0.0, 1.0]])  # up
    assert np.array_equal(high_developmental @ alignment, [[1.0, 0.0]])  # across


def test_profiles_are_distinct_counted_and_largest_first():
    X = np.array([[1, 0], [0, 0], [1, 0], [1, 1], [1, 0]])
    profiles, profile_of_child, counts = distinct_profiles(X)
    assert counts.tolist() == [3, 1, 1] and counts.sum() == len(X)
    assert np.array_equal(profiles[profile_of_child], X)
    assert profile_strings(profiles)[0] == "10"


def test_per_profile_refuses_values_that_vary_within_a_profile():
    X = np.array([[1, 0], [0, 0], [1, 0]])
    _, profile_of_child, _ = distinct_profiles(X)
    assert per_profile(np.array([5, 7, 5]), profile_of_child, 2).tolist() == [5, 7]
    with pytest.raises(ValueError):
        per_profile(np.array([5, 7, 6]), profile_of_child, 2)


def test_children_are_shuffled_but_stay_whole():
    n = 200
    positions = np.column_stack([np.arange(n), -np.arange(n)]).astype(float)
    columns = children_columns(positions, positions + 0.5, np.arange(n) % 7, np.repeat([2016, 2017], n // 2), np.tile([8, 15], n // 2), seed=1)
    assert columns["x"] != sorted(columns["x"]), "row order is the survey order"
    assert sorted(columns["x"]) == list(range(n))
    for x, y, sharp_x, profile, older in zip(columns["x"], columns["y"], columns["sharp_x"], columns["profile"], columns["older"], strict=True):
        assert y == -x and sharp_x == x + 0.5 and profile == x % 7 and older == int(x % 2 == 1)
    assert set(columns) == {"x", "y", "sharp_x", "sharp_y", "profile", "year", "older"}


def test_order_by_size():
    assert order_by_size(np.array([2, 2, 2, 0, 1, 1]), 3).tolist() == [0, 0, 0, 2, 1, 1]


def test_outcome_surface_leaves_out_small_cells_and_unanswered_children():
    rng = np.random.default_rng(0)
    positions = np.vstack([rng.uniform(0.01, 0.49, size=(50, 2)), rng.uniform(1.01, 1.49, size=(MIN_CELL - 1, 2))])
    has = np.zeros(len(positions), dtype=bool)
    has[:10] = True
    answered = np.ones(len(positions), dtype=bool)
    answered[40:50] = False
    cells = outcome_surface(positions, has, answered)
    assert len(cells) == 1, "the cell with fewer than the minimum is published"
    assert cells[0] == {"x": 0.25, "y": 0.25, "n": 40, "share": 0.25}


def test_share_with_interval_and_ranges():
    interval = share_with_interval(np.array([1] * 30 + [0] * 70))
    assert interval["n"] == 100 and interval["share"] == 0.3 and interval["low"] < 0.3 < interval["high"]
    rows = [{"data": "a", "k": 2, "v": 1.0}, {"data": "a", "k": 2, "v": 3.0}, {"data": "b", "k": 2, "v": 5.0}]
    assert ranges_by(rows, ("data", "k"), "v") == [{"data": "a", "k": 2, "min": 1.0, "max": 3.0, "n": 2}, {"data": "b", "k": 2, "min": 5.0, "max": 5.0, "n": 1}]


# ---------------------------------------------------------------------------- #
# The committed files
# ---------------------------------------------------------------------------- #
def test_no_committed_file_carries_an_identifier():
    files = sorted(SITE_DATA.glob("*.json"))
    assert len(files) == 8
    for path in files:
        assert_no_identifiers(json.loads(path.read_text()))


def test_children_and_profiles_agree_with_the_analysis():
    children, profiles = load("children.json"), load("profiles.json")
    summary = pd.read_csv(TABLES / "summary.csv", header=None, index_col=0)[1]
    n = int(summary["n_main_complete"])
    assert children["n"] == n
    assert all(len(children[column]) == n for column in ("x", "y", "sharp_x", "sharp_y", "profile", "year", "older"))
    assert sum(profiles["count"]) == n
    assert np.array_equal(np.bincount(children["profile"]), profiles["count"]), "profile counts match the children"
    assert profiles["count"] == sorted(profiles["count"], reverse=True)
    assert len(set(profiles["answers"])) == len(profiles["answers"]) and all(set(answers) <= {"0", "1"} for answers in profiles["answers"])
    assert children["year"] != sorted(children["year"]), "children are in survey order"
    assert set(children["year"]) == set(range(2016, 2025)) and set(children["older"]) == {0, 1}
    assert max(map(abs, children["x"] + children["y"] + children["sharp_x"] + children["sharp_y"])) <= 4.5


def test_group_labels_and_probabilities_are_consistent():
    children, profiles, model = load("children.json"), load("profiles.json"), load("model.json")
    probabilities = np.array(profiles["class_probabilities"])
    assert probabilities.shape == (len(profiles["count"]), 4) and np.allclose(probabilities.sum(axis=1), 1.0, atol=0.002)
    assert np.array_equal(probabilities.argmax(axis=1), profiles["label_by_k"]["4"]), "the K = 4 label is the most likely class"
    for k in range(2, 9):
        assert set(profiles["label_by_k"][str(k)]) == set(range(k))
    of_child = np.array(children["profile"])
    for by_k in model["by_k"]:
        shares = np.bincount(np.array(profiles["label_by_k"][str(by_k["k"])])[of_child], minlength=by_k["k"]) / children["n"]
        assert np.allclose(shares, by_k["shares"], atol=1e-4)
    assert [c["name"] for c in model["classes"]] == ["fewer co-occurring conditions", "developmental-language", "psychiatric", "developmental and psychiatric"]
    centers = np.array([c["center"] for c in model["classes"]])
    assert centers[1, 0] > centers[0, 0] and centers[2, 1] > centers[0, 1], "developmental is across and psychiatric is up"
    assert len(model["conditions"]) == 18 and np.array(model["trait"]["slopes"]).shape == (18, 2)


def test_quoted_numbers_match_the_tables():
    numbers = {key: entry["value"] for key, entry in load("numbers.json").items()}
    checks = pd.read_csv(TABLES / "checks_by_k.csv").set_index("K").loc[4]
    assert numbers["ari_kmeans"] == pytest.approx(checks["second method (k-means)"], abs=1e-4)
    assert numbers["ari_heldout_years"] == pytest.approx(checks["held-out years"], abs=1e-4)
    assert numbers["ari_2025"] == pytest.approx(checks["2025 forward test"], abs=1e-4)
    stability = pd.read_csv(TABLES / "stability_by_k.csv").set_index("k")
    assert numbers["least_stable_jaccard_k4"] == pytest.approx(stability.loc[4, "min_jaccard"], abs=1e-4)
    mixture = pd.read_csv(TABLES / "factor_mixture.csv")
    two = mixture[(mixture["version"] == "shared spread") & (mixture["groups"] == 2)].iloc[0]
    assert numbers["mixture_bic_two_groups"] == pytest.approx(two["bic_vs_one_group"], abs=1e-3)
    # The export reruns k-means; it must land on the solution step 4 reported.
    assert load("model.json")["kmeans"]["ari"] == pytest.approx(checks["second method (k-means)"], abs=2e-3)
    assert numbers["children"] == load("children.json")["n"] and numbers["distinct_profiles"] == len(load("profiles.json")["count"])


def test_outcomes_are_published_only_in_aggregate():
    outcomes = load("outcomes.json")["outcomes"]
    assert set(outcomes) == {"severity", "forgone_care", "er_visits"}
    for outcome in outcomes.values():
        assert outcome["surface"] and all(cell["n"] >= MIN_CELL and 0 <= cell["share"] <= 1 for cell in outcome["surface"])
        assert len(outcome["by_class"]) == 4 and all(group["n"] > 500 for group in outcome["by_class"])


def test_clouds_and_evidence_have_what_the_figures_need():
    clouds, evidence = load("clouds.json"), load("evidence.json")
    assert set(clouds) == {"real", "continuum", "classes"}
    for one in clouds.values():
        assert len(one["x"]) == len(one["y"]) == len(one["sharp_x"]) == len(one["sharp_y"]) == 2500
    # The measured order behind the side-by-side comparison: no groups < real < four classes.
    assert clouds["continuum"]["gain_over_bell_curve"] < clouds["real"]["gain_over_bell_curve"] < clouds["classes"]["gain_over_bell_curve"]
    assert {row["data"] for row in evidence["replication"]["simulated"]} == {"continuum", "4 classes"}
    assert len(evidence["replication"]["real"]) >= 4 and len(evidence["mixture"]["calibration"]) == 33
