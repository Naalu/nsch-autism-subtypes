"""Tests for the data steps: reading the config, selecting children, encoding flags, the prevalence rule.

These use tiny hand-made tables, so you can see exactly what goes in and
what should come out.
"""

import numpy as np
import pandas as pd
import pytest

from nsch_subtypes.config import DEFAULT_CONFIG_PATH, Condition, load_config
from nsch_subtypes.data import (
    apply_prevalence_rule,
    build_analysis_table,
    drop_incomplete_rows,
    encode_flag,
    select_children,
    to_flag_matrix,
    topical_archive_name,
)


@pytest.fixture
def config():
    return load_config(DEFAULT_CONFIG_PATH)


def test_config_loads_and_has_what_we_need(config):
    assert config.min_prevalence == 0.01
    assert {"year", "household", "age", "sex", "weight", "stratum", "state", "asd_current", "asd_severity"} <= set(config.keys)
    names = [condition.name for condition in config.conditions]
    assert len(names) == len(set(names)), "condition names must be unique"
    assert "adhd" in [condition.name for condition in config.included]
    assert "allergies" not in [condition.name for condition in config.included]


def test_encode_flag_follows_the_rule_table():
    """One row per case in the rule table in encode_flag's docstring."""
    condition = Condition(name="x", label="X", ever="e", currently="c", years="2016-2025", include=True)
    frame = pd.DataFrame(
        {
            "e": [2, 1, 1, 1, np.nan],  # ever told: no, yes, yes, yes, blank
            "c": [np.nan, 1, 2, np.nan, np.nan],  # currently: skipped, yes, no, blank, blank
        }
    )
    flag = encode_flag(frame, condition).tolist()
    assert flag[0] == 0.0  # never told
    assert flag[1] == 1.0  # ever and currently
    assert flag[2] == 0.0  # ever, but not any more
    assert np.isnan(flag[3])  # ever, follow-up unanswered
    assert np.isnan(flag[4])  # ever unanswered


def test_encode_flag_without_a_followup_item():
    condition = Condition(name="x", label="X", ever="e", currently=None, years="2016-2025", include=True)
    frame = pd.DataFrame({"e": [1, 2, np.nan]})
    flag = encode_flag(frame, condition).tolist()
    assert flag[0] == 1.0 and flag[1] == 0.0 and np.isnan(flag[2])


def test_select_children_counts_each_filter():
    keys = {"age": "age", "asd_ever": "ever", "asd_current": "cur"}
    frame = pd.DataFrame(
        {
            "age": [5, 6, 10, 17, 18],  # 5 and 18 are out of range
            "ever": [1, 1, 1, 2, 1],
            "cur": [1, 1, 2, np.nan, 1],  # only the 6-year-old is in range and currently autistic
        }
    )
    selected, counts = select_children(frame, keys)
    assert counts == {"n_children": 5, "n_age_6_17": 3, "n_asd_ever": 2, "n_asd_current": 1, "n_selected": 1}
    assert selected["age"].tolist() == [6]


def test_prevalence_rule_drops_rare_flags_and_records_everything(config):
    """Build 500 fake children where every included condition is common except cystic fibrosis, which nobody has."""
    rng = np.random.default_rng(0)
    n = 500
    frame = {
        "year": 2020,
        "hhid": np.arange(n),
        "sc_age_years": 10,
        "sc_sex": 1,
        "fwc": 1.0,
        "stratum": "1",
        "fipsst": 4,
        "k2q35a": 1,
        "k2q35b": 1,
        "k2q35c": 1,
    }
    for condition in config.included:
        share_with_it = 0.0 if condition.name == "cystic_fibrosis" else 0.5
        frame[condition.ever] = np.where(rng.random(n) < share_with_it, 1, 2)
        if condition.currently:
            frame[condition.currently] = np.where(frame[condition.ever] == 1, 1, np.nan)

    table = build_analysis_table(pd.DataFrame(frame), config)
    kept, codebook = apply_prevalence_rule(table, config)

    assert "cystic_fibrosis" not in kept and "adhd" in kept
    assert set(codebook["flag"]) == {condition.name for condition in config.conditions}, "codebook lists every condition, kept or not"
    assert codebook.loc[codebook["flag"] == "allergies", "kept"].item() == False  # noqa: E712

    flag_matrix = to_flag_matrix(drop_incomplete_rows(table, kept), kept)
    assert flag_matrix.dtype == np.uint8 and set(np.unique(flag_matrix)) <= {0, 1}


def test_archive_name_handles_census_naming_quirks():
    listing = '<a href="nsch_2023e_screener_SAS.zip">x</a><a href="nsch_2023e_topical_Stata.zip">y</a>'
    assert topical_archive_name(2023, listing) == "nsch_2023e_topical_Stata.zip"
    assert topical_archive_name(2024, '<a href="nsch_2024_topical_Stata.zip">') == "nsch_2024_topical_Stata.zip"
    with pytest.raises(FileNotFoundError):
        topical_archive_name(2030, "")
