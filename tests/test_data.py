"""Selection and encoding rules on small hand-made frames."""

import numpy as np
import pandas as pd
import pytest

from nsch_subtypes.config import DEFAULT_CONFIG, Condition, load_config
from nsch_subtypes.data import apply_prevalence_rule, complete_rows, encode, encode_flag, flag_matrix, select, topical_archive_name


@pytest.fixture
def config():
    return load_config(DEFAULT_CONFIG)


def test_config_loads_and_keys_present(config):
    assert config.min_prevalence == 0.01
    assert {"year", "household", "age", "sex", "weight", "stratum", "state", "asd_current", "asd_severity"} <= set(config.keys)
    names = [c.name for c in config.conditions]
    assert len(names) == len(set(names))
    assert "adhd" in [c.name for c in config.included]
    assert "allergies" not in [c.name for c in config.included]


def test_encode_flag_rules():
    cond = Condition(name="x", label="X", ever="e", currently="c", years="2016-2025", include=True)
    df = pd.DataFrame({"e": [2, 1, 1, 1, np.nan], "c": [np.nan, 1, 2, np.nan, np.nan]})
    out = encode_flag(df, cond).tolist()
    assert out[0] == 0.0  # never told
    assert out[1] == 1.0  # ever and currently
    assert out[2] == 0.0  # ever but not currently
    assert np.isnan(out[3])  # ever, follow-up unanswered
    assert np.isnan(out[4])  # ever unanswered


def test_encode_flag_without_followup():
    cond = Condition(name="x", label="X", ever="e", currently=None, years="2016-2025", include=True)
    df = pd.DataFrame({"e": [1, 2, np.nan]})
    out = encode_flag(df, cond).tolist()
    assert out[0] == 1.0 and out[1] == 0.0 and np.isnan(out[2])


def test_select_counts():
    keys = {"age": "age", "asd_ever": "ever", "asd_current": "cur"}
    df = pd.DataFrame({"age": [5, 6, 10, 17, 18], "ever": [1, 1, 1, 2, 1], "cur": [1, 1, 2, np.nan, 1]})
    out, counts = select(df, keys)
    assert counts == {"n_children": 5, "n_age_6_17": 3, "n_asd_ever": 2, "n_asd_current": 1, "n_selected": 1}
    assert out["age"].tolist() == [6]


def test_prevalence_rule_and_codebook(config):
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
    for cond in config.included:
        p = 0.5 if cond.name != "cystic_fibrosis" else 0.0
        frame[cond.ever] = np.where(rng.random(n) < p, 1, 2)
        if cond.currently:
            frame[cond.currently] = np.where(frame[cond.ever] == 1, 1, np.nan)
    table = encode(pd.DataFrame(frame), config)
    kept, codebook = apply_prevalence_rule(table, config)
    assert "cystic_fibrosis" not in kept and "adhd" in kept
    assert set(codebook["flag"]) == {c.name for c in config.conditions}
    assert codebook.loc[codebook["flag"] == "allergies", "kept"].item() == False  # noqa: E712
    X = flag_matrix(complete_rows(table, kept), kept)
    assert X.dtype == np.uint8 and set(np.unique(X)) <= {0, 1}


def test_archive_name_handles_census_naming():
    html = '<a href="nsch_2023e_screener_SAS.zip">x</a><a href="nsch_2023e_topical_Stata.zip">y</a>'
    assert topical_archive_name(2023, html) == "nsch_2023e_topical_Stata.zip"
    assert topical_archive_name(2024, '<a href="nsch_2024_topical_Stata.zip">') == "nsch_2024_topical_Stata.zip"
    with pytest.raises(FileNotFoundError):
        topical_archive_name(2030, "")
