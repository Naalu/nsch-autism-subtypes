"""Get the survey data into shape: download, pick the children we study, turn answers into 0/1 flags.

This module backs pipeline steps 1 and 2 (``scripts/01_select.py`` and
``scripts/02_encode.py``). If you are new to the project, read it top to
bottom; it is the part that decides what the clustering methods see.

About the data
--------------
The National Survey of Children's Health (NSCH) is a yearly survey run by the
U.S. Census Bureau. A parent or caregiver answers a long questionnaire about
one child in the household. We use the public files for 2016 through 2025,
one file per year, each with 20,000 to 55,000 children.

Each year ships as a Stata archive (a ``.zip`` holding a ``.dta`` data file
and a ``.do`` file with variable labels). We read the ``.dta`` with pandas and
keep only the columns listed in ``config/conditions.yaml``.

Two ways to read a year are offered:

- ``census`` (default): read the Census ``.dta`` file directly.
- ``nsch``: use the ``nsch`` package (NAU-ASD3/nsch-py), which harmonizes the
  years for you. Needs ``pip install "nsch-autism-subtypes[nsch]"`` and is
  experimental until that package has a release.

How the survey codes answers
----------------------------
Yes/no questions are coded ``1`` for yes and ``2`` for no (not 1 and 0). The
condition questions come in pairs: "Has a doctor ever told you this child
has X?" and, only if yes, "Does the child currently have X?" A blank in the
follow-up therefore usually means the question was skipped, not refused.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .config import Condition, Config

# Census posts each year's files under a URL of this shape.
CENSUS_INDEX_URL = "https://www2.census.gov/programs-surveys/nsch/datasets/{year}/"

# We study children aged 6 through 17. Younger children get a different
# questionnaire and many conditions are not yet diagnosed.
AGE_MIN, AGE_MAX = 6, 17

# NSCH coding for yes/no items.
YES, NO = 1, 2


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #
def topical_archive_name(year: int, index_html: str) -> str:
    """Find the name of the Stata archive on a year's Census listing page.

    Census is not consistent about file names. Most years use
    ``nsch_{year}_topical_Stata.zip`` but 2023 is ``nsch_2023e_topical_Stata.zip``
    (the "e" marks a re-release). So instead of guessing the name we read the
    listing page and match either form.
    """
    pattern = rf'href="(nsch_{year}e?_topical_Stata\.zip)"'
    match = re.search(pattern, index_html)
    if not match:
        raise FileNotFoundError(f"No topical Stata archive listed for {year}")
    return match.group(1)


def find_dta(year: int, raw_dir: Path) -> Path | None:
    """Return the ``.dta`` file for ``year`` under ``raw_dir/{year}/``, or ``None`` if it is not there."""
    hits = sorted(Path(raw_dir, str(year)).glob("*.dta"))
    return hits[0] if hits else None


def download_year(year: int, raw_dir: Path, force: bool = False) -> Path:
    """Download and unzip one survey year. Returns the path of the ``.dta`` file.

    Skips the download if the file is already present unless ``force`` is set.
    Files are 5 to 7 MB zipped and 10 to 30 MB unzipped.
    """
    year_dir = Path(raw_dir) / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    existing = find_dta(year, raw_dir)
    if existing and not force:
        return existing

    # Step 1: read the listing page to learn the archive's exact name.
    index_url = CENSUS_INDEX_URL.format(year=year)
    with urlopen(index_url, timeout=60) as response:
        listing_html = response.read().decode("utf-8", errors="replace")
    archive_name = topical_archive_name(year, listing_html)

    # Step 2: stream the archive to disk one megabyte at a time.
    archive_path = year_dir / archive_name
    with urlopen(index_url + archive_name, timeout=600) as response, open(archive_path, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)

    # Step 3: unzip and confirm a .dta came out.
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(year_dir)
    dta_path = find_dta(year, raw_dir)
    if dta_path is None:
        raise FileNotFoundError(f"Archive for {year} did not contain a .dta file")
    return dta_path


# --------------------------------------------------------------------------- #
# Reading one year
# --------------------------------------------------------------------------- #
def read_year(year: int, raw_dir: Path, columns: list[str], loader: str = "census") -> pd.DataFrame:
    """Read one survey year, keeping only ``columns`` (lowercase names).

    Any requested column that does not exist in that year's file is added as
    all-missing, so every year comes back with the same columns in the same
    order. A ``year`` column is filled in as well.
    """
    if loader == "census":
        frame = _read_census_file(year, raw_dir, columns)
    elif loader == "nsch":
        frame = _read_with_nsch_package(year, raw_dir, columns)
    else:
        raise ValueError(f"Unknown loader {loader!r}; use 'census' or 'nsch'")

    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["year"] = year
    return frame[columns]


def _read_census_file(year: int, raw_dir: Path, columns: list[str]) -> pd.DataFrame:
    """Read the requested columns from the Census ``.dta`` file for one year."""
    dta_path = find_dta(year, raw_dir)
    if dta_path is None:
        raise FileNotFoundError(f"No .dta file for {year} under {raw_dir}; run scripts/01_select.py --download")

    # Variable names differ in case between years (K2Q31A vs k2q31a), so we
    # look up the file's own spelling for each lowercase name we want.
    with pd.read_stata(dta_path, iterator=True) as reader:
        name_in_file = {name.lower(): name for name in reader.variable_labels()}
    columns_to_read = [name_in_file[column] for column in columns if column in name_in_file]

    # convert_categoricals=False keeps the numeric codes (1 = yes, 2 = no)
    # instead of turning them into text labels.
    frame = pd.read_stata(dta_path, columns=columns_to_read, convert_categoricals=False)
    frame.columns = [column.lower() for column in frame.columns]
    return frame


def _read_with_nsch_package(year: int, raw_dir: Path, columns: list[str]) -> pd.DataFrame:
    """Read one year through the ``nsch`` package instead of the raw file."""
    try:
        from nsch import get_clean_data  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("The 'nsch' loader needs the nsch package: pip install 'nsch-autism-subtypes[nsch]'") from exc

    frame = get_clean_data(years=[year], data_dir=str(raw_dir))
    # The package may return a polars frame; make sure we have pandas.
    frame = frame.to_pandas() if hasattr(frame, "to_pandas") else pd.DataFrame(frame)
    frame.columns = [column.lower() for column in frame.columns]

    missing = [column for column in columns if column not in frame.columns and column != "year"]
    if missing:
        raise KeyError(f"nsch output for {year} lacks columns needed here: {missing}")
    return frame


# --------------------------------------------------------------------------- #
# Step 1: choose the children
# --------------------------------------------------------------------------- #
def select_children(frame: pd.DataFrame, keys: dict[str, str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep children aged 6 to 17 whose parent says they currently have autism.

    Two filters, applied together:

    1. age between ``AGE_MIN`` and ``AGE_MAX`` inclusive;
    2. the "currently has autism" item (``k2q35b``) equals yes.

    We use "currently" rather than "ever told" so that children whose early
    diagnosis was later ruled out are not included.

    Returns the selected rows and a dictionary of counts at each stage, which
    becomes one row of ``data/processed/selection_counts.csv``.
    """
    age = frame[keys["age"]]
    ever_told_autism = frame[keys["asd_ever"]]
    currently_autism = frame[keys["asd_current"]]

    in_age_range = age.between(AGE_MIN, AGE_MAX)
    counts = {
        "n_children": int(len(frame)),
        "n_age_6_17": int(in_age_range.sum()),
        "n_asd_ever": int((in_age_range & (ever_told_autism == YES)).sum()),
        "n_asd_current": int((in_age_range & (currently_autism == YES)).sum()),
    }
    selected = frame.loc[in_age_range & (currently_autism == YES)].copy()
    counts["n_selected"] = int(len(selected))
    return selected, counts


# --------------------------------------------------------------------------- #
# Step 2: turn answers into flags
# --------------------------------------------------------------------------- #
def encode_flag(frame: pd.DataFrame, condition: Condition) -> pd.Series:
    """Build one 0/1 flag for a condition, with missing (NaN) where the parent did not answer.

    The rule, in words: a child is flagged ``1`` if the parent says the child
    currently has the condition, ``0`` if the parent says no (either never told,
    or told but no longer has it), and missing otherwise.

    ========================  =======================  ======
    "ever told" answer        "currently" answer       flag
    ========================  =======================  ======
    no                        (skipped)                0
    yes                       yes                      1
    yes                       no                       0
    yes                       (no follow-up item)      1
    anything else             anything else            NaN
    ========================  =======================  ======

    The last row covers refusals and blanks. We keep them as missing rather
    than guessing, and the clustering steps later drop children with any
    missing flag.
    """
    ever = frame[condition.ever]

    if condition.currently is None:
        # Lifelong conditions (Down syndrome, cerebral palsy, ...) have no
        # follow-up question, so "ever told" is the whole answer.
        values = np.where(ever == YES, 1.0, np.where(ever == NO, 0.0, np.nan))
    else:
        currently = frame[condition.currently]
        values = np.where(
            ever == NO,
            0.0,
            np.where(
                (ever == YES) & (currently == YES),
                1.0,
                np.where((ever == YES) & (currently == NO), 0.0, np.nan),
            ),
        )
    return pd.Series(values, index=frame.index, name=condition.name)


def build_analysis_table(selected: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Turn selected survey rows into the analysis table: one row per child, readable columns, one flag per condition.

    The output columns are:

    ``year, hhid, age, sex, weight, stratum, state, severity`` and then one
    0/1/NaN column per included condition (``adhd``, ``anxiety``, ...).

    ``severity`` is the parent's rating of how severe the child's autism is
    (mild, moderate, severe). It is never used for clustering; it is held
    back so we can test whether the clusters tell us anything about it.
    ``weight``, ``stratum`` and ``state`` describe how the survey was sampled
    and are used only in step 7 to turn sample counts into national shares.
    """
    key = config.keys
    table = pd.DataFrame(
        {
            "year": selected[key["year"]].astype(int),
            "hhid": selected[key["household"]].astype("int64"),
            "age": selected[key["age"]].astype(int),
            "sex": selected[key["sex"]].map({1: "male", 2: "female"}),
            "weight": selected[key["weight"]].astype(float),
            "stratum": selected[key["stratum"]].astype(str),
            "state": selected[key["state"]].astype(int),
            "severity": selected[key["asd_severity"]].map({1: "mild", 2: "moderate", 3: "severe"}),
        },
        index=selected.index,
    )
    for condition in config.included:
        table[condition.name] = encode_flag(selected, condition)
    # The "ever told" answer alone, kept for the parallel-items check (step 10):
    # 1 = yes, 0 = no, NaN = no answer. Never used to fit the main model.
    for condition in config.included:
        ever = selected[condition.ever]
        table[f"ever_{condition.name}"] = np.where(ever == YES, 1.0, np.where(ever == NO, 0.0, np.nan))
    return table.reset_index(drop=True)


def apply_prevalence_rule(table: pd.DataFrame, config: Config) -> tuple[list[str], pd.DataFrame]:
    """Decide which flags are common enough to keep, and write down every decision in a codebook.

    A flag is kept if at least ``config.min_prevalence`` (1% by default) of the
    children in ``table`` have it. The returned codebook has one row for every
    condition in the config, including those excluded before this step, with
    the prevalence, the share missing, and the reason for any exclusion. It is
    saved as ``data/processed/codebook.csv`` so readers can see every choice.
    """
    codebook_rows = []
    kept_flags = []
    for condition in config.conditions:
        row = {
            "flag": condition.name,
            "label": condition.label,
            "ever_item": condition.ever,
            "currently_item": condition.currently or "",
            "years_asked": condition.years,
        }
        if not condition.include:
            # Excluded in the config (wording change, not asked every year, ...).
            row.update(kept=False, prevalence=np.nan, missing=np.nan, reason=condition.reason or "")
        else:
            flag = table[condition.name]
            prevalence = float(flag.mean())  # mean of a 0/1 column is the share with the condition
            share_missing = float(flag.isna().mean())
            keep = prevalence >= config.min_prevalence
            row.update(
                kept=keep,
                prevalence=round(prevalence, 4),
                missing=round(share_missing, 4),
                reason="" if keep else f"prevalence {prevalence:.2%} below {config.min_prevalence:.0%}",
            )
            if keep:
                kept_flags.append(condition.name)
        if condition.note:
            row["note"] = condition.note
        codebook_rows.append(row)

    codebook = pd.DataFrame(codebook_rows)
    if "note" in codebook:
        codebook["note"] = codebook["note"].fillna("")
    else:
        codebook["note"] = ""
    return kept_flags, codebook


def drop_incomplete_rows(table: pd.DataFrame, flags: list[str]) -> pd.DataFrame:
    """Keep only children with an answer for every kept flag. The clustering methods cannot handle missing values."""
    return table.dropna(subset=flags).reset_index(drop=True)


def to_flag_matrix(table: pd.DataFrame, flags: list[str]) -> np.ndarray:
    """Pull the flag columns out as a plain NumPy array of 0s and 1s (children in rows, conditions in columns).

    This array is what every clustering method takes as input. ``uint8`` is
    the smallest integer type and is plenty for 0/1 values.
    """
    return table[flags].to_numpy(dtype=np.uint8)
