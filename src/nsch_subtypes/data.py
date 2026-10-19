"""Download, select and encode the NSCH files (pipeline steps 1 and 2).

The Census Bureau publishes one "topical" file per survey year as a Stata
archive. Each archive holds a ``.dta`` data file and a ``.do`` file with the
variable and value labels. We read the ``.dta`` directly with pandas and keep
only the columns named in ``config/conditions.yaml``.

Two loaders are available:

- ``census`` (default): reads the Census ``.dta`` file for one year.
- ``nsch``: uses the ``nsch`` package (NAU-ASD3/nsch-py) to load and harmonize
  the year, then selects the same columns. Requires ``pip install
  "nsch-autism-subtypes[nsch]"``. Treat it as experimental until that package
  publishes a release.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .config import Condition, Config

CENSUS_INDEX = "https://www2.census.gov/programs-surveys/nsch/datasets/{year}/"
ASD_AGE_MIN, ASD_AGE_MAX = 6, 17
YES, NO = 1, 2


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def topical_archive_name(year: int, index_html: str) -> str:
    """Find the topical Stata archive on a year's Census index page.

    Census is not consistent about the file name: most years use
    ``nsch_{year}_topical_Stata.zip`` but 2023 uses ``nsch_2023e_topical_Stata.zip``.
    """
    pattern = rf'href="(nsch_{year}e?_topical_Stata\.zip)"'
    match = re.search(pattern, index_html)
    if not match:
        raise FileNotFoundError(f"No topical Stata archive listed for {year}")
    return match.group(1)


def download_year(year: int, raw_dir: Path, force: bool = False) -> Path:
    """Download and unzip one year's topical file. Returns the ``.dta`` path."""
    year_dir = Path(raw_dir) / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    existing = find_dta(year, raw_dir)
    if existing and not force:
        return existing
    index_url = CENSUS_INDEX.format(year=year)
    with urlopen(index_url, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    name = topical_archive_name(year, html)
    archive = year_dir / name
    with urlopen(index_url + name, timeout=600) as resp, open(archive, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(year_dir)
    dta = find_dta(year, raw_dir)
    if dta is None:
        raise FileNotFoundError(f"Archive for {year} did not contain a .dta file")
    return dta


def find_dta(year: int, raw_dir: Path) -> Path | None:
    hits = sorted(Path(raw_dir, str(year)).glob("*.dta"))
    return hits[0] if hits else None


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #
def read_year(year: int, raw_dir: Path, columns: list[str], loader: str = "census") -> pd.DataFrame:
    """Read one survey year with only the requested columns (lowercase names).

    Columns absent from that year's file are added as all-missing so that
    every year has the same shape.
    """
    if loader == "census":
        df = _read_census(year, raw_dir, columns)
    elif loader == "nsch":
        df = _read_nsch(year, raw_dir, columns)
    else:
        raise ValueError(f"Unknown loader {loader!r}; use 'census' or 'nsch'")
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    df["year"] = year
    return df[columns]


def _read_census(year: int, raw_dir: Path, columns: list[str]) -> pd.DataFrame:
    dta = find_dta(year, raw_dir)
    if dta is None:
        raise FileNotFoundError(f"No .dta file for {year} under {raw_dir}; run scripts/01_select.py --download")
    with pd.read_stata(dta, iterator=True) as reader:
        available = {name.lower(): name for name in reader.variable_labels()}
    use = [available[c] for c in columns if c in available]
    df = pd.read_stata(dta, columns=use, convert_categoricals=False)
    df.columns = [c.lower() for c in df.columns]
    return df


def _read_nsch(year: int, raw_dir: Path, columns: list[str]) -> pd.DataFrame:
    try:
        from nsch import get_clean_data  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("The 'nsch' loader needs the nsch package: pip install 'nsch-autism-subtypes[nsch]'") from exc
    frame = get_clean_data(years=[year], data_dir=str(raw_dir))
    df = frame.to_pandas() if hasattr(frame, "to_pandas") else pd.DataFrame(frame)
    df.columns = [c.lower() for c in df.columns]
    missing = [c for c in columns if c not in df.columns and c != "year"]
    if missing:
        raise KeyError(f"nsch output for {year} lacks columns needed here: {missing}")
    return df


# --------------------------------------------------------------------------- #
# Select (step 1)
# --------------------------------------------------------------------------- #
def select(df: pd.DataFrame, keys: dict[str, str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep children who currently have ASD and are aged 6-17.

    Returns the subset and the counts at each filter, for the selection table.
    """
    age = df[keys["age"]]
    ever = df[keys["asd_ever"]]
    current = df[keys["asd_current"]]
    in_age = age.between(ASD_AGE_MIN, ASD_AGE_MAX)
    counts = {
        "n_children": int(len(df)),
        "n_age_6_17": int(in_age.sum()),
        "n_asd_ever": int((in_age & (ever == YES)).sum()),
        "n_asd_current": int((in_age & (current == YES)).sum()),
    }
    out = df.loc[in_age & (current == YES)].copy()
    counts["n_selected"] = int(len(out))
    return out, counts


# --------------------------------------------------------------------------- #
# Encode (step 2)
# --------------------------------------------------------------------------- #
def encode_flag(df: pd.DataFrame, cond: Condition) -> pd.Series:
    """Build one 0/1 flag with missing where the parent did not answer.

    Rule:
      ever == no                      -> 0
      ever == yes and currently == yes -> 1
      ever == yes and currently == no  -> 0
      ever == yes and no currently item (lifelong conditions) -> 1
      anything else                    -> missing
    """
    ever = df[cond.ever]
    if cond.currently is None:
        values = np.where(ever == YES, 1.0, np.where(ever == NO, 0.0, np.nan))
    else:
        curr = df[cond.currently]
        values = np.where(
            ever == NO,
            0.0,
            np.where((ever == YES) & (curr == YES), 1.0, np.where((ever == YES) & (curr == NO), 0.0, np.nan)),
        )
    return pd.Series(values, index=df.index, name=cond.name)


def encode(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Turn selected rows into the analysis table: keys plus one column per flag."""
    k = config.keys
    out = pd.DataFrame(
        {
            "year": df[k["year"]].astype(int),
            "hhid": df[k["household"]].astype("int64"),
            "age": df[k["age"]].astype(int),
            "sex": df[k["sex"]].map({1: "male", 2: "female"}),
            "weight": df[k["weight"]].astype(float),
            "stratum": df[k["stratum"]].astype(str),
            "state": df[k["state"]].astype(int),
            "severity": df[k["asd_severity"]].map({1: "mild", 2: "moderate", 3: "severe"}),
        },
        index=df.index,
    )
    for cond in config.included:
        out[cond.name] = encode_flag(df, cond)
    return out.reset_index(drop=True)


def apply_prevalence_rule(table: pd.DataFrame, config: Config) -> tuple[list[str], pd.DataFrame]:
    """Decide which flags stay, and build the codebook.

    A flag stays if its prevalence in the pooled table is at least
    ``config.min_prevalence``. The codebook lists every configured item,
    including the ones excluded before this step, with the reason.
    """
    rows = []
    kept = []
    for cond in config.conditions:
        row = {
            "flag": cond.name,
            "label": cond.label,
            "ever_item": cond.ever,
            "currently_item": cond.currently or "",
            "years_asked": cond.years,
        }
        if not cond.include:
            row.update(kept=False, prevalence=np.nan, missing=np.nan, reason=cond.reason or "")
        else:
            s = table[cond.name]
            prev, miss = float(s.mean()), float(s.isna().mean())
            keep = prev >= config.min_prevalence
            row.update(
                kept=keep,
                prevalence=round(prev, 4),
                missing=round(miss, 4),
                reason="" if keep else f"prevalence {prev:.2%} below {config.min_prevalence:.0%}",
            )
            if keep:
                kept.append(cond.name)
        if cond.note:
            row["note"] = cond.note
        rows.append(row)
    codebook = pd.DataFrame(rows)
    codebook["note"] = codebook.get("note", "").fillna("") if "note" in codebook else ""
    return kept, codebook


def complete_rows(table: pd.DataFrame, flags: list[str]) -> pd.DataFrame:
    """Rows with every retained flag answered. The models need complete rows."""
    return table.dropna(subset=flags).reset_index(drop=True)


def flag_matrix(table: pd.DataFrame, flags: list[str]) -> np.ndarray:
    """The 0/1 matrix the clustering methods take (children x flags, uint8)."""
    return table[flags].to_numpy(dtype=np.uint8)
