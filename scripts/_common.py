"""Small helpers shared by the numbered scripts: where files live, command-line flags, loading, saving.

The numbered scripts are meant to be read in order (01 to 08). Each one is
short and does one job; the shared bits are here so the scripts stay short.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------- #
# Where things live
# ---------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[1]
RAW_DIR = REPO / "data" / "raw"  # Census files as downloaded, one folder per year (not committed)
INTERIM_DIR = REPO / "data" / "interim"  # selected rows per year, raw column names (not committed)
PROCESSED_DIR = REPO / "data" / "processed"  # the analysis table and codebook (committed)
RESULTS_DIR = REPO / "results"  # model fits, check results, figures (not committed)

ANALYSIS_PATH = PROCESSED_DIR / "analysis.parquet"
CODEBOOK_PATH = PROCESSED_DIR / "codebook.csv"

# The pooled analysis uses 2016 through 2024. The 2025 file is held out for
# the forward test in step 8 and is never used to fit or choose anything.
MAIN_YEARS = range(2016, 2025)


# ---------------------------------------------------------------------------- #
# Command-line arguments
# ---------------------------------------------------------------------------- #
def parse_years(text: str) -> list[int]:
    """Turn ``"2016-2024"`` or ``"2016,2018,2020"`` into a list of ints."""
    years: list[int] = []
    for part in text.split(","):
        if "-" in part:
            first, last = part.split("-")
            years.extend(range(int(first), int(last) + 1))
        else:
            years.append(int(part))
    return years


def make_parser(description: str, backend: bool = True) -> argparse.ArgumentParser:
    """An argument parser with the flags every script shares.

    ``--results`` says where to write; ``--seed`` makes random steps repeatable;
    ``--backend`` (when the script clusters anything) picks the from-scratch
    code or the library versions.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR, help="directory for outputs")
    parser.add_argument("--seed", type=int, default=0, help="random seed, so runs can be repeated")
    if backend:
        parser.add_argument(
            "--backend",
            choices=["ours", "reference"],
            default="ours",
            help="'ours' = the from-scratch code; 'reference' = StepMix / SciPy / scikit-learn",
        )
    return parser


# ---------------------------------------------------------------------------- #
# Loading the prepared data
# ---------------------------------------------------------------------------- #
def kept_flags(codebook_path: Path = CODEBOOK_PATH) -> list[str]:
    """The names of the condition flags that survived the prevalence rule, read from the codebook."""
    codebook = pd.read_csv(codebook_path)
    return codebook.loc[codebook["kept"] == True, "flag"].tolist()  # noqa: E712


def load_analysis(
    years: list[int] | range,
    analysis_path: Path = ANALYSIS_PATH,
    codebook_path: Path = CODEBOOK_PATH,
) -> tuple[pd.DataFrame, list[str]]:
    """Load the analysis table for the given years, dropping children with any missing flag.

    Returns the table and the list of flag column names, in the order the
    clustering methods expect them.
    """
    from nsch_subtypes.data import drop_incomplete_rows

    flags = kept_flags(codebook_path)
    table = pd.read_parquet(analysis_path)
    table = table[table["year"].isin(list(years))]
    return drop_incomplete_rows(table, flags), flags


def load_lca_fit(results_dir: Path, n_classes: int) -> dict[str, np.ndarray]:
    """Load the arrays step 3 saved for a given K: class_sizes, condition_rates, responsibilities, flags."""
    archive = np.load(results_dir / f"lca_k{n_classes}.npz")
    return {name: archive[name] for name in archive.files}


def read_best_k(results_dir: Path) -> int:
    """The K that step 3 chose, from ``lca_best.json``."""
    return json.loads((results_dir / "lca_best.json").read_text())["best_k"]


# ---------------------------------------------------------------------------- #
# Saving
# ---------------------------------------------------------------------------- #
def save_json(obj: dict, path: Path) -> None:
    """Write a dictionary as indented JSON, converting NumPy numbers and arrays to plain Python first."""

    def to_plain_python(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Cannot serialize {type(value)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=to_plain_python) + "\n")


# ---------------------------------------------------------------------------- #
# Picking the clustering code
# ---------------------------------------------------------------------------- #
def backend_or_exit(name: str, *methods: str) -> dict:
    """Return the backend's functions, after checking that the requested methods are implemented.

    With ``--backend ours``, a method that is still a stub would otherwise fail
    deep inside a script with a traceback. This probes each one on a tiny
    matrix first and exits with a readable message instead.
    """
    from nsch_subtypes.backends import get_backend

    backend = get_backend(name)
    if name == "ours":
        tiny_matrix = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.uint8)
        for method in methods:
            try:
                if method in ("lca", "kmeans"):
                    backend[method](tiny_matrix, 1, n_starts=1)
                else:
                    backend[method](tiny_matrix, 1)
            except NotImplementedError:
                sys.exit(f"The from-scratch {method} is not in place yet. Run with --backend reference, or see docs/implementation-spec.md.")
            except Exception:
                pass  # any other error is for the real run to surface
    return backend


def log(message: str) -> None:
    """Print immediately (no buffering), so progress shows up while long steps run."""
    print(message, flush=True)
