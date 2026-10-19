"""Paths, argument parsing and loading shared by the numbered scripts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
INTERIM = REPO / "data" / "interim"
PROCESSED = REPO / "data" / "processed"
RESULTS = REPO / "results"
FIGURES = REPO / "reports" / "figures"

ANALYSIS = PROCESSED / "analysis.parquet"
CODEBOOK = PROCESSED / "codebook.csv"
MAIN_YEARS = range(2016, 2025)  # the pooled analysis; 2025 is held out for the forward test


def parse_years(text: str) -> list[int]:
    """'2016-2024' or '2016,2018,2020' -> list of ints."""
    years: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            years.extend(range(int(a), int(b) + 1))
        else:
            years.append(int(part))
    return years


def parser(description: str, backend: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--results", type=Path, default=RESULTS, help="results directory")
    p.add_argument("--seed", type=int, default=0)
    if backend:
        p.add_argument("--backend", choices=["ours", "reference"], default="ours", help="'ours' = the from-scratch code; 'reference' = StepMix/SciPy")
    return p


def kept_flags(codebook_path: Path = CODEBOOK) -> list[str]:
    cb = pd.read_csv(codebook_path)
    return cb.loc[cb["kept"] == True, "flag"].tolist()  # noqa: E712


def load_analysis(years: list[int] | range, analysis_path: Path = ANALYSIS, codebook_path: Path = CODEBOOK) -> tuple[pd.DataFrame, list[str]]:
    """Complete rows for the given years, plus the list of retained flags."""
    from nsch_subtypes.data import complete_rows

    flags = kept_flags(codebook_path)
    table = pd.read_parquet(analysis_path)
    table = table[table["year"].isin(list(years))]
    return complete_rows(table, flags), flags


def save_json(obj: dict, path: Path) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=default) + "\n")


def load_lca(results: Path, k: int) -> dict:
    z = np.load(results / f"lca_k{k}.npz")
    return {name: z[name] for name in z.files}


def backend_or_exit(name: str, *methods: str):
    """The backend's functions, after checking that the requested methods exist.

    With ``--backend ours`` a method that is still a stub exits with a pointer
    to the spec instead of failing deep inside a script.
    """
    from nsch_subtypes.backends import get_backend

    backend = get_backend(name)
    if name == "ours":
        probe = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.uint8)
        for method in methods:
            try:
                if method == "lca":
                    backend[method](probe, 1, n_starts=1)
                elif method == "kmeans":
                    backend[method](probe, 1, n_starts=1)
                else:
                    backend[method](probe, 1)
            except NotImplementedError:
                sys.exit(f"The from-scratch {method} is not in place yet. Run with --backend reference, or see docs/implementation-spec.md.")
            except Exception:
                pass  # anything else is for the implementation to report when it runs for real
    return backend


def log(msg: str) -> None:
    print(msg, flush=True)
