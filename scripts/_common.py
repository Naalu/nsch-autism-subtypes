"""Small helpers shared by the numbered scripts: where files live, command-line flags, loading, saving.

The numbered scripts are meant to be read in order (01 to 14). Each one is
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


def save_lca_fit(results_dir: Path, fit, table: pd.DataFrame, flags: list[str]) -> None:
    """Save a fitted latent class model with the identity of the rows it was fit on.

    Storing ``year`` and ``hhid`` lets every later step check that it is
    pairing the saved class probabilities with the same children, in the same
    order, instead of trusting that the analysis table has not changed.
    """
    np.savez_compressed(
        results_dir / f"lca_k{fit.n_classes}.npz",
        class_sizes=fit.class_sizes,
        condition_rates=fit.condition_rates,
        responsibilities=fit.responsibilities,
        flags=np.array(flags),
        year=table["year"].to_numpy(),
        hhid=table["hhid"].to_numpy(),
        log_likelihood=fit.log_likelihood,
        n_iter=fit.n_iter,
        converged=fit.converged,
    )


def load_lca_fit(results_dir: Path, n_classes: int, table: pd.DataFrame | None = None, flags: list[str] | None = None):
    """Load the latent class model step 3 saved for K = ``n_classes`` as an :class:`LCAResult`.

    If ``table`` is given, check that the saved model was fit on exactly these
    children in this order (by ``year`` and ``hhid`` when the file has them,
    otherwise by row count). If ``flags`` is given, check that it was fit on
    the same conditions in the same order. Either mismatch stops the script
    with a clear message, because pairing a saved model with a changed table
    would otherwise give wrong answers without any error.
    """
    from nsch_subtypes.lca import LCAResult

    path = results_dir / f"lca_k{n_classes}.npz"
    if not path.exists():
        sys.exit(f"{path} not found. Run scripts/03_fit_lca.py (or copy the step 3 fits into {results_dir}).")
    archive = np.load(path)
    responsibilities = archive["responsibilities"]
    if table is not None:
        if "hhid" in archive.files:
            same = (
                len(archive["hhid"]) == len(table)
                and np.array_equal(archive["hhid"], table["hhid"].to_numpy())
                and np.array_equal(archive["year"], table["year"].to_numpy())
            )
        else:
            same = responsibilities.shape[0] == len(table)
        if not same:
            sys.exit(f"{path} was fit on different children than the current analysis table. Rerun scripts/03_fit_lca.py.")
    if flags is not None and "flags" in archive.files and archive["flags"].tolist() != list(flags):
        sys.exit(f"{path} was fit on different conditions than the current codebook lists. Rerun scripts/03_fit_lca.py.")
    return LCAResult(
        n_classes=n_classes,
        class_sizes=archive["class_sizes"],
        condition_rates=archive["condition_rates"],
        responsibilities=responsibilities,
        log_likelihood=float(archive["log_likelihood"]) if "log_likelihood" in archive.files else float("nan"),
        n_iter=int(archive["n_iter"]) if "n_iter" in archive.files else -1,
        converged=bool(archive["converged"]) if "converged" in archive.files else True,
    )


def resolve_k(args, results_dir: Path | None = None) -> int:
    """The number of classes a check should use.

    ``--k`` on the command line wins. Otherwise use the K chosen in step 13,
    read from ``model_choice.json`` in ``results_dir`` (by default the
    script's ``--results`` folder). Step 3's BIC minimum is deliberately not
    a fallback: on this data it is just the largest K tried.
    """
    if getattr(args, "k", None):
        return int(args.k)
    if results_dir is None:
        results_dir = getattr(args, "results", RESULTS_DIR)
    choice = results_dir / "model_choice.json"
    if choice.exists():
        return int(json.loads(choice.read_text())["chosen_k"])
    sys.exit("No K given. Pass --k, or run scripts/13_heldout_selection.py first so the chosen K is on file.")


# ---------------------------------------------------------------------------- #
# Saving
# ---------------------------------------------------------------------------- #
def _to_plain_python(value):
    """Convert NumPy scalars, arrays and dict keys into types the json module accepts."""
    if isinstance(value, dict):
        return {(k.item() if isinstance(k, np.generic) else k): _to_plain_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_python(v) for v in value]
    if isinstance(value, np.ndarray):
        return _to_plain_python(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else float(value)  # JSON has no NaN; write null
    return value


def save_json(obj: dict, path: Path) -> None:
    """Write a dictionary as indented JSON. NumPy numbers, booleans, arrays and keys are converted; NaN becomes null."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_plain_python(obj), indent=2, allow_nan=False) + "\n")


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
