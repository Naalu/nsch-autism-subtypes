"""Step 2 of 14. Turn survey answers into the 0/1 table the clustering methods use.

What happens here
-----------------
1. Stack the per-year files from step 1 into one table.
2. For each condition in the config, build a flag: 1 if the child currently
   has it, 0 if not, missing if the parent did not answer. The exact rule is
   in ``nsch_subtypes.data.encode_flag``.
3. Apply the prevalence rule on the 2016-2024 children: drop any flag that
   fewer than 1% of them have. (2025 is held out and does not get a vote.)
4. Write ``data/processed/analysis.parquet`` (one row per child, all years)
   and ``data/processed/codebook.csv`` (every condition we considered and
   whether it made it in).

Usage::

    python scripts/02_encode.py
"""

from __future__ import annotations

import pandas as pd

from _common import ANALYSIS_PATH, CODEBOOK_PATH, INTERIM_DIR, MAIN_YEARS, log, make_parser
from nsch_subtypes.config import load_config
from nsch_subtypes.data import apply_prevalence_rule, build_analysis_table


def main() -> None:
    parser = make_parser("Encode condition flags from the selected rows", backend=False)
    parser.parse_args()  # no extra options; this keeps --help consistent across scripts

    config = load_config()
    interim_files = sorted(INTERIM_DIR.glob("selected_*.parquet"))
    if not interim_files:
        raise SystemExit("No interim files; run scripts/01_select.py first")

    selected_rows = pd.concat([pd.read_parquet(path) for path in interim_files], ignore_index=True)
    table = build_analysis_table(selected_rows, config)

    # Decide which flags to keep using only the main analysis years.
    main_rows = table[table["year"].isin(list(MAIN_YEARS))]
    kept, codebook = apply_prevalence_rule(main_rows, config)
    dropped = [condition.name for condition in config.included if condition.name not in kept]
    table = table.drop(columns=dropped + [f"ever_{name}" for name in dropped])

    # Write to a temporary name and then rename, so a script that is reading the
    # table while this runs never sees a half-written file.
    partial_path = ANALYSIS_PATH.with_suffix(".parquet.partial")
    table.to_parquet(partial_path, index=False)
    partial_path.replace(ANALYSIS_PATH)
    codebook.to_csv(CODEBOOK_PATH, index=False)

    complete = table.dropna(subset=kept)
    log(f"{len(table)} children across {table['year'].nunique()} years; {len(kept)} flags kept, {len(dropped)} dropped by the prevalence rule: {dropped}")
    log(f"{len(complete)} rows have every flag answered ({len(table) - len(complete)} with at least one missing)")
    log(f"wrote {ANALYSIS_PATH} and {CODEBOOK_PATH}")


if __name__ == "__main__":
    main()
