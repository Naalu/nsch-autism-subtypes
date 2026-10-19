"""Step 2. Encode the condition flags and build the analysis table.

Reads the interim files from step 1, builds one 0/1 flag per condition,
applies the prevalence rule on the pooled 2016-2024 sample, and writes

    data/processed/analysis.parquet   one row per selected child, all years
    data/processed/codebook.csv       every configured item and whether it was kept

2025 rows are encoded with the same rule but do not take part in deciding
which flags are kept; that year is held out for the forward test.
"""

from __future__ import annotations

import pandas as pd

from _common import ANALYSIS, CODEBOOK, INTERIM, MAIN_YEARS, log, parser
from nsch_subtypes.config import load_config
from nsch_subtypes.data import apply_prevalence_rule, encode


def main() -> None:
    p = parser("Encode condition flags from the selected rows", backend=False)
    args = p.parse_args()  # noqa: F841

    config = load_config()
    files = sorted(INTERIM.glob("selected_*.parquet"))
    if not files:
        raise SystemExit("No interim files; run scripts/01_select.py first")
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    table = encode(raw, config)

    main_rows = table[table["year"].isin(list(MAIN_YEARS))]
    kept, codebook = apply_prevalence_rule(main_rows, config)
    drop = [c.name for c in config.included if c.name not in kept]
    table = table.drop(columns=drop)

    table.to_parquet(ANALYSIS, index=False)
    codebook.to_csv(CODEBOOK, index=False)

    complete = table.dropna(subset=kept)
    log(f"{len(table)} children across {table['year'].nunique()} years; {len(kept)} flags kept, {len(drop)} dropped by the prevalence rule: {drop}")
    log(f"{len(complete)} rows have every flag answered ({len(table) - len(complete)} with at least one missing)")
    log(f"wrote {ANALYSIS} and {CODEBOOK}")


if __name__ == "__main__":
    main()
