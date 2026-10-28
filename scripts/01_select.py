"""Step 1 of 14. Pick out the children we study from each year's survey file.

What happens here
-----------------
For each survey year (2016 to 2025 by default):

1. Download the Census file if we do not have it yet (``--download``).
2. Read only the columns named in ``config/conditions.yaml``. The full file has
   over 400 columns; we need about 50.
3. Keep children aged 6 to 17 whose parent says they currently have autism.
4. Save those rows to ``data/interim/selected_{year}.parquet`` with their
   original Census column names, and record how many children passed each
   filter.

The counts go to ``data/processed/selection_counts.csv``, which is the
"how did we get from 400,000 children to 11,000" table in the report.

Usage::

    python scripts/01_select.py --download              # all years
    python scripts/01_select.py --years 2025 --download # one year
    python scripts/01_select.py --loader nsch           # read via the nsch package instead
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import INTERIM_DIR, PROCESSED_DIR, RAW_DIR, log, make_parser, parse_years
from nsch_subtypes.config import load_config
from nsch_subtypes.data import download_year, read_year, select_children


def main() -> None:
    parser = make_parser("Select autistic children aged 6-17 from the NSCH files", backend=False)
    parser.add_argument("--years", default="2016-2025", help="e.g. 2016-2025 or 2022,2023")
    parser.add_argument("--download", action="store_true", help="download missing years from the Census Bureau")
    parser.add_argument("--loader", choices=["census", "nsch"], default="census", help="how to read the files")
    parser.add_argument("--raw", type=Path, default=RAW_DIR, help="where the Census files live")
    args = parser.parse_args()

    config = load_config()
    columns_needed = config.raw_columns
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    count_rows = []
    for year in parse_years(args.years):
        if args.download:
            download_year(year, args.raw)
        all_children = read_year(year, args.raw, columns_needed, loader=args.loader)
        selected, counts = select_children(all_children, config.keys)
        selected.to_parquet(INTERIM_DIR / f"selected_{year}.parquet", index=False)
        count_rows.append({"year": year, **counts})
        log(
            f"{year}: {counts['n_children']:>6} children, {counts['n_age_6_17']:>6} aged 6-17, "
            f"{counts['n_asd_current']:>5} currently autistic -> {counts['n_selected']} selected"
        )

    # Merge with counts from earlier runs so that running one year does not
    # erase the others.
    new_counts = pd.DataFrame(count_rows)
    counts_path = PROCESSED_DIR / "selection_counts.csv"
    if counts_path.exists():
        old_counts = pd.read_csv(counts_path)
        kept_old = old_counts[~old_counts["year"].isin(new_counts["year"])]
        new_counts = pd.concat([kept_old, new_counts]).sort_values("year")
    new_counts.to_csv(counts_path, index=False)
    log(f"wrote {counts_path}")


if __name__ == "__main__":
    main()
