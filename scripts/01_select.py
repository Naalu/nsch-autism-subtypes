"""Step 1. Select currently-autistic children aged 6-17 from each survey year.

Reads the Census topical file for each year (downloading it first with
``--download``), keeps the columns named in config/conditions.yaml, applies the
selection, and writes one interim file per year plus a table of the counts at
each filter.

    python scripts/01_select.py --download
    python scripts/01_select.py --years 2025 --download
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import INTERIM, PROCESSED, RAW, log, parse_years, parser
from nsch_subtypes.config import load_config
from nsch_subtypes.data import download_year, read_year, select


def main() -> None:
    p = parser("Select autistic children aged 6-17 from the NSCH files", backend=False)
    p.add_argument("--years", default="2016-2025", help="e.g. 2016-2025 or 2022,2023")
    p.add_argument("--download", action="store_true", help="download missing years from the Census Bureau")
    p.add_argument("--loader", choices=["census", "nsch"], default="census")
    p.add_argument("--raw", type=Path, default=RAW, help="where the Census files live")
    args = p.parse_args()

    config = load_config()
    columns = config.raw_columns
    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    counts = []
    for year in parse_years(args.years):
        if args.download:
            download_year(year, args.raw)
        df = read_year(year, args.raw, columns, loader=args.loader)
        selected, c = select(df, config.keys)
        selected.to_parquet(INTERIM / f"selected_{year}.parquet", index=False)
        counts.append({"year": year, **c})
        log(f"{year}: {c['n_children']:>6} children, {c['n_age_6_17']:>6} aged 6-17, {c['n_asd_current']:>5} currently autistic -> {c['n_selected']} selected")

    out = pd.DataFrame(counts)
    existing = PROCESSED / "selection_counts.csv"
    if existing.exists():  # keep counts for years not run this time
        prev = pd.read_csv(existing)
        out = pd.concat([prev[~prev["year"].isin(out["year"])], out]).sort_values("year")
    out.to_csv(existing, index=False)
    log(f"wrote {existing}")


if __name__ == "__main__":
    main()
