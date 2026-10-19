"""Load the condition configuration from ``config/conditions.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "conditions.yaml"


@dataclass(frozen=True)
class Condition:
    """One analysis flag and the raw survey items it is built from."""

    name: str
    label: str
    ever: str
    currently: str | None
    years: str
    include: bool
    note: str | None = None
    reason: str | None = None

    @property
    def raw_columns(self) -> list[str]:
        return [self.ever] + ([self.currently] if self.currently else [])


@dataclass(frozen=True)
class Config:
    conditions: list[Condition]
    keys: dict[str, str]
    min_prevalence: float

    @property
    def included(self) -> list[Condition]:
        return [c for c in self.conditions if c.include]

    @property
    def raw_columns(self) -> list[str]:
        """Every raw column the pipeline reads, keys first, in a stable order."""
        cols = list(self.keys.values())
        for c in self.included:
            cols.extend(c.raw_columns)
        seen: set[str] = set()
        return [c for c in cols if not (c in seen or seen.add(c))]


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    conditions = [Condition(**c) for c in raw["conditions"]]
    return Config(
        conditions=conditions,
        keys=dict(raw["keys"]),
        min_prevalence=float(raw.get("min_prevalence", 0.01)),
    )
