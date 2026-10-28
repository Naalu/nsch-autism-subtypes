"""Read ``config/conditions.yaml``, the one file that says which survey questions we use.

The YAML file lists every condition we considered, whether it is in or out,
and why; its header comments explain each field. This module turns it into
two small objects, ``Condition`` (one entry) and ``Config`` (the whole file),
that the rest of the package reads from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# The repository root is two directories above this file (src/nsch_subtypes/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "conditions.yaml"


@dataclass(frozen=True)
class Condition:
    """One health condition and the raw survey items it is built from.

    Attributes
    ----------
    name : str
        Short identifier used as a column name, for example ``"adhd"``.
    label : str
        Readable label for tables and figures, for example ``"ADD/ADHD"``.
    ever : str
        Census variable name of the "ever told" question (1 = yes, 2 = no).
    currently : str or None
        Census variable name of the "currently has it" follow-up, or ``None``
        for lifelong conditions such as Down syndrome that have no follow-up.
    years : str
        Years the question was asked, as written in the config ("2016-2025").
    include : bool
        Whether the condition is a candidate for the analysis.
    note : str or None
        Anything worth knowing about the item (wording changes, quirks).
    reason : str or None
        Why the condition was excluded, when ``include`` is ``False``.
    """

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
        """The survey columns this condition needs: the "ever" item, plus the follow-up if there is one."""
        columns = [self.ever]
        if self.currently:
            columns.append(self.currently)
        return columns


@dataclass(frozen=True)
class Config:
    """Everything in ``conditions.yaml``: the condition list, the key columns, and the prevalence rule.

    Attributes
    ----------
    conditions : list of Condition
        Every condition in the file, included or not.
    keys : dict
        Our names for the non-condition columns we carry along, mapped to the
        Census variable names. For example ``keys["age"] == "sc_age_years"``.
    min_prevalence : float
        A condition is dropped if fewer than this share of the pooled autistic
        sample has it (default 0.01, meaning 1%). Clustering on a flag that is
        almost always zero adds noise and little else.
    """

    conditions: list[Condition]
    keys: dict[str, str]
    min_prevalence: float

    @property
    def included(self) -> list[Condition]:
        """Only the conditions marked ``include: true``."""
        return [condition for condition in self.conditions if condition.include]

    @property
    def raw_columns(self) -> list[str]:
        """Every raw survey column the pipeline needs to read, keys first, with no duplicates.

        Reading only these columns (instead of all 400-plus in each file) is
        what keeps the data step fast.
        """
        columns = list(self.keys.values())
        for condition in self.included:
            columns.extend(condition.raw_columns)
        # Drop duplicates while keeping the original order.
        seen: set[str] = set()
        unique_columns = []
        for column in columns:
            if column not in seen:
                seen.add(column)
                unique_columns.append(column)
        return unique_columns


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Parse the YAML file at ``path`` into a :class:`Config`."""
    raw = yaml.safe_load(Path(path).read_text())
    conditions = [Condition(**entry) for entry in raw["conditions"]]
    return Config(
        conditions=conditions,
        keys=dict(raw["keys"]),
        min_prevalence=float(raw.get("min_prevalence", 0.01)),
    )
