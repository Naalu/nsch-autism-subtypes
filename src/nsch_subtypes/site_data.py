"""Turning results into the small data files the project website draws from.

The website shows every child as a dot on the two axes and lets a visitor
recolor and cut that cloud. This module holds the pieces that shape
the pipeline's results into what the page needs; ``scripts/build_site_data.py``
loads the results and calls them.

What leaves the repository
--------------------------
The survey files are public, but the site still publishes as little about
any one child as the figures need:

- no household ID, state, sampling stratum, survey weight, sex or exact age;
- children are written in a shuffled order, so a row cannot be matched back
  to the survey file by position;
- a child's dot carries only its position and which condition profile it
  has: no survey year and no age band. Everything else (group
  probabilities, cluster labels) is stored once per profile, because it
  depends only on the 18 answers;
- every distinct set of 18 answers is published once, with how many
  children share it. With no year, age, state or survey order attached, a
  set of answers shared by one child points to no particular survey record
  beyond what the public-use files already hold;
- survey years appear only as counts of children per map cell and group,
  in cells that hold at least ``MIN_YEAR_CELL`` children, and as shares;
- outcomes (severity, missed care, emergency visits) appear only as rates
  by group.

The survey's public-use files already contain every child's answers; the
point of these rules is that the site adds nothing that ties a set of
answers to a survey year, an age or a place.

``assert_no_identifiers`` checks the first point on everything that is
written, and the tests check the rest.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

# Keys that must never appear in a file written for the website.
IDENTIFYING_KEYS = frozenset({"hhid", "state", "fipsst", "stratum", "weight", "fwc", "sex", "age"})

# Fewest children a map cell in one survey year may hold before its counts are published.
MIN_YEAR_CELL = 5

# The conditions that name the two axes, as in the results figures.
DEVELOPMENTAL = ("dev_delay", "learning", "speech", "intellectual")
PSYCHIATRIC = ("anxiety", "adhd", "depression", "behavior")


def assert_no_identifiers(payload: object, path: str = "") -> None:
    """Raise if any dictionary key anywhere in ``payload`` names an identifying field."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in IDENTIFYING_KEYS:
                raise ValueError(f"identifying field '{key}' at {path or 'the top level'}")
            assert_no_identifiers(value, f"{path}.{key}" if path else str(key))
    elif isinstance(payload, list | tuple):
        for item in payload:
            assert_no_identifiers(item, path)


def rounded(payload: object, digits: int = 4) -> object:
    """A copy of ``payload`` with every float rounded and every NumPy value turned into plain Python.

    Keeps the files small and makes them valid JSON (NumPy numbers and NaN are
    not). NaN becomes ``None``.
    """
    if isinstance(payload, dict):
        return {str(key): rounded(value, digits) for key, value in payload.items()}
    if isinstance(payload, list | tuple):
        return [rounded(value, digits) for value in payload]
    if isinstance(payload, np.ndarray):
        return rounded(payload.tolist(), digits)
    if isinstance(payload, bool | np.bool_):
        return bool(payload)
    if isinstance(payload, int | np.integer):
        return int(payload)
    if isinstance(payload, float | np.floating):
        return None if np.isnan(payload) else round(float(payload), digits)
    return payload


# ---------------------------------------------------------------------------- #
# Axes
# ---------------------------------------------------------------------------- #
def axis_alignment(slopes: np.ndarray, flags: list[str]) -> np.ndarray:
    """The 2-by-2 matrix that turns a model's two dimensions into the site's axes.

    The site always draws the developmental-language axis across and the
    mental-health axis up, both pointing toward "more conditions". A fitted
    model's dimensions come in either order and either sign, so this picks
    the order by which dimension the developmental marker conditions load on
    more, and flips a dimension whose marker slopes are negative. Multiply
    positions (or group means) by the result: ``positions @ alignment``.
    """
    slopes = np.asarray(slopes, dtype=float)
    developmental = [flags.index(name) for name in DEVELOPMENTAL]
    psychiatric = [flags.index(name) for name in PSYCHIATRIC]
    first_is_developmental = np.abs(slopes[developmental, 0]).mean() - np.abs(slopes[psychiatric, 0]).mean()
    second_is_developmental = np.abs(slopes[developmental, 1]).mean() - np.abs(slopes[psychiatric, 1]).mean()
    across, up = (0, 1) if first_is_developmental >= second_is_developmental else (1, 0)
    alignment = np.zeros((2, 2))
    alignment[across, 0] = np.sign(slopes[developmental, across].mean()) or 1.0
    alignment[up, 1] = np.sign(slopes[psychiatric, up].mean()) or 1.0
    return alignment


# ---------------------------------------------------------------------------- #
# Children and profiles
# ---------------------------------------------------------------------------- #
def distinct_profiles(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The distinct rows of a 0/1 matrix, each child's row number among them, and how many children share each.

    Profiles are ordered from most to least common, so profile 0 is the
    largest.
    """
    X = np.asarray(X).astype(np.uint8)
    profiles, profile_of_child, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    profile_of_child = profile_of_child.ravel()
    order = np.argsort(-counts, kind="stable")
    new_index = np.empty(len(order), dtype=int)
    new_index[order] = np.arange(len(order))
    return profiles[order], new_index[profile_of_child], counts[order]


def profile_strings(profiles: np.ndarray) -> list[str]:
    """Each profile as a string of 0s and 1s in flag order, for example ``"110000010000000000"``."""
    return ["".join(str(int(value)) for value in row) for row in profiles]


def per_profile(values: np.ndarray, profile_of_child: np.ndarray, n_profiles: int) -> np.ndarray:
    """Collapse a per-child array to one row per profile, checking that children with one profile agree.

    Group probabilities and cluster labels depend only on a child's 18
    answers, so every child with the same profile has the same value. This
    takes the first child's value for each profile and raises if any other
    child with that profile differs, which would mean the value depends on
    something else and cannot be stored per profile.
    """
    values = np.asarray(values)
    first_child = np.full(n_profiles, -1)
    first_child[profile_of_child[::-1]] = np.arange(len(profile_of_child))[::-1]
    collapsed = values[first_child]
    if not np.allclose(collapsed[profile_of_child], values, atol=1e-9):
        raise ValueError("children with the same profile have different values; this cannot be stored per profile")
    return collapsed


def children_columns(positions: np.ndarray, sharp_positions: np.ndarray, profile_of_child: np.ndarray, seed: int = 0) -> dict:
    """The per-child file: columns of equal length, in a shuffled order.

    ``x`` and ``y`` are the position on the site's axes to two decimals,
    drawn assuming positions follow a bell curve; ``sharp_x`` and ``sharp_y``
    are the same child's position drawn from the distribution estimated from
    the data instead (see ``build_site_data.py``). ``profile`` is the child's
    row in the profiles file. Nothing else about the child is written.
    """
    order = np.random.default_rng(seed).permutation(len(profile_of_child))
    return {
        "x": np.round(positions[order, 0], 2).tolist(),
        "y": np.round(positions[order, 1], 2).tolist(),
        "sharp_x": np.round(sharp_positions[order, 0], 2).tolist(),
        "sharp_y": np.round(sharp_positions[order, 1], 2).tolist(),
        "profile": profile_of_child[order].astype(int).tolist(),
    }


def order_by_size(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Renumber cluster labels so that 0 is the largest cluster, 1 the next, and so on."""
    sizes = np.bincount(labels, minlength=n_classes)
    order = np.argsort(-sizes, kind="stable")
    new_label = np.empty(n_classes, dtype=int)
    new_label[order] = np.arange(n_classes)
    return new_label[labels]


def _match_slots(labels: np.ndarray, n_classes: int, reference: np.ndarray, reference_slots: list[int]) -> list[int]:
    """Give each of ``n_classes`` clusters the slot of the reference cluster it overlaps most, one to one.

    With more clusters than reference clusters, the unmatched ones get new
    slots after the largest slot in use, in order of size.
    """
    n_reference = len(reference_slots)
    overlap = np.zeros((n_classes, n_reference))
    np.add.at(overlap, (labels, reference), 1)
    rows, columns = linear_sum_assignment(-overlap)
    slots = [-1] * n_classes
    for row, column in zip(rows, columns, strict=True):
        slots[row] = reference_slots[column]
    next_slot = max(reference_slots) + 1
    sizes = np.bincount(labels, minlength=n_classes)
    for c in sorted(range(n_classes), key=lambda c: -sizes[c]):
        if slots[c] < 0:
            slots[c] = next_slot
            next_slot += 1
    return slots


def color_slots(labels_by_k: dict[int, np.ndarray], reference_k: int) -> dict[int, list[int]]:
    """For each K, the color slot of each cluster, so a group keeps its color as K changes.

    At ``reference_k`` cluster ``c`` has slot ``c``. Moving one step up or
    down in K, each cluster takes the slot of the cluster at the neighboring
    K that it shares the most children with (one to one), and a cluster with
    no partner, which happens only going up, takes the next unused slot. So
    when the four groups split into five, four of the five keep their colors
    and the new one gets a fifth.
    """
    slots = {reference_k: list(range(reference_k))}
    ks = sorted(labels_by_k)
    for k in [k for k in ks if k > reference_k]:
        slots[k] = _match_slots(labels_by_k[k], k, labels_by_k[k - 1], slots[k - 1])
    for k in [k for k in ks if k < reference_k][::-1]:
        slots[k] = _match_slots(labels_by_k[k], k, labels_by_k[k + 1], slots[k + 1])
    return {k: slots[k] for k in ks}


# ---------------------------------------------------------------------------- #
# Aggregates
# ---------------------------------------------------------------------------- #
def binned_counts(
    positions: np.ndarray, labels: np.ndarray, n_classes: int, cell: float = 0.75, limit: float = 3.75, min_cell: int = MIN_YEAR_CELL
) -> list[dict]:
    """How many children of each group fall in each square cell of the map.

    Cells are ``cell`` wide and cover ``-limit`` to ``limit`` on both axes;
    children outside are left out. A cell with fewer than ``min_cell``
    children is left out too. Each returned cell has its center (``x``,
    ``y``) and ``counts``, one per group.
    """
    positions = np.asarray(positions)
    edges = np.arange(-limit, limit + cell / 2, cell)
    column = np.digitize(positions[:, 0], edges) - 1
    row = np.digitize(positions[:, 1], edges) - 1
    inside = (column >= 0) & (column < len(edges) - 1) & (row >= 0) & (row < len(edges) - 1)
    cells = []
    for c, r in sorted(set(zip(column[inside].tolist(), row[inside].tolist(), strict=True))):
        members = inside & (column == c) & (row == r)
        if members.sum() >= min_cell:
            cells.append(
                {"x": float(edges[c] + cell / 2), "y": float(edges[r] + cell / 2), "counts": np.bincount(labels[members], minlength=n_classes).tolist()}
            )
    return cells


def share_with_interval(has_outcome: np.ndarray) -> dict:
    """A share with a 95% Wilson interval, for a 0/1 array."""
    n = len(has_outcome)
    p = float(np.mean(has_outcome)) if n else float("nan")
    z = 1.959964
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return {"n": int(n), "share": p, "low": float(center - half), "high": float(center + half)}


def ranges_by(records: Iterable[dict], keys: tuple[str, ...], value: str) -> list[dict]:
    """Smallest and largest ``value`` among records that share the same ``keys``, one output record per combination."""
    records = sorted(records, key=lambda record: tuple(str(record[key]) for key in keys))
    out = []
    for combination, group in itertools.groupby(records, key=lambda record: tuple(record[key] for key in keys)):
        values = [record[value] for record in group]
        out.append({**dict(zip(keys, combination, strict=True)), "min": min(values), "max": max(values), "n": len(values)})
    return out
