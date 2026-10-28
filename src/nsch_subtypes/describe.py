"""Describe the classes once they exist: how big each is nationally, what conditions define it, who is in it.

This backs step 7 and is the only part of the project that uses survey
weights.

Why weights, and why only here
------------------------------
The NSCH does not sample children uniformly. Some states and some kinds of
households are deliberately over-sampled, and each child in the file comes
with a weight saying how many U.S. children that child stands for. If we
want to say "class 2 is 18% of autistic children in the United States", we
have to add up weights, not count rows.

We do not use the weights when fitting the clusters. The clusters describe
the structure of the sample; the weights then say how common each pattern is
in the population. Pooling several survey years means each child's weight is
divided by the number of years, so the total still equals one year's child
population.

Standard errors
---------------
A survey estimate comes with uncertainty, and because of the sampling design
that uncertainty is larger than a simple random sample of the same size
would give. We follow the standard recipe (Taylor linearization with the
with-replacement approximation, which is what Stata's ``svy`` does by
default): strata are state by sampling stratum (by year when pooled), one
child per household, and the variance adds up the spread of each stratum's
contributions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def design_strata(table: pd.DataFrame) -> np.ndarray:
    """Give each row an integer stratum id: state x sampling stratum, and x year when more than one year is present."""
    stratum_key = table["state"].astype(str) + "_" + table["stratum"].astype(str)
    if table["year"].nunique() > 1:
        stratum_key = table["year"].astype(str) + "_" + stratum_key
    return pd.factorize(stratum_key)[0]


def weighted_mean_with_se(values: np.ndarray, weights: np.ndarray, strata: np.ndarray, domain: np.ndarray | None = None) -> tuple[float, float]:
    """Weighted mean of ``values`` and its design-based standard error, optionally within a subgroup.

    ``values`` can be 0/1 (giving a share), a class probability, or a number
    such as age. The estimate is ``sum(w * d * y) / sum(w * d)`` where ``d``
    marks membership of the subgroup (``domain``; everyone if not given).

    For the standard error each child's contribution ("linearized score") is
    ``z_i = w_i * d_i * (y_i - mean) / sum(w * d)``, and the variance is the
    sum over strata of ``n_h / (n_h - 1)`` times the squared deviations of
    ``z`` from the stratum mean. Children outside the subgroup stay in with a
    score of zero. That is the standard way to estimate within a subgroup
    (what Stata's ``svy, subpop()`` does): dropping them instead would
    pretend the subgroup's size was fixed by the design and would discard
    strata where the subgroup happens to be empty, understating the error.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    d = np.ones_like(weights) if domain is None else np.asarray(domain, dtype=float)

    total = (weights * d).sum()
    mean = float((weights * d * np.nan_to_num(values)).sum() / total)
    scores = weights * d * (np.nan_to_num(values) - mean) / total

    variance = 0.0
    for stratum in np.unique(strata):
        scores_in_stratum = scores[strata == stratum]
        n_in_stratum = scores_in_stratum.size
        if n_in_stratum > 1:
            variance += n_in_stratum / (n_in_stratum - 1) * ((scores_in_stratum - scores_in_stratum.mean()) ** 2).sum()
    return mean, float(np.sqrt(variance))


def class_shares(table: pd.DataFrame, responsibilities: np.ndarray, pooled_years: int | None = None) -> pd.DataFrame:
    """Weighted share of U.S. autistic children in each class, with a 95% confidence interval.

    Instead of assigning each child to one class and counting, we average each
    child's class probabilities (the ``responsibilities``). A child who is 60%
    class 1 and 40% class 2 contributes 0.6 and 0.4. This carries the model's
    own uncertainty about assignments into the shares.

    ``pooled_years`` divides the weights so that a multi-year table sums to
    one year's population. It changes ``weighted_count`` but not ``share``.

    Returns one row per class with columns ``class, share, se, ci_low, ci_high,
    n_modal`` (children whose most likely class this is) and ``weighted_count``.
    """
    n_years = pooled_years or table["year"].nunique()
    weights = table["weight"].to_numpy() / n_years
    strata = design_strata(table)

    rows = []
    for k in range(responsibilities.shape[1]):
        share, se = weighted_mean_with_se(responsibilities[:, k], weights, strata)
        rows.append(
            {
                "class": k,
                "share": share,
                "se": se,
                "ci_low": max(0.0, share - 1.96 * se),
                "ci_high": min(1.0, share + 1.96 * se),
                "n_modal": int((responsibilities.argmax(axis=1) == k).sum()),
                "weighted_count": float((weights * responsibilities[:, k]).sum()),
            }
        )
    return pd.DataFrame(rows)


def class_profiles(X: np.ndarray, labels: np.ndarray, flags: list[str], n_classes: int | None = None) -> pd.DataFrame:
    """The observed rate of each condition within each class (unweighted), one row per class.

    This is the table a reader looks at to understand what a class "is":
    a class where 95% have ADHD and 80% have anxiety reads very differently
    from one where 70% have intellectual disability and 60% epilepsy.

    Pass ``n_classes`` so that an empty class still gets a row (of NaN) and
    the row count always equals K; without it, K is inferred from the labels.
    """
    n_classes = int(labels.max()) + 1 if n_classes is None else n_classes
    rates = np.full((n_classes, X.shape[1]), np.nan)
    for k in range(n_classes):
        members = labels == k
        if members.any():
            rates[k] = X[members].mean(axis=0)
    profiles = pd.DataFrame(rates, columns=flags)
    profiles.insert(0, "n", [int((labels == k).sum()) for k in range(n_classes)])
    profiles.insert(0, "class", range(n_classes))
    return profiles


def severity_by_class(table: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Weighted share of children rated mild, moderate and severe within each class, with standard errors.

    Children whose severity is missing are treated as outside every class for
    this table (they contribute zero), so the shares are among children with
    a rating.
    """
    strata = design_strata(table)
    weights = table["weight"].to_numpy()
    rated = table["severity"].notna().to_numpy()
    rows = []
    for k in sorted(np.unique(labels)):
        in_class = (labels == k) & rated
        row = {"class": int(k), "n": int(in_class.sum())}
        for level in ("mild", "moderate", "severe"):
            is_level = (table["severity"] == level).to_numpy()
            share, se = weighted_mean_with_se(is_level, weights, strata, domain=in_class)
            row[level] = share
            row[f"{level}_se"] = se
        rows.append(row)
    return pd.DataFrame(rows)


def demographics_by_class(table: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Weighted mean age and share female within each class, with standard errors."""
    strata = design_strata(table)
    weights = table["weight"].to_numpy()
    age = table["age"].to_numpy()
    is_female = (table["sex"] == "female").to_numpy()
    rows = []
    for k in sorted(np.unique(labels)):
        in_class = labels == k
        mean_age, mean_age_se = weighted_mean_with_se(age, weights, strata, domain=in_class)
        share_female, share_female_se = weighted_mean_with_se(is_female, weights, strata, domain=in_class)
        rows.append({"class": int(k), "mean_age": mean_age, "mean_age_se": mean_age_se, "share_female": share_female, "share_female_se": share_female_se})
    return pd.DataFrame(rows)
