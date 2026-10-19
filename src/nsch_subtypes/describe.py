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


def weighted_mean_with_se(values: np.ndarray, weights: np.ndarray, strata: np.ndarray) -> tuple[float, float]:
    """Weighted mean of ``values`` and its design-based standard error.

    ``values`` can be 0/1 (giving a share), a class probability, or a number
    such as age. The estimate is ``sum(w * y) / sum(w)``. For the standard
    error, each child's contribution ("linearized score") is
    ``z_i = w_i * (y_i - mean) / sum(w)``, and the variance is the sum over
    strata of ``n_h / (n_h - 1)`` times the squared deviations of ``z`` from
    the stratum mean. A stratum with a single child adds nothing.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mean = float((weights * values).sum() / weights.sum())
    scores = weights * (values - mean) / weights.sum()

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


def class_profiles(X: np.ndarray, labels: np.ndarray, flags: list[str]) -> pd.DataFrame:
    """The observed rate of each condition within each class (unweighted), one row per class.

    This is the table a reader looks at to understand what a class "is":
    a class where 95% have ADHD and 80% have anxiety reads very differently
    from one where 70% have intellectual disability and 60% epilepsy.
    """
    n_classes = labels.max() + 1
    rates = np.vstack([X[labels == k].mean(axis=0) for k in range(n_classes)])
    profiles = pd.DataFrame(rates, columns=flags)
    profiles.insert(0, "n", [int((labels == k).sum()) for k in range(n_classes)])
    profiles.insert(0, "class", range(n_classes))
    return profiles


def severity_by_class(table: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Weighted share of children rated mild, moderate and severe within each class, with standard errors."""
    frame = table.assign(cls=labels).dropna(subset=["severity"])
    strata = design_strata(frame)

    rows = []
    for k in sorted(frame["cls"].unique()):
        in_class = (frame["cls"] == k).to_numpy()
        weights = frame.loc[in_class, "weight"].to_numpy()
        row = {"class": int(k), "n": int(in_class.sum())}
        for level in ("mild", "moderate", "severe"):
            is_level = (frame.loc[in_class, "severity"] == level).to_numpy()
            share, se = weighted_mean_with_se(is_level, weights, strata[in_class])
            row[level] = share
            row[f"{level}_se"] = se
        rows.append(row)
    return pd.DataFrame(rows)


def demographics_by_class(table: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Weighted mean age and share female within each class, with standard errors."""
    frame = table.assign(cls=labels)
    strata = design_strata(frame)

    rows = []
    for k in sorted(frame["cls"].unique()):
        in_class = (frame["cls"] == k).to_numpy()
        weights = frame.loc[in_class, "weight"].to_numpy()
        mean_age, mean_age_se = weighted_mean_with_se(frame.loc[in_class, "age"].to_numpy(), weights, strata[in_class])
        is_female = (frame.loc[in_class, "sex"] == "female").to_numpy()
        share_female, share_female_se = weighted_mean_with_se(is_female, weights, strata[in_class])
        rows.append(
            {
                "class": int(k),
                "mean_age": mean_age,
                "mean_age_se": mean_age_se,
                "share_female": share_female,
                "share_female_se": share_female_se,
            }
        )
    return pd.DataFrame(rows)
