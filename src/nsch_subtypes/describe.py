"""Describe the classes: survey-weighted shares, condition profiles, severity (step 7).

Weights are used here and nowhere else. The models are fit unweighted; the
weights then say how common each pattern is among US autistic children.

Standard errors use the NSCH design: strata are state by sampling stratum,
one child per household, Taylor linearization with the with-replacement
approximation (what Stata's ``svy`` with pweights does by default).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def design_strata(table: pd.DataFrame) -> np.ndarray:
    """Stratum id per row: state x sampling stratum (x year, when years are pooled)."""
    parts = table["state"].astype(str) + "_" + table["stratum"].astype(str)
    if table["year"].nunique() > 1:
        parts = table["year"].astype(str) + "_" + parts
    return pd.factorize(parts)[0]


def weighted_mean_se(y: np.ndarray, w: np.ndarray, strata: np.ndarray) -> tuple[float, float]:
    """Weighted mean of ``y`` with a design-based standard error.

    ``y`` may be 0/1 (a share) or a class probability. Linearized score
    ``z_i = w_i (y_i - p) / sum(w)``; variance sums, within each stratum,
    ``n_h / (n_h - 1)`` times the squared deviations of ``z`` from the
    stratum mean. Strata with one observation contribute nothing.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    p = float((w * y).sum() / w.sum())
    z = w * (y - p) / w.sum()
    var = 0.0
    for h in np.unique(strata):
        zh = z[strata == h]
        n_h = zh.size
        if n_h > 1:
            var += n_h / (n_h - 1) * ((zh - zh.mean()) ** 2).sum()
    return p, float(np.sqrt(var))


def class_shares(table: pd.DataFrame, responsibilities: np.ndarray, pooled_years: int | None = None) -> pd.DataFrame:
    """Weighted share of each class with a 95% confidence interval.

    Uses each child's class probabilities, so uncertainty of assignment is
    carried into the share. ``pooled_years`` divides the weights so that a
    multi-year table still sums to one year's population; it does not change
    the shares, only the implied counts.
    """
    years = pooled_years or table["year"].nunique()
    w = table["weight"].to_numpy() / years
    strata = design_strata(table)
    rows = []
    for k in range(responsibilities.shape[1]):
        p, se = weighted_mean_se(responsibilities[:, k], w, strata)
        rows.append(
            {
                "class": k,
                "share": p,
                "se": se,
                "ci_low": max(0.0, p - 1.96 * se),
                "ci_high": min(1.0, p + 1.96 * se),
                "n_modal": int((responsibilities.argmax(axis=1) == k).sum()),
                "weighted_count": float((w * responsibilities[:, k]).sum()),
            }
        )
    return pd.DataFrame(rows)


def class_profiles(X: np.ndarray, labels: np.ndarray, flags: list[str]) -> pd.DataFrame:
    """Observed rate of each flag within each class (rows = classes)."""
    k = labels.max() + 1
    rates = np.vstack([X[labels == c].mean(axis=0) for c in range(k)])
    out = pd.DataFrame(rates, columns=flags)
    out.insert(0, "n", [int((labels == c).sum()) for c in range(k)])
    out.insert(0, "class", range(k))
    return out


def severity_by_class(table: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Weighted distribution of parent-rated severity within each class."""
    df = table.assign(cls=labels).dropna(subset=["severity"])
    strata = design_strata(df)
    rows = []
    for c in sorted(df["cls"].unique()):
        sub = df["cls"] == c
        row = {"class": int(c), "n": int(sub.sum())}
        for level in ("mild", "moderate", "severe"):
            p, se = weighted_mean_se((df.loc[sub, "severity"] == level).to_numpy(), df.loc[sub, "weight"].to_numpy(), strata[sub.to_numpy()])
            row[level] = p
            row[f"{level}_se"] = se
        rows.append(row)
    return pd.DataFrame(rows)


def demographics_by_class(table: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Weighted mean age and share female within each class."""
    df = table.assign(cls=labels)
    strata = design_strata(df)
    rows = []
    for c in sorted(df["cls"].unique()):
        sub = (df["cls"] == c).to_numpy()
        w = df.loc[sub, "weight"].to_numpy()
        age, age_se = weighted_mean_se(df.loc[sub, "age"].to_numpy(), w, strata[sub])
        fem, fem_se = weighted_mean_se((df.loc[sub, "sex"] == "female").to_numpy(), w, strata[sub])
        rows.append({"class": int(c), "mean_age": age, "mean_age_se": age_se, "share_female": fem, "share_female_se": fem_se})
    return pd.DataFrame(rows)
