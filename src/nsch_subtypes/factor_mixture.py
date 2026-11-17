"""Groups and continuous axes in one model: the factor mixture model.

Why this exists
---------------
The latent class model (``lca.py``) says children come in K kinds and that
two children of the same kind differ only by chance. The latent trait model
(``latent_trait.py``) says there are no kinds at all, only positions on
continuous axes. Real data can sit between the two: children spread along
the axes, but bunched around a few typical positions. A factor mixture model
(Lubke and Muthen 2005) allows exactly that, and has both simpler models
inside it, so it can be asked directly: once the axes are in the model, does
adding groups still help?

The model
---------
Each condition depends on a child's position ``theta`` exactly as in the
latent trait model, with one slope vector ``a_j`` and one intercept ``b_j``
per condition, the same for every child:

    P(x_ij = 1 | theta_i) = sigmoid(a_j . theta_i + b_j)

What changes is where the positions come from. Instead of one bell curve,
they come from a mixture of K bell curves, one per group:

    theta_i ~ share_1 N(mean_1, cov_1) + ... + share_K N(mean_K, cov_K)

With K = 1 this is the latent trait model. As the groups' spreads shrink
toward zero it becomes a latent class model whose classes sit at K points on
the axes.

Two versions are fit. In the shared-spread version every group has the same
covariance, so groups differ only in where they sit; this is the main model
because it has the fewest parameters and is the most stable with yes/no
items. In the separate-spread version each group also has its own
covariance; it is the sensitivity check, because groups that differ in how
spread out they are would be invisible to the first version.

Keeping every condition's slopes and intercept the same across groups
(measurement invariance) is what makes the groups comparable: a group is a
place on the shared axes, not a different meaning for the conditions.

How it is fit
-------------
Marginal maximum likelihood by EM on a fixed, evenly spaced grid of
positions. Each group's bell curve becomes a set of weights on the grid. The
E-step works out, for every child, how plausible each (group, grid point)
pair is given the child's answers. The M-step then updates the group shares,
means and covariances from those posteriors, and each condition's slopes and
intercept by a weighted logistic regression on the grid, as in the latent
trait model. Children with identical rows are fit once with a count.

Shifting, stretching or rotating the axes changes nothing about the fit as
long as the slopes and intercepts move with them, so after every step the
solution is put in a standard frame: positions have mean zero and identity
covariance over all children. In that frame the slopes are on the same
scale as the latent trait model's, and the grid always covers the data. The
returned slopes are varimax-rotated, with the group means and covariances
rotated to match.

Mixture likelihoods have many local optima, and EM for this model is slow
near the top (the likelihood surface is nearly flat in the directions that
move the groups). So the fit runs in two stages: a short run from each of
several starting points (one from k-means on the latent trait scores, the
rest random), then the most promising few are run on until the
log-likelihood stops changing, and the best is kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .latent_trait import MAX_SLOPE, LatentTraitResult, _even_grid, _varimax, fit_latent_trait

# Smallest standard deviation a group may have along any direction, in units
# where all children together have standard deviation 1. Below this a group
# is narrower than the grid can represent, and the model is turning into a
# latent class model anyway.
MIN_GROUP_SD = 0.25


@dataclass
class FactorMixtureResult:
    """A fitted factor mixture model.

    Attributes
    ----------
    n_groups, n_dims : int
        K and D.
    shared_spread : bool
        True if every group has the same covariance.
    slopes : ndarray, shape (J, D)
        ``a_j``, varimax-rotated, in the frame where positions have mean zero
        and identity covariance over all children.
    intercepts : ndarray, shape (J,)
    group_shares : ndarray, shape (K,)
        Estimated share of children in each group; sums to 1. Largest first.
    group_means : ndarray, shape (K, D)
        Where each group sits on the axes.
    group_covariances : ndarray, shape (K, D, D)
        Each group's spread around its mean.
    group_probabilities : ndarray, shape (n, K)
        Each child's posterior probability of belonging to each group.
    scores : ndarray, shape (n, D)
        Each child's posterior mean position.
    log_likelihood : float
    n_iter : int
        EM iterations of the best start.
    converged : bool
    n_starts : int
        Starting points tried.
    grid_points, grid_limit : int, float
        The grid has ``grid_points`` evenly spaced values from ``-grid_limit``
        to ``grid_limit`` on each axis.
    max_slope : float
    """

    n_groups: int
    n_dims: int
    shared_spread: bool
    slopes: np.ndarray
    intercepts: np.ndarray
    group_shares: np.ndarray
    group_means: np.ndarray
    group_covariances: np.ndarray
    group_probabilities: np.ndarray
    scores: np.ndarray
    log_likelihood: float
    n_iter: int
    converged: bool
    n_starts: int = 1
    grid_points: int = 41
    grid_limit: float = 5.0
    max_slope: float = MAX_SLOPE

    @property
    def n_parameters(self) -> int:
        """Free parameters.

        Conditions: J slopes per dimension and J intercepts, less the
        D(D-1)/2 rotations that leave the fit unchanged. Groups: K - 1 shares
        and K means, less the D shifts that the intercepts absorb. Spreads:
        nothing in the shared version (the one shared covariance is fixed by
        the choice of scale); D(D+1)/2 per group beyond the first in the
        separate version.
        """
        n_items, d, k = self.slopes.shape[0], self.n_dims, self.n_groups
        items = n_items * (d + 1) - d * (d - 1) // 2
        groups = (k - 1) + (k - 1) * d
        spreads = 0 if self.shared_spread else (k - 1) * d * (d + 1) // 2
        return items + groups + spreads

    def bic(self, n: int | None = None) -> float:
        n_children = self.scores.shape[0] if n is None else n
        return -2.0 * self.log_likelihood + self.n_parameters * np.log(n_children)

    @property
    def labels(self) -> np.ndarray:
        """Each child's most likely group."""
        return self.group_probabilities.argmax(axis=1)

    @property
    def at_bound(self) -> bool:
        """True if any slope ended on the bound (see ``LatentTraitResult.at_bound``)."""
        return bool(np.isfinite(self.max_slope) and (np.abs(self.slopes) >= self.max_slope - 1e-3).any())

    @property
    def smallest_group_sd(self) -> float:
        """The smallest standard deviation of any group along any direction."""
        return float(np.sqrt(min(np.linalg.eigvalsh(cov).min() for cov in self.group_covariances)))


# ---------------------------------------------------------------------------- #
# Pieces of the EM algorithm
# ---------------------------------------------------------------------------- #
def _group_weights_on_grid(nodes: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """Each group's bell curve as weights on the grid, shape (K, Q); every row sums to 1."""
    weights = np.empty((len(means), len(nodes)))
    for k, (mean, covariance) in enumerate(zip(means, covariances, strict=True)):
        centered = nodes - mean
        squared_distance = np.einsum("qd,de,qe->q", centered, np.linalg.inv(covariance), centered)
        density = np.exp(-0.5 * (squared_distance - squared_distance.min()))
        weights[k] = density / density.sum()
    return weights


def _row_likelihood_on_grid(X: np.ndarray, slopes: np.ndarray, intercepts: np.ndarray, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P(row | position) for every row and grid point, scaled by each row's largest value.

    Returns the scaled likelihoods, shape (rows, Q), and the log of each
    row's scale, shape (rows,). Scaling keeps the numbers away from
    underflow: a product of 18 small probabilities is tiny.
    """
    logits = nodes @ slopes.T + intercepts[None, :]
    log_likelihood = X @ (-np.logaddexp(0.0, -logits)).T + (1 - X) @ (-np.logaddexp(0.0, logits)).T
    log_scale = log_likelihood.max(axis=1)
    return np.exp(log_likelihood - log_scale[:, None]), log_scale


def _inverse_square_root(matrix: np.ndarray) -> np.ndarray:
    """The symmetric matrix A with A @ matrix @ A = identity."""
    values, vectors = np.linalg.eigh(matrix)
    return (vectors / np.sqrt(values)) @ vectors.T


def _floor_spread(covariance: np.ndarray, min_sd: float) -> np.ndarray:
    """Raise any standard deviation of a covariance matrix below ``min_sd`` up to it."""
    values, vectors = np.linalg.eigh(covariance)
    return (vectors * np.maximum(values, min_sd**2)) @ vectors.T


def _standardize(slopes, intercepts, shares, means, covariances):
    """Move the solution to the frame where positions have mean zero and identity covariance over all children.

    The fit is unchanged: shifting positions by ``c`` is absorbed by the
    intercepts, and a linear change of axes ``A`` by the slopes.
    """
    center = shares @ means
    means = means - center
    intercepts = intercepts + slopes @ center
    total = np.einsum("k,kde->de", shares, covariances) + (means * shares[:, None]).T @ means
    change = _inverse_square_root(total)
    slopes = slopes @ np.linalg.inv(change)
    means = means @ change
    covariances = np.einsum("ab,kbc,cd->kad", change, covariances, change)
    return slopes, intercepts, means, covariances


def _update_conditions(slopes, intercepts, design, node_weight, node_success, max_slope: float):
    """M-step for the conditions: one weighted logistic regression per condition on the grid.

    The same update as in ``fit_latent_trait``: a few Newton steps each,
    starting from the current values.
    """
    n_items, n_dims = slopes.shape
    slopes, intercepts = slopes.copy(), intercepts.copy()
    for j in range(n_items):
        beta = np.append(slopes[j], intercepts[j])
        for _ in range(10):
            p = 1 / (1 + np.exp(-(design @ beta)))
            gradient = design.T @ (node_success[:, j] - node_weight * p)
            hessian = (design * (node_weight * p * (1 - p))[:, None]).T @ design + 1e-6 * np.eye(n_dims + 1)
            step = np.linalg.solve(hessian, gradient)
            beta += step
            if np.abs(step).max() < 1e-7:
                break
        slopes[j], intercepts[j] = np.clip(beta[:n_dims], -max_slope, max_slope), beta[n_dims]
    return slopes, intercepts


def _run_em(distinct, counts, nodes, slopes, intercepts, shares, means, covariances, shared_spread, max_iter, tol, max_slope):
    """EM from one starting point. Returns the parameters, log-likelihood, iterations and whether it converged."""
    design = np.column_stack([nodes, np.ones(len(nodes))])
    n_children = counts.sum()
    n_groups, n_dims = means.shape
    slopes, intercepts, means, covariances = _standardize(slopes, intercepts, shares, means, covariances)
    previous, log_likelihood, converged, iteration = -np.inf, -np.inf, False, 0
    for iteration in range(1, max_iter + 1):  # noqa: B007 (reported after the loop)
        # E-step. ``prior[k, q]`` is the share of children in group k at grid point q.
        likelihood, log_scale = _row_likelihood_on_grid(distinct, slopes, intercepts, nodes)
        prior = shares[:, None] * _group_weights_on_grid(nodes, means, covariances)
        node_prior = prior.sum(axis=0)
        row_likelihood = likelihood @ node_prior
        log_likelihood = float(counts @ (np.log(row_likelihood) + log_scale))
        if abs(log_likelihood - previous) < tol:
            converged = True
            break
        previous = log_likelihood
        per_row = counts / row_likelihood
        expected = prior * (per_row @ likelihood)  # expected children in group k at grid point q, (K, Q)

        # M-step for the groups: shares, means and covariances from the expected counts.
        group_total = expected.sum(axis=1)
        shares = group_total / n_children
        means = (expected @ nodes) / group_total[:, None]
        for k in range(n_groups):
            centered = nodes - means[k]
            covariances[k] = (centered * expected[k][:, None]).T @ centered / group_total[k]
        if shared_spread:
            covariances[:] = np.einsum("k,kde->de", shares, covariances)
        else:
            covariances = np.stack([_floor_spread(covariance, MIN_GROUP_SD) for covariance in covariances])

        # M-step for the conditions, using each grid point's expected children and expected "yes" answers.
        node_weight = expected.sum(axis=0)
        node_success = node_prior[:, None] * (likelihood.T @ (per_row[:, None] * distinct))
        slopes, intercepts = _update_conditions(slopes, intercepts, design, node_weight, node_success, max_slope)

        slopes, intercepts, means, covariances = _standardize(slopes, intercepts, shares, means, covariances)
    return slopes, intercepts, shares, means, covariances, log_likelihood, iteration, converged


def _kmeans_on_scores(scores: np.ndarray, n_groups: int, n_iter: int = 50) -> np.ndarray:
    """A few rounds of plain k-means on continuous scores; returns each child's cluster.

    Only used to place the groups for the first starting point, so it does
    not need restarts: centers begin at evenly spaced children along the
    first axis and are updated ``n_iter`` times. (``partition.cluster_kmeans``
    is for the 0/1 flags and rejects other input.)
    """
    order = np.argsort(scores[:, 0])
    centers = scores[order[np.linspace(0, len(scores) - 1, n_groups + 2).astype(int)[1:-1]]]
    labels = np.zeros(len(scores), dtype=int)
    for _ in range(n_iter):
        labels = ((scores[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        centers = np.vstack([scores[labels == k].mean(axis=0) if (labels == k).any() else centers[k] for k in range(n_groups)])
    return labels


def _starting_groups(scores: np.ndarray, n_groups: int, rng: np.random.Generator | None):
    """Starting shares, means and covariances for the groups.

    With ``rng`` None, k-means on the latent trait scores (a sensible first
    guess at where children bunch). Otherwise random means with equal shares.
    Every group starts with the identity covariance.
    """
    n_dims = scores.shape[1]
    if n_groups == 1:
        means, shares = np.zeros((1, n_dims)), np.ones(1)
    elif rng is None:
        labels = _kmeans_on_scores(scores, n_groups)
        means = np.vstack([scores[labels == k].mean(axis=0) for k in range(n_groups)])
        shares = np.clip(np.bincount(labels, minlength=n_groups) / len(labels), 0.01, None)
        shares = shares / shares.sum()
    else:
        means = rng.normal(0.0, 1.0, size=(n_groups, n_dims))
        shares = np.full(n_groups, 1.0 / n_groups)
    return shares, means, np.repeat(np.eye(n_dims)[None, :, :], n_groups, axis=0)


def _posteriors(X, slopes, intercepts, shares, means, covariances, nodes):
    """Per-row log-likelihood, group probabilities (rows, K) and posterior mean positions (rows, D)."""
    likelihood, log_scale = _row_likelihood_on_grid(X, slopes, intercepts, nodes)
    prior = shares[:, None] * _group_weights_on_grid(nodes, means, covariances)
    node_prior = prior.sum(axis=0)
    row_likelihood = likelihood @ node_prior
    group_probabilities = (likelihood @ prior.T) / row_likelihood[:, None]
    scores = ((likelihood * node_prior) @ nodes) / row_likelihood[:, None]
    return np.log(row_likelihood) + log_scale, group_probabilities, scores


# ---------------------------------------------------------------------------- #
# Fitting, scoring and simulating
# ---------------------------------------------------------------------------- #
def fit_factor_mixture(
    X: np.ndarray,
    n_groups: int,
    n_dims: int = 2,
    shared_spread: bool = True,
    n_starts: int = 10,
    short_iter: int = 200,
    n_finalists: int = 2,
    max_iter: int = 5000,
    tol: float = 1e-5,
    seed: int = 0,
    trait: LatentTraitResult | None = None,
    grid_points: int = 41,
    grid_limit: float = 5.0,
    max_slope: float = MAX_SLOPE,
) -> FactorMixtureResult:
    """Fit a factor mixture model with ``n_groups`` groups on ``n_dims`` axes to a 0/1 matrix by EM.

    Parameters
    ----------
    X : ndarray, shape (n, J), values 0 or 1
    n_groups : K. With K = 1 the model is the latent trait model.
    n_dims : D, the number of continuous axes.
    shared_spread : if True (the main model) all groups share one covariance;
        if False each group has its own.
    n_starts : starting points. The first places the groups by k-means on the
        latent trait scores; the rest draw the group means at random.
    short_iter : EM iterations given to every starting point in the first stage.
    n_finalists : how many of the best first-stage runs are continued.
    max_iter, tol : a finalist stops when the log-likelihood changes by less
        than ``tol`` in one iteration, or after ``max_iter`` more iterations.
    seed : for the random starts.
    trait : a latent trait model already fitted to ``X`` with the same
        ``n_dims``. Its slopes and intercepts start every run, so the search
        is over where the groups sit. Fitted here if not given.
    grid_points, grid_limit : the grid of positions (see the module
        docstring). The default spacing of 0.25 equals ``MIN_GROUP_SD``, the
        narrowest bell curve an even grid of that spacing represents well.
    max_slope : bound on every slope, as in the latent trait model.
    """
    X = np.asarray(X, dtype=float)
    if trait is None:
        trait = fit_latent_trait(X, n_dims=n_dims, seed=seed)
    if trait.n_dims != n_dims:
        raise ValueError(f"the latent trait model has {trait.n_dims} dimensions, not {n_dims}")
    distinct, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    inverse, counts = inverse.ravel(), counts.astype(float)
    nodes = _even_grid(n_dims, grid_points, grid_limit)
    rng = np.random.default_rng(seed)
    n_starts = 1 if n_groups == 1 else n_starts

    def run(state, iterations):
        return _run_em(distinct, counts, nodes, *state, shared_spread, iterations, tol, max_slope)

    # Stage 1: a short run from every starting point.
    short_runs = []
    for start in range(n_starts):
        groups = _starting_groups(trait.scores, n_groups, None if start == 0 else rng)
        short_runs.append(run((trait.slopes, trait.intercepts, *groups), short_iter))
    # Stage 2: carry the best few on to convergence and keep the best of those.
    short_runs.sort(key=lambda result: -result[5])
    finished = [result if result[7] else run(result[:5], max_iter) for result in short_runs[:n_finalists]]
    best = max(finished, key=lambda result: result[5])
    slopes, intercepts, shares, means, covariances, _, n_iter, converged = best

    # Report in the varimax frame, largest group first. Rotating the axes moves
    # the slopes, means and covariances together and leaves the fit unchanged.
    rotation = _varimax(slopes)
    rotated = slopes @ rotation
    rotation = rotation * np.sign(rotated[np.abs(rotated).argmax(axis=0), np.arange(n_dims)])[None, :]
    order = np.argsort(-shares)
    slopes = slopes @ rotation
    shares, means = shares[order], means[order] @ rotation
    covariances = np.einsum("ba,kbc,cd->kad", rotation, covariances[order], rotation)

    row_log_likelihood, group_probabilities, scores = _posteriors(distinct, slopes, intercepts, shares, means, covariances, nodes)
    return FactorMixtureResult(
        n_groups=n_groups,
        n_dims=n_dims,
        shared_spread=shared_spread,
        slopes=slopes,
        intercepts=intercepts,
        group_shares=shares,
        group_means=means,
        group_covariances=covariances,
        group_probabilities=group_probabilities[inverse],
        scores=scores[inverse],
        log_likelihood=float(counts @ row_log_likelihood),
        n_iter=n_iter,
        converged=converged,
        n_starts=n_starts,
        grid_points=grid_points,
        grid_limit=grid_limit,
        max_slope=max_slope,
    )


def factor_mixture_log_likelihood(result: FactorMixtureResult, X: np.ndarray, per_child: bool = False):
    """Marginal log-likelihood of new children under a fitted model, without refitting (for held-out scoring)."""
    nodes = _even_grid(result.n_dims, result.grid_points, result.grid_limit)
    rows, _, _ = _posteriors(
        np.asarray(X, dtype=float), result.slopes, result.intercepts, result.group_shares, result.group_means, result.group_covariances, nodes
    )
    return rows if per_child else float(rows.sum())


def factor_mixture_group_probabilities(result: FactorMixtureResult, X: np.ndarray) -> np.ndarray:
    """Group probabilities of new children under a fitted model, shape (n, K)."""
    nodes = _even_grid(result.n_dims, result.grid_points, result.grid_limit)
    return _posteriors(np.asarray(X, dtype=float), result.slopes, result.intercepts, result.group_shares, result.group_means, result.group_covariances, nodes)[
        1
    ]


def simulate_factor_mixture(result: FactorMixtureResult, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` fake children from a fitted model. Returns the 0/1 matrix and each child's true group."""
    rng = np.random.default_rng(seed)
    groups = rng.choice(result.n_groups, size=n, p=result.group_shares)
    theta = np.empty((n, result.n_dims))
    for k in range(result.n_groups):
        members = groups == k
        theta[members] = rng.multivariate_normal(result.group_means[k], result.group_covariances[k], size=int(members.sum()))
    p = 1 / (1 + np.exp(-(theta @ result.slopes.T + result.intercepts[None, :])))
    return (rng.random(p.shape) < p).astype(np.uint8), groups
