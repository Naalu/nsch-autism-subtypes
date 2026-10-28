"""A continuous alternative to the latent class model: the multidimensional latent trait (item response) model.

Why this exists
---------------
The latent class model says there are K kinds of children. A latent trait
model says each child has a position on one or more continuous scales, and
the probability of each condition rises smoothly with that position. Both
are fit to the same 0/1 matrix and both give a log-likelihood on the same
scale, so they can be compared directly, either by BIC or by how well each
predicts children it was not fit on. Bauer and Curran (2004) showed that a
latent class model will find tidy classes in data that are continuous, which
is why this comparison matters for any subtyping claim.

The model
---------
Child ``i`` has a position ``theta_i`` in D dimensions, drawn from a standard
normal. Item ``j`` has a slope vector ``a_j`` (how strongly each dimension
drives the item) and an intercept ``b_j`` (how common the item is overall):

    P(x_ij = 1 | theta_i) = sigmoid(a_j . theta_i + b_j)

This is the multidimensional two-parameter logistic model, also called
binary factor analysis. Given ``theta`` the items are independent, the same
local-independence assumption the latent class model makes.

How it is fit
-------------
Marginal maximum likelihood by EM, with the normal distribution of ``theta``
replaced by a grid of Gauss-Hermite quadrature points. The E-step gives each
child a posterior weight over grid points; the M-step fits one weighted
logistic regression per item on the grid coordinates. Children with
identical rows are collapsed and counted once with a weight, which is exact
and makes the fit several times faster (8,891 children have only about 1,350
distinct rows).

The slopes are identified only up to rotation, so the reported slopes are
varimax-rotated for interpretation. The likelihood is invariant to rotation
in exact arithmetic; on a finite grid the difference is a few hundredths of
a log-likelihood unit, which is why the stored likelihood and the held-out
scoring use the unrotated parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite_e import hermegauss


@dataclass
class LatentTraitResult:
    """A fitted latent trait model.

    Attributes
    ----------
    n_dims : int
        D, the number of continuous dimensions.
    slopes : ndarray, shape (J, D)
        ``a_j``, varimax-rotated. Larger means the item depends more on that dimension.
    intercepts : ndarray, shape (J,)
        ``b_j``, the logit of the item's probability for a child at the origin.
    scores : ndarray, shape (n, D)
        Each child's posterior mean position, in the rotated frame.
    log_likelihood : float
        Marginal log-likelihood at the returned parameters.
    n_iter : int
        EM iterations run.
    converged : bool
        True if the last change in log-likelihood was below the tolerance.
    rotation : ndarray, shape (D, D)
        The varimax rotation applied, so ``slopes @ rotation.T`` recovers the
        fitted (unrotated) slopes. Defaults to the identity (no rotation).
    n_quadrature : int
        Grid points per dimension used in the fit.
    """

    n_dims: int
    slopes: np.ndarray
    intercepts: np.ndarray
    scores: np.ndarray
    log_likelihood: float
    n_iter: int
    converged: bool
    rotation: np.ndarray | None = None
    n_quadrature: int = 15

    def __post_init__(self) -> None:
        if self.rotation is None:
            self.rotation = np.eye(self.n_dims)

    @property
    def n_parameters(self) -> int:
        """J slopes per dimension plus J intercepts, less the D(D-1)/2 rotational freedoms."""
        n_items = self.slopes.shape[0]
        return n_items * (self.n_dims + 1) - self.n_dims * (self.n_dims - 1) // 2

    def bic(self, n: int | None = None) -> float:
        n_children = self.scores.shape[0] if n is None else n
        return -2.0 * self.log_likelihood + self.n_parameters * np.log(n_children)

    @property
    def unrotated_slopes(self) -> np.ndarray:
        return self.slopes @ self.rotation.T


def _quadrature_grid(n_dims: int, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Tensor-product Gauss-Hermite grid for a standard normal in ``n_dims`` dimensions: (nodes (Q, D), weights (Q,))."""
    nodes_1d, weights_1d = hermegauss(n_points)
    weights_1d = weights_1d / weights_1d.sum()
    index = np.indices([n_points] * n_dims).reshape(n_dims, -1).T  # every combination of 1-D node indices
    nodes = nodes_1d[index]
    weights = weights_1d[index].prod(axis=1)
    return nodes, weights


def _varimax(loadings: np.ndarray, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    """Varimax rotation matrix for a (J, D) loading matrix (Kaiser 1958)."""
    n_items, n_dims = loadings.shape
    rotation = np.eye(n_dims)
    if n_dims == 1:
        return rotation
    variance = 0.0
    for _ in range(max_iter):
        rotated = loadings @ rotation
        u, s, vt = np.linalg.svd(loadings.T @ (rotated**3 - rotated @ np.diag((rotated**2).sum(axis=0)) / n_items))
        rotation = u @ vt
        new_variance = s.sum()
        if new_variance - variance < tol:
            break
        variance = new_variance
    return rotation


def _e_step(X: np.ndarray, counts: np.ndarray, slopes: np.ndarray, intercepts: np.ndarray, nodes: np.ndarray, prior: np.ndarray):
    """Posterior over grid points for each distinct row, and the total marginal log-likelihood."""
    logits = nodes @ slopes.T + intercepts[None, :]  # (Q, J)
    log_p = -np.logaddexp(0.0, -logits)  # log sigmoid
    log_1mp = -np.logaddexp(0.0, logits)
    log_joint = X @ log_p.T + (1 - X) @ log_1mp.T + np.log(prior)[None, :]  # (rows, Q)
    row_max = log_joint.max(axis=1, keepdims=True)
    row_log_lik = row_max.ravel() + np.log(np.exp(log_joint - row_max).sum(axis=1))
    posterior = np.exp(log_joint - row_log_lik[:, None])
    return float((counts * row_log_lik).sum()), posterior, row_log_lik


def _polish(distinct, counts, slopes, intercepts, nodes, prior, design, gtol: float = 1e-3):
    """Finish the fit with L-BFGS on the marginal log-likelihood. Returns (slopes, intercepts, converged)."""
    from scipy.optimize import minimize

    n_items, n_dims = slopes.shape

    def objective(theta: np.ndarray):
        beta = theta.reshape(n_items, n_dims + 1)
        ll, posterior, _ = _e_step(distinct, counts, beta[:, :n_dims], beta[:, n_dims], nodes, prior)
        weighted = posterior * counts[:, None]
        node_weight = weighted.sum(axis=0)
        node_success = weighted.T @ distinct
        p = 1 / (1 + np.exp(-(design @ beta.T)))  # (Q, J)
        gradient = (node_success - node_weight[:, None] * p).T @ design  # (J, D+1)
        return -ll, -gradient.ravel()

    start = np.column_stack([slopes, intercepts]).ravel()
    result = minimize(objective, start, jac=True, method="L-BFGS-B", options={"maxiter": 5000, "gtol": gtol})
    beta = result.x.reshape(n_items, n_dims + 1)
    gradient_norm = float(np.abs(objective(result.x)[1]).max())
    return beta[:, :n_dims].copy(), beta[:, n_dims].copy(), bool(result.success or gradient_norm < gtol)


def fit_latent_trait(
    X: np.ndarray,
    n_dims: int = 2,
    n_quadrature: int | None = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    seed: int = 0,
) -> LatentTraitResult:
    """Fit a ``n_dims``-dimensional two-parameter logistic model to a 0/1 matrix by EM.

    Parameters
    ----------
    X : ndarray, shape (n, J), values 0 or 1
    n_dims : number of dimensions. The grid has ``n_quadrature ** n_dims``
        points, so 4 is about the practical limit.
    n_quadrature : grid points per dimension. Default 21, 15, 11, 7 for
        D = 1, 2, 3, 4 and above.
    max_iter : EM iterations before handing over to L-BFGS. EM gets close
        quickly and then crawls, so this is a warm start, not a cap on accuracy.
    tol : stop EM early if the log-likelihood changes by less than this.
        ``converged`` on the result reports whether the final optimizer
        reached a point where the gradient is essentially zero.
    seed : for the random starting slopes.
    """
    if n_quadrature is None:
        n_quadrature = {1: 21, 2: 15, 3: 11}.get(n_dims, 7)
    X = np.asarray(X, dtype=float)
    distinct, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    counts = counts.astype(float)
    n_items = X.shape[1]

    nodes, prior = _quadrature_grid(n_dims, n_quadrature)
    design = np.column_stack([nodes, np.ones(nodes.shape[0])])  # (Q, D+1): slopes then intercept

    rng = np.random.default_rng(seed)
    slopes = rng.uniform(0.5, 1.5, size=(n_items, n_dims))
    item_means = np.clip(X.mean(axis=0), 1e-3, 1 - 1e-3)
    intercepts = np.log(item_means / (1 - item_means))

    previous = -np.inf
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):  # noqa: B007 (reported after the loop)
        log_likelihood, posterior, _ = _e_step(distinct, counts, slopes, intercepts, nodes, prior)
        if abs(log_likelihood - previous) < tol:
            converged = True
            break
        previous = log_likelihood

        # M-step: one weighted logistic regression per item on the grid, a few Newton steps each.
        weighted_posterior = posterior * counts[:, None]
        node_weight = weighted_posterior.sum(axis=0)  # expected children at each node, (Q,)
        node_success = weighted_posterior.T @ distinct  # expected 1s at each node per item, (Q, J)
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
            slopes[j], intercepts[j] = beta[:n_dims], beta[n_dims]

    # EM for this model crawls near the optimum (the likelihood can still be
    # rising by a few units after thousands of iterations). Finish with a
    # quasi-Newton optimizer on the marginal likelihood. Its gradient comes
    # from the same posterior the E-step computes (Fisher's identity), so it
    # is exact and cheap.
    if not converged:
        slopes, intercepts, converged = _polish(distinct, counts, slopes, intercepts, nodes, prior, design)

    # Likelihood and posterior at the parameters actually returned.
    log_likelihood, posterior, _ = _e_step(distinct, counts, slopes, intercepts, nodes, prior)
    scores = (posterior @ nodes)[inverse]  # posterior mean position per child, (n, D)

    rotation = _varimax(slopes)
    rotated = slopes @ rotation
    sign = np.sign(rotated[np.abs(rotated).argmax(axis=0), np.arange(n_dims)])
    rotation = rotation * sign[None, :]  # fold the sign flip into the rotation so it stays orthogonal

    return LatentTraitResult(
        n_dims=n_dims,
        slopes=slopes @ rotation,
        intercepts=intercepts,
        scores=scores @ rotation,
        log_likelihood=log_likelihood,
        n_iter=iteration,
        converged=converged,
        rotation=rotation,
        n_quadrature=n_quadrature,
    )


def latent_trait_log_likelihood(result: LatentTraitResult, X: np.ndarray, per_child: bool = False):
    """Marginal log-likelihood of new children under a fitted model, without refitting (for held-out scoring)."""
    X = np.asarray(X, dtype=float)
    nodes, prior = _quadrature_grid(result.n_dims, result.n_quadrature)
    total, _, rows = _e_step(X, np.ones(X.shape[0]), result.unrotated_slopes, result.intercepts, nodes, prior)
    return rows if per_child else total


def latent_trait_pair_probabilities(result: LatentTraitResult) -> tuple[np.ndarray, np.ndarray]:
    """Model-implied P(x_j = 1) for each item and P(x_j = 1, x_k = 1) for each pair, by quadrature."""
    nodes, prior = _quadrature_grid(result.n_dims, result.n_quadrature)
    p = 1 / (1 + np.exp(-(nodes @ result.unrotated_slopes.T + result.intercepts[None, :])))  # (Q, J)
    single = prior @ p
    joint = (p * prior[:, None]).T @ p
    np.fill_diagonal(joint, single)  # an item "with itself" is just the item
    return single, joint


def simulate_latent_trait(result: LatentTraitResult, n: int, seed: int = 0) -> np.ndarray:
    """Draw ``n`` fake children from a fitted latent trait model (standard normal positions, independent items)."""
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal((n, result.n_dims))
    p = 1 / (1 + np.exp(-(theta @ result.slopes.T + result.intercepts[None, :])))
    return (rng.random(p.shape) < p).astype(np.uint8)


# ---------------------------------------------------------------------------- #
# What does the cloud of children look like? (estimated latent density)
# ---------------------------------------------------------------------------- #
@dataclass
class LatentDensity:
    """The distribution of children's positions, estimated from the data instead of assumed normal.

    Attributes
    ----------
    nodes : ndarray, shape (Q, D)
        An evenly spaced grid of positions, in the same (rotated) frame as the
        model's slopes, so dimension 1 and 2 mean what the loadings say.
    weights : ndarray, shape (Q,)
        Estimated share of children at each grid point. Sums to 1.
    normal_weights : ndarray, shape (Q,)
        The standard normal the model assumes, on the same grid, for comparison.
    log_likelihood, normal_log_likelihood : float
        Marginal log-likelihood of the data under each.
    n_iter : int
    converged : bool
    """

    nodes: np.ndarray
    weights: np.ndarray
    normal_weights: np.ndarray
    log_likelihood: float
    normal_log_likelihood: float
    n_iter: int
    converged: bool

    @property
    def improvement(self) -> float:
        """How much better the estimated density explains the data than the assumed normal (log-likelihood units)."""
        return self.log_likelihood - self.normal_log_likelihood

    @property
    def grid_shape(self) -> tuple[int, ...]:
        side = round(len(self.weights) ** (1 / self.nodes.shape[1]))
        return (side,) * self.nodes.shape[1]


def _even_grid(n_dims: int, n_points: int, limit: float) -> np.ndarray:
    """An evenly spaced grid on [-limit, limit] in each dimension, shape (n_points ** n_dims, n_dims), first dimension varying slowest."""
    axis = np.linspace(-limit, limit, n_points)
    index = np.indices([n_points] * n_dims).reshape(n_dims, -1).T
    return axis[index]


def _row_log_likelihood_on_grid(X: np.ndarray, result: LatentTraitResult, nodes: np.ndarray) -> np.ndarray:
    """log P(row | position) for every row and grid point, shape (rows, Q), using the rotated slopes."""
    logits = nodes @ result.slopes.T + result.intercepts[None, :]
    return X @ (-np.logaddexp(0.0, -logits)).T + (1 - X) @ (-np.logaddexp(0.0, logits)).T


def fit_latent_density(
    result: LatentTraitResult,
    X: np.ndarray,
    n_points: int = 41,
    limit: float = 4.0,
    max_iter: int = 5000,
    tol: float = 1e-7,
) -> LatentDensity:
    """Estimate where children sit in the latent space without assuming a normal distribution.

    The latent trait model assumes positions follow a standard normal. If the
    children really fell into a few groups, the positions would instead pile
    up in a few places. This function keeps the fitted slopes and intercepts
    fixed and lets the data choose the weight on every point of an even grid
    (the nonparametric maximum likelihood estimate of a mixing distribution;
    Laird 1978). Each EM step sets a grid point's weight to the average
    posterior probability of that point across children.

    The estimate is always somewhat lumpy, even when the truth is smooth,
    because maximum likelihood puts mass on a few points when it can. So the
    picture and the ``improvement`` over the normal only mean something next
    to the same estimate on simulated data with a known shape, which is what
    ``scripts/14_shape_and_stability.py`` does.

    Holding the item parameters fixed at their normal-theory values is a
    simplification: it keeps the axes and scale identified, at the cost of
    favoring shapes the normal fit can express.
    """
    X = np.asarray(X, dtype=float)
    distinct, counts = np.unique(X, axis=0, return_counts=True)
    counts = counts.astype(float)
    nodes = _even_grid(result.n_dims, n_points, limit)
    normal = np.exp(-0.5 * (nodes**2).sum(axis=1))
    normal /= normal.sum()

    log_lik = _row_log_likelihood_on_grid(distinct, result, nodes)
    row_max = log_lik.max(axis=1, keepdims=True)
    lik = np.exp(log_lik - row_max)  # scaled likelihood, (rows, Q)

    def total(weights: np.ndarray) -> tuple[float, np.ndarray]:
        mix = lik @ weights
        return float((counts * (np.log(mix) + row_max.ravel())).sum()), mix

    normal_ll, mix = total(normal)
    weights, previous, converged, n_iter = normal.copy(), normal_ll, False, 0
    while n_iter < max_iter and not converged:
        weights = weights * ((counts / mix) @ lik) / counts.sum()  # EM: new weight = average posterior at that point
        current, mix = total(weights)
        converged = current - previous < tol * abs(current)
        previous = current
        n_iter += 1
    return LatentDensity(nodes, weights, normal, previous, normal_ll, n_iter, converged)


def draw_positions(
    result: LatentTraitResult,
    X: np.ndarray,
    n_draws: int = 1,
    seed: int = 0,
    n_points: int = 61,
    limit: float = 4.0,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Random draws of each child's position from its posterior ("plausible values"), shape (n, n_draws, D).

    For plotting children in the latent space. Posterior means bunch up
    (8,891 children have about 1,350 distinct rows, so there are only that
    many distinct means, and they are pulled toward the center), which can
    make a scatter plot look clustered when it is not. A draw from each
    child's posterior has the right spread. Each draw picks a grid point in
    proportion to its posterior probability and adds uniform jitter within
    the grid cell. ``weights`` replaces the normal prior (for example with
    :func:`fit_latent_density` weights on the same grid).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    nodes = _even_grid(result.n_dims, n_points, limit)
    step = 2 * limit / (n_points - 1)
    prior = np.exp(-0.5 * (nodes**2).sum(axis=1)) if weights is None else np.asarray(weights, dtype=float)
    prior = prior / prior.sum()
    distinct, row_of_child = np.unique(X, axis=0, return_inverse=True)
    log_post = _row_log_likelihood_on_grid(distinct, result, nodes) + np.log(np.clip(prior, 1e-300, None))[None, :]
    post = np.exp(log_post - log_post.max(axis=1, keepdims=True))
    post /= post.sum(axis=1, keepdims=True)
    rows = row_of_child.ravel()
    picks = np.empty((len(rows), n_draws), dtype=int)
    for r in range(len(distinct)):  # children with the same row share a posterior
        children = np.flatnonzero(rows == r)
        picks[children] = rng.choice(len(nodes), size=(len(children), n_draws), p=post[r])
    jitter = rng.uniform(-step / 2, step / 2, size=(len(rows), n_draws, result.n_dims))
    return nodes[picks] + jitter
