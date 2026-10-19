"""A continuous alternative to the latent class model: a two-dimensional latent trait (item response) model.

Why this exists
---------------
The latent class model says there are K kinds of children. A latent trait
model says each child has a position on one or more continuous scales, and
the probability of each condition rises smoothly with position. The two
models are fit to the same 0/1 matrix and produce comparable log-likelihoods,
so BIC can say which description the data prefer. Bauer and Curran (2004)
showed that a latent class model will "find" classes in data that are
really continuous and skewed, which is why this comparison matters for any
subtyping claim.

The model
---------
Child ``i`` has a latent position ``theta_i`` in D dimensions (D = 1 or 2
here), drawn from a standard normal. Item ``j`` has a slope vector ``a_j``
(how strongly each dimension drives the item) and an intercept ``b_j``
(how common the item is overall):

    P(x_ij = 1 | theta_i) = sigmoid(a_j . theta_i + b_j)

This is the multidimensional two-parameter logistic model, which is the same
thing as binary factor analysis. Within a child the items are independent
given ``theta``, the same local-independence assumption the latent class
model makes.

How it is fit
-------------
Marginal maximum likelihood by EM, with the normal prior replaced by a grid
of quadrature points (Gauss-Hermite). The E-step gives each child a
posterior weight over grid points; the M-step fits one weighted logistic
regression per item on the grid coordinates. The log-likelihood is the sum
over children of the log of the prior-weighted likelihood over grid points,
which is directly comparable to the latent class log-likelihood.

The slopes are only identified up to rotation (any rotation of ``theta`` with
the matching rotation of ``a`` gives the same likelihood), so the fitted
slopes are varimax-rotated for interpretation. Rotation does not change the
log-likelihood or BIC.
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
        ``b_j``. The logit of the item's probability for a child at the origin.
    scores : ndarray, shape (n, D)
        Each child's expected position (posterior mean), in the rotated frame.
    log_likelihood : float
        Marginal log-likelihood at the fitted parameters.
    n_iter : int
    converged : bool
    """

    n_dims: int
    slopes: np.ndarray
    intercepts: np.ndarray
    scores: np.ndarray
    log_likelihood: float
    n_iter: int
    converged: bool

    @property
    def n_parameters(self) -> int:
        """J slopes per dimension plus J intercepts, less the D(D-1)/2 rotational freedoms."""
        n_items = self.slopes.shape[0]
        return n_items * (self.n_dims + 1) - self.n_dims * (self.n_dims - 1) // 2

    def bic(self, n: int | None = None) -> float:
        n_children = self.scores.shape[0] if n is None else n
        return -2.0 * self.log_likelihood + self.n_parameters * np.log(n_children)


def _quadrature_grid(n_dims: int, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Tensor-product Gauss-Hermite grid for a standard normal in ``n_dims`` dimensions."""
    nodes_1d, weights_1d = hermegauss(n_points)
    weights_1d = weights_1d / weights_1d.sum()
    grids = np.meshgrid(*([nodes_1d] * n_dims), indexing="ij")
    nodes = np.column_stack([g.ravel() for g in grids])
    weights = np.ones(nodes.shape[0])
    for d in range(n_dims):
        weights *= weights_1d[np.searchsorted(nodes_1d, nodes[:, d])]
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


def fit_latent_trait(
    X: np.ndarray,
    n_dims: int = 2,
    n_quadrature: int = 15,
    max_iter: int = 300,
    tol: float = 1e-4,
    seed: int = 0,
) -> LatentTraitResult:
    """Fit a ``n_dims``-dimensional two-parameter logistic model to a 0/1 matrix by EM.

    Parameters
    ----------
    X : ndarray, shape (n, J), values 0 or 1
    n_dims : 1 or 2 (3 works but is slow; the grid has ``n_quadrature ** n_dims`` points)
    n_quadrature : grid points per dimension
    max_iter, tol : EM stopping rule on the change in log-likelihood
    seed : for the random starting slopes
    """
    X = np.asarray(X, dtype=float)
    n_children, n_items = X.shape
    nodes, prior = _quadrature_grid(n_dims, n_quadrature)  # (Q, D), (Q,)
    n_nodes = nodes.shape[0]
    design = np.column_stack([nodes, np.ones(n_nodes)])  # (Q, D+1): slopes then intercept

    rng = np.random.default_rng(seed)
    slopes = rng.uniform(0.5, 1.5, size=(n_items, n_dims))
    item_means = np.clip(X.mean(axis=0), 1e-3, 1 - 1e-3)
    intercepts = np.log(item_means / (1 - item_means))

    log_likelihood = -np.inf
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):  # noqa: B007 (used after the loop)
        # E-step: posterior weight of every grid point for every child, shape (n, Q).
        logits = nodes @ slopes.T + intercepts[None, :]  # (Q, J)
        log_p = -np.logaddexp(0.0, -logits)  # log sigmoid
        log_1mp = -np.logaddexp(0.0, logits)
        log_lik_nodes = X @ log_p.T + (1 - X) @ log_1mp.T + np.log(prior)[None, :]  # (n, Q)
        row_max = log_lik_nodes.max(axis=1, keepdims=True)
        new_log_likelihood = float((row_max.ravel() + np.log(np.exp(log_lik_nodes - row_max).sum(axis=1))).sum())
        posterior = np.exp(log_lik_nodes - row_max)
        posterior /= posterior.sum(axis=1, keepdims=True)

        # M-step: one weighted logistic regression per item on the grid.
        node_weight = posterior.sum(axis=0)  # expected children at each node, (Q,)
        node_success = posterior.T @ X  # expected 1s at each node per item, (Q, J)
        for j in range(n_items):
            beta = np.append(slopes[j], intercepts[j])
            for _ in range(10):  # Newton steps
                p = 1 / (1 + np.exp(-(design @ beta)))
                gradient = design.T @ (node_success[:, j] - node_weight * p)
                hessian = (design * (node_weight * p * (1 - p))[:, None]).T @ design + 1e-6 * np.eye(n_dims + 1)
                step = np.linalg.solve(hessian, gradient)
                beta += step
                if np.abs(step).max() < 1e-6:
                    break
            slopes[j], intercepts[j] = beta[:n_dims], beta[n_dims]

        if new_log_likelihood - log_likelihood < tol:
            log_likelihood = new_log_likelihood
            converged = True
            break
        log_likelihood = new_log_likelihood

    # Final E-step quantities at the converged parameters, then rotate for interpretation.
    scores = posterior @ nodes  # posterior mean position per child, (n, D)
    rotation = _varimax(slopes)
    slopes = slopes @ rotation
    scores = scores @ rotation
    # Orient each dimension so its largest slope is positive.
    sign = np.sign(slopes[np.abs(slopes).argmax(axis=0), np.arange(n_dims)])
    slopes *= sign
    scores *= sign

    return LatentTraitResult(
        n_dims=n_dims,
        slopes=slopes,
        intercepts=intercepts,
        scores=scores,
        log_likelihood=log_likelihood,
        n_iter=iteration,
        converged=converged,
    )


def simulate_latent_trait(result: LatentTraitResult, n: int, seed: int = 0) -> np.ndarray:
    """Draw ``n`` fake children from a fitted latent trait model (standard normal positions, independent items)."""
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal((n, result.n_dims))
    p = 1 / (1 + np.exp(-(theta @ result.slopes.T + result.intercepts[None, :])))
    return (rng.random(p.shape) < p).astype(np.uint8)
