"""Drop-in JAX replacement for the original ``iso.py`` module.

Typical migration
-----------------
Change only the import:

    import iso_jax_dropin as iso

Existing calls such as

    theta = iso.fit(Y, lam=lam, parallel=True)
    theta_sparse = iso.fit_sparse(Y, lam=lam, parallel=True)
    samples = iso.gibbs(theta_sparse, n_samples=1000)

continue to use the original public function names and argument order.

Implemented API
---------------
- iso
- fit
- iso_sparse
- fit_sparse
- dynamic_iso
- gibbs
- unnormalized_likelihood
- unnormalized_log_likelihood
- log_likelihood
- log_partition

Notes
-----
1. ``parallel`` is accepted for API compatibility but is intentionally ignored.
   JAX compiles and executes the numerical kernels; wrapping GPU work in joblib
   generally makes it slower and can cause device contention.
2. The original ``iso.py`` constructed finite-``beta`` constraints but did not
   pass them to ``cp.Problem``. Consequently, ``beta`` had no effect there.
   This compatibility module accepts ``beta`` and likewise leaves the fitted
   objective unconstrained, preserving the effective behavior of the old file.
3. ``fit_sparse`` implements the same two-stage paradigm as the original:
   group-lasso screening followed by an unregularized fit restricted to each
   node's selected active set.
4. The first call for a new array shape includes JAX compilation time. Later
   calls with the same shapes reuse compiled executables.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Sequence

import jax

# Request numerical behavior comparable to NumPy/CVXPY float64 calculations.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import scipy as sp


Array = jax.Array


@dataclass(frozen=True)
class SolverDiagnostics:
    """Optional diagnostics for internal/debug use.

    The original API does not return this object, so public functions below
    return only the same arrays/lists as the original module.
    """

    iterations: np.ndarray
    relative_steps: np.ndarray
    lipschitz_estimates: np.ndarray


def _default_lambda(n: int, p: int) -> float:
    epsilon = 0.1
    return float(4.0 * np.sqrt(np.log(8.0 * p**2 / epsilon) / n))


def _validate_y(Y: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError("Y must have shape (n_samples, n_nodes).")
    if Y.shape[0] == 0:
        raise ValueError("Y must contain at least one sample.")
    if Y.shape[1] < 2:
        raise ValueError("Y must contain at least two nodes.")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y contains NaN or infinite values.")
    return Y


def _note_ignored_beta(beta: Optional[float], verbose: bool) -> None:
    if beta is not None and verbose:
        warnings.warn(
            "beta is accepted for compatibility but is not applied. This "
            "matches the effective behavior of the original iso.py, whose "
            "constructed beta constraints were omitted from cp.Problem.",
            RuntimeWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Design matrices and optimization kernels
# ---------------------------------------------------------------------------


def _node_design_from_trig(
    cos_y: Array,
    sin_y: Array,
    other_nodes: Array,
    node: Array,
) -> Array:
    """Construct the original oriented nodewise [cos, sin] design matrix."""
    cos_u = jax.lax.dynamic_slice_in_dim(cos_y, node, 1, axis=1)
    sin_u = jax.lax.dynamic_slice_in_dim(sin_y, node, 1, axis=1)

    cos_other = cos_y[:, other_nodes]
    sin_other = sin_y[:, other_nodes]

    # Base angle difference is Y_u - Y_i.
    cos_offset = cos_u * cos_other + sin_u * sin_other
    sin_offset = sin_u * cos_other - cos_u * sin_other

    # Original convention:
    #   i < u: Y_u - Y_i
    #   i > u: Y_i - Y_u
    orientation = jnp.where(other_nodes < node, 1.0, -1.0)
    sin_offset = sin_offset * orientation[None, :]
    return jnp.concatenate((cos_offset, sin_offset), axis=1)


def _smooth_loss(theta: Array, design: Array) -> Array:
    return jnp.mean(jnp.exp(-(design @ theta)))


_smooth_value_and_grad = jax.value_and_grad(_smooth_loss)


def _group_lasso_prox(theta: Array, threshold: Array) -> Array:
    """Block soft threshold the cosine/sine coefficient pairs."""
    d = theta.shape[0] // 2
    groups = jnp.stack((theta[:d], theta[d:]), axis=1)
    norms = jnp.linalg.norm(groups, axis=1, keepdims=True)
    tiny = jnp.finfo(theta.dtype).tiny
    scales = jnp.maximum(1.0 - threshold / jnp.maximum(norms, tiny), 0.0)
    shrunk = groups * scales
    return jnp.concatenate((shrunk[:, 0], shrunk[:, 1]))


@partial(jax.jit, static_argnames=("maxiter", "max_backtracks"))
def _solve_regularized_node(
    cos_y: Array,
    sin_y: Array,
    other_nodes: Array,
    node: Array,
    lam: Array,
    tol: Array,
    maxiter: int,
    max_backtracks: int,
) -> tuple[Array, Array, Array, Array]:
    """FISTA with backtracking for one regularized nodewise problem."""
    design = _node_design_from_trig(cos_y, sin_y, other_nodes, node)
    q = design.shape[1]

    x0 = jnp.zeros(q, dtype=design.dtype)
    accelerated0 = x0
    momentum0 = jnp.array(1.0, dtype=design.dtype)
    lipschitz0 = jnp.array(1.0, dtype=design.dtype)
    error0 = jnp.array(jnp.inf, dtype=design.dtype)
    iteration0 = jnp.array(0, dtype=jnp.int32)

    def condition(state: tuple[Array, ...]) -> Array:
        iteration, _x, _accelerated, _momentum, _lipschitz, error = state
        return (iteration < maxiter) & (error > tol)

    def step(state: tuple[Array, ...]) -> tuple[Array, ...]:
        iteration, x, accelerated, momentum, lipschitz, _error = state
        value, gradient = _smooth_value_and_grad(accelerated, design)

        def candidate(current_lipschitz: Array) -> tuple[Array, Array]:
            proposed = _group_lasso_prox(
                accelerated - gradient / current_lipschitz,
                lam / current_lipschitz,
            )
            delta = proposed - accelerated
            upper_bound = (
                value
                + jnp.vdot(gradient, delta).real
                + 0.5 * current_lipschitz * jnp.vdot(delta, delta).real
            )
            accepted = _smooth_loss(proposed, design) <= upper_bound + 1e-12
            return proposed, accepted

        proposed0, accepted0 = candidate(lipschitz)

        def bt_condition(bt_state: tuple[Array, ...]) -> Array:
            bt, _lip, _proposed, accepted = bt_state
            return (bt < max_backtracks) & (~accepted)

        def bt_step(bt_state: tuple[Array, ...]) -> tuple[Array, ...]:
            bt, current_lipschitz, _proposed, _accepted = bt_state
            new_lipschitz = 2.0 * current_lipschitz
            new_proposed, new_accepted = candidate(new_lipschitz)
            return bt + 1, new_lipschitz, new_proposed, new_accepted

        _, new_lipschitz, x_new, _ = jax.lax.while_loop(
            bt_condition,
            bt_step,
            (
                jnp.array(0, dtype=jnp.int32),
                lipschitz,
                proposed0,
                accepted0,
            ),
        )

        momentum_new = 0.5 * (
            1.0 + jnp.sqrt(1.0 + 4.0 * momentum * momentum)
        )
        accelerated_new = (
            x_new
            + ((momentum - 1.0) / momentum_new) * (x_new - x)
        )
        relative_step = jnp.linalg.norm(x_new - x) / jnp.maximum(
            1.0, jnp.linalg.norm(x)
        )
        return (
            iteration + 1,
            x_new,
            accelerated_new,
            momentum_new,
            new_lipschitz,
            relative_step,
        )

    iterations, solution, _acc, _mom, lipschitz, relative_step = (
        jax.lax.while_loop(
            condition,
            step,
            (
                iteration0,
                x0,
                accelerated0,
                momentum0,
                lipschitz0,
                error0,
            ),
        )
    )
    return solution, iterations, relative_step, lipschitz


@partial(jax.jit, static_argnames=("maxiter", "max_backtracks"))
def _solve_unregularized_active_node(
    cos_y: Array,
    sin_y: Array,
    other_nodes: Array,
    node: Array,
    coordinate_mask: Array,
    tol: Array,
    maxiter: int,
    max_backtracks: int,
) -> tuple[Array, Array, Array, Array]:
    """Unregularized active-set fit using a projected full-size vector.

    Inactive coordinates are projected to exact zero at every step. Keeping the
    full 2*(p-1) shape lets all nodes reuse one compiled executable even when
    their active degrees differ.
    """
    design = _node_design_from_trig(cos_y, sin_y, other_nodes, node)
    coordinate_mask = coordinate_mask.astype(design.dtype)
    q = design.shape[1]

    x0 = jnp.zeros(q, dtype=design.dtype)
    accelerated0 = x0
    momentum0 = jnp.array(1.0, dtype=design.dtype)
    lipschitz0 = jnp.array(1.0, dtype=design.dtype)
    error0 = jnp.array(jnp.inf, dtype=design.dtype)
    iteration0 = jnp.array(0, dtype=jnp.int32)

    def condition(state: tuple[Array, ...]) -> Array:
        iteration, _x, _accelerated, _momentum, _lipschitz, error = state
        return (iteration < maxiter) & (error > tol)

    def step(state: tuple[Array, ...]) -> tuple[Array, ...]:
        iteration, x, accelerated, momentum, lipschitz, _error = state
        value, gradient = _smooth_value_and_grad(accelerated, design)

        def candidate(current_lipschitz: Array) -> tuple[Array, Array]:
            proposed = (
                accelerated - gradient / current_lipschitz
            ) * coordinate_mask
            delta = proposed - accelerated
            upper_bound = (
                value
                + jnp.vdot(gradient, delta).real
                + 0.5 * current_lipschitz * jnp.vdot(delta, delta).real
            )
            accepted = _smooth_loss(proposed, design) <= upper_bound + 1e-12
            return proposed, accepted

        proposed0, accepted0 = candidate(lipschitz)

        def bt_condition(bt_state: tuple[Array, ...]) -> Array:
            bt, _lip, _proposed, accepted = bt_state
            return (bt < max_backtracks) & (~accepted)

        def bt_step(bt_state: tuple[Array, ...]) -> tuple[Array, ...]:
            bt, current_lipschitz, _proposed, _accepted = bt_state
            new_lipschitz = 2.0 * current_lipschitz
            new_proposed, new_accepted = candidate(new_lipschitz)
            return bt + 1, new_lipschitz, new_proposed, new_accepted

        _, new_lipschitz, x_new, _ = jax.lax.while_loop(
            bt_condition,
            bt_step,
            (
                jnp.array(0, dtype=jnp.int32),
                lipschitz,
                proposed0,
                accepted0,
            ),
        )

        momentum_new = 0.5 * (
            1.0 + jnp.sqrt(1.0 + 4.0 * momentum * momentum)
        )
        accelerated_new = (
            x_new
            + ((momentum - 1.0) / momentum_new) * (x_new - x)
        ) * coordinate_mask
        relative_step = jnp.linalg.norm(x_new - x) / jnp.maximum(
            1.0, jnp.linalg.norm(x)
        )
        return (
            iteration + 1,
            x_new,
            accelerated_new,
            momentum_new,
            new_lipschitz,
            relative_step,
        )

    iterations, solution, _acc, _mom, lipschitz, relative_step = (
        jax.lax.while_loop(
            condition,
            step,
            (
                iteration0,
                x0,
                accelerated0,
                momentum0,
                lipschitz0,
                error0,
            ),
        )
    )
    return solution, iterations, relative_step, lipschitz


def _assemble_nodewise_solutions(
    node_solutions: Sequence[np.ndarray],
    other_nodes_by_node: Sequence[np.ndarray],
    p: int,
) -> list[np.ndarray]:
    """Match the original averaging/symmetrization convention."""
    theta_c_upper = np.zeros((p, p), dtype=np.float64)
    theta_s_upper = np.zeros((p, p), dtype=np.float64)

    for node, (solution, other_nodes) in enumerate(
        zip(node_solutions, other_nodes_by_node)
    ):
        solution = np.asarray(solution, dtype=np.float64)
        d = len(other_nodes)
        if solution.shape != (2 * d,):
            raise ValueError(
                f"Node {node}: expected solution shape {(2 * d,)}, "
                f"received {solution.shape}."
            )
        cosine = solution[:d]
        sine = solution[d:]
        for local_index, other_raw in enumerate(other_nodes):
            other = int(other_raw)
            if other < node:
                theta_c_upper[other, node] += cosine[local_index]
                theta_s_upper[other, node] += sine[local_index]
            elif other > node:
                theta_c_upper[node, other] += cosine[local_index]
                theta_s_upper[node, other] += sine[local_index]

    theta_c_upper *= 0.5
    theta_s_upper *= 0.5
    theta_c = theta_c_upper + theta_c_upper.T
    theta_s = theta_s_upper - theta_s_upper.T
    return [theta_c, theta_s]


# ---------------------------------------------------------------------------
# Original public fitting API
# ---------------------------------------------------------------------------


def iso(
    Y,
    u=None,
    beta=None,
    alpha=None,
    lam=None,
    verbose=False,
):
    """Fit the regularized interaction-screening objective for one node.

    Signature and returned vector layout match the original ``iso.iso``:
    cosine coefficients for all nodes except ``u``, followed by sine
    coefficients in the same node order.
    """
    Y = _validate_y(Y)
    n, p = Y.shape
    if u is None:
        raise ValueError("u must be specified for iso(Y, u=...).")
    u = int(u)
    if not 0 <= u < p:
        raise IndexError(f"u must lie in [0, {p}); received {u}.")
    _note_ignored_beta(beta, bool(verbose))
    if lam is None:
        lam = _default_lambda(n, p)

    cos_y = jax.device_put(np.cos(Y))
    sin_y = jax.device_put(np.sin(Y))
    other_nodes = np.concatenate((np.arange(u), np.arange(u + 1, p))).astype(
        np.int32,
        copy=False,
    )
    solution, *_ = _solve_regularized_node(
        cos_y,
        sin_y,
        jax.device_put(other_nodes),
        jnp.asarray(u, dtype=jnp.int32),
        jnp.asarray(lam, dtype=cos_y.dtype),
        jnp.asarray(1e-6, dtype=cos_y.dtype),
        1000,
        40,
    )
    theta = np.array(jax.device_get(solution), dtype=np.float64, copy=True)

    if alpha is not None:
        d = p - 1
        inactive = np.hypot(theta[:d], theta[d:]) < float(alpha) / 2.0
        inactive_idx = np.flatnonzero(inactive)
        theta[inactive_idx] = 0.0
        theta[d + inactive_idx] = 0.0
    return theta


def fit(
    Y,
    beta=None,
    alpha=None,
    lam=None,
    parallel=False,
    verbose=False,
):
    """Fit all nodewise regularized models using the original API."""
    del parallel  # Accepted for source compatibility; JAX owns execution.
    Y = _validate_y(Y)
    n, p = Y.shape
    _note_ignored_beta(beta, bool(verbose))
    if lam is None:
        lam = _default_lambda(n, p)

    if verbose:
        print("Fitting model with JAX")

    cos_y = jax.device_put(np.cos(Y))
    sin_y = jax.device_put(np.sin(Y))
    lam_device = jnp.asarray(lam, dtype=cos_y.dtype)
    tol_device = jnp.asarray(1e-6, dtype=cos_y.dtype)

    node_solutions: list[np.ndarray] = []
    other_nodes_by_node: list[np.ndarray] = []
    for u in range(p):
        other_nodes = np.concatenate(
            (np.arange(u), np.arange(u + 1, p))
        ).astype(np.int32, copy=False)
        solution, *_ = _solve_regularized_node(
            cos_y,
            sin_y,
            jax.device_put(other_nodes),
            jnp.asarray(u, dtype=jnp.int32),
            lam_device,
            tol_device,
            1000,
            40,
        )
        theta = np.array(jax.device_get(solution), dtype=np.float64, copy=True)
        if alpha is not None:
            d = p - 1
            inactive = np.hypot(theta[:d], theta[d:]) < float(alpha) / 2.0
            inactive_idx = np.flatnonzero(inactive)
            theta[inactive_idx] = 0.0
            theta[d + inactive_idx] = 0.0
        node_solutions.append(theta)
        other_nodes_by_node.append(other_nodes)

    return _assemble_nodewise_solutions(
        node_solutions,
        other_nodes_by_node,
        p,
    )


def iso_sparse(
    Y,
    active_set,
    u=None,
    beta=None,
    alpha=None,
    verbose=False,
):
    """Unregularized refit for one node restricted to ``active_set``.

    The returned compact vector has the same layout as the original:
    active cosine coefficients followed by active sine coefficients.
    """
    del alpha  # Present in the old signature but unused in its objective.
    Y = _validate_y(Y)
    _n, p = Y.shape
    if u is None:
        raise ValueError("u must be specified for iso_sparse(..., u=...).")
    u = int(u)
    if not 0 <= u < p:
        raise IndexError(f"u must lie in [0, {p}); received {u}.")
    _note_ignored_beta(beta, bool(verbose))

    active = np.asarray(active_set, dtype=np.int32).reshape(-1)
    if active.size == 0:
        return np.zeros(0, dtype=np.float64)
    if np.any((active < 0) | (active >= p)):
        raise IndexError("active_set contains a node index outside Y.")
    if np.any(active == u):
        raise ValueError("active_set must not include the target node u.")
    if np.unique(active).size != active.size:
        raise ValueError("active_set contains duplicate node indices.")

    # Preserve the caller's order in the returned compact vector.
    full_other_nodes = np.concatenate(
        (np.arange(u), np.arange(u + 1, p))
    ).astype(np.int32, copy=False)
    active_local = np.isin(full_other_nodes, active)
    coordinate_mask = np.concatenate((active_local, active_local))

    cos_y = jax.device_put(np.cos(Y))
    sin_y = jax.device_put(np.sin(Y))
    solution, *_ = _solve_unregularized_active_node(
        cos_y,
        sin_y,
        jax.device_put(full_other_nodes),
        jnp.asarray(u, dtype=jnp.int32),
        jax.device_put(coordinate_mask),
        jnp.asarray(1e-6, dtype=cos_y.dtype),
        3000,
        40,
    )
    full_solution = np.asarray(jax.device_get(solution), dtype=np.float64)
    d = p - 1

    # Extract coefficients in exactly the active_set order supplied by caller.
    local_lookup = {int(node): idx for idx, node in enumerate(full_other_nodes)}
    local_indices = np.asarray(
        [local_lookup[int(node)] for node in active],
        dtype=np.int32,
    )
    cosine = full_solution[:d][local_indices]
    sine = full_solution[d:][local_indices]
    return np.concatenate((cosine, sine))


def fit_sparse(
    Y,
    beta=None,
    alpha=None,
    lam=None,
    parallel=False,
    verbose=False,
):
    """Two-stage sparse estimator with the original function signature.

    Stage 1 performs group-lasso screening with ``fit``. Stage 2 performs an
    unregularized nodewise fit only on the selected support. Coefficients not
    selected at stage 1 remain exactly zero in the returned matrices.
    """
    del parallel  # Accepted for compatibility.
    Y = _validate_y(Y)
    _n, p = Y.shape

    theta_c_reg, theta_s_reg = fit(
        Y,
        beta=beta,
        alpha=alpha,
        lam=lam,
        parallel=False,
        verbose=verbose,
    )
    kappas = np.hypot(theta_c_reg, theta_s_reg)
    active_sets = [
        np.flatnonzero(kappas[u] > 0.0).astype(np.int32, copy=False)
        for u in range(p)
    ]

    if verbose:
        directed_count = int(sum(len(active) for active in active_sets))
        print(
            "Refitting sparse model with JAX "
            f"({directed_count} directed active coefficients)"
        )

    node_solutions: list[np.ndarray] = []
    for u in range(p):
        node_solutions.append(
            iso_sparse(
                Y,
                active_sets[u],
                u=u,
                beta=beta,
                alpha=alpha,
                verbose=verbose,
            )
        )

    # Assembly writes only active entries; all other entries stay exact zero.
    result = _assemble_nodewise_solutions(node_solutions, active_sets, p)

    # Explicit final support mask protects against accidental numerical fill-in.
    selected_union = np.zeros((p, p), dtype=bool)
    for u, active in enumerate(active_sets):
        selected_union[u, active] = True
    selected_union = selected_union | selected_union.T
    np.fill_diagonal(selected_union, False)
    result[0][~selected_union] = 0.0
    result[1][~selected_union] = 0.0
    return result


def dynamic_iso():
    """Placeholder retained from the original module."""
    pass


# ---------------------------------------------------------------------------
# Sampling API
# ---------------------------------------------------------------------------


def gibbs(thetas, n_samples=1000, burn_in=100, Q=1, fsave=None):
    """Sample from the fitted circular graphical model.

    The signature and returned retained-sample shape match the original. This
    implementation maintains all nodewise conditional fields incrementally,
    avoiding repeated Python construction of p-length complex arrays.
    """
    theta_c = np.asarray(thetas[0], dtype=np.float64)
    theta_s = np.asarray(thetas[1], dtype=np.float64)
    if theta_c.shape != theta_s.shape:
        raise ValueError("theta_c and theta_s must have the same shape.")
    if theta_c.ndim != 2 or theta_c.shape[0] != theta_c.shape[1]:
        raise ValueError("theta matrices must be square.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if burn_in < 0:
        raise ValueError("burn_in must be nonnegative.")
    if Q <= 0:
        raise ValueError("Q must be positive.")

    p = theta_c.shape[0]
    kappas = np.hypot(theta_c, theta_s)
    mus = np.arctan2(theta_s, theta_c)
    coupling_phasors = kappas * np.exp(-1j * mus)

    state = np.random.uniform(-np.pi, np.pi, size=p)
    unit_state = np.exp(1j * state)
    conditional_fields = coupling_phasors @ unit_state

    retained = np.empty((n_samples, p), dtype=np.float64)
    retained_count = 0
    total_sweeps = burn_in + n_samples * Q

    for sweep in range(total_sweeps):
        for u in range(p):
            field = conditional_fields[u]
            concentration = float(np.abs(field))
            mean_direction = float(np.angle(field))

            old_unit = unit_state[u]
            state[u] = np.random.vonmises(mean_direction, concentration)
            new_unit = np.exp(1j * state[u])
            unit_state[u] = new_unit
            conditional_fields += coupling_phasors[:, u] * (
                new_unit - old_unit
            )

        # Periodically correct accumulated floating-point drift.
        if (sweep + 1) % 256 == 0:
            conditional_fields = coupling_phasors @ unit_state

        if sweep >= burn_in and (sweep - burn_in) % Q == 0:
            retained[retained_count] = state
            retained_count += 1

    if retained_count != n_samples:
        raise RuntimeError(
            f"Expected {n_samples} retained samples; got {retained_count}."
        )

    if fsave is not None:
        fsave = Path(fsave)
        fsave.parent.mkdir(parents=True, exist_ok=True)
        if fsave.exists():
            existing_mat = sp.io.loadmat(fsave)
            existing_samples = np.asarray(existing_mat.get("samples", []))
            if existing_samples.size:
                retained_to_save = np.vstack((existing_samples, retained))
            else:
                retained_to_save = retained
        else:
            retained_to_save = retained
        sp.io.savemat(
            fsave,
            {
                "samples": retained_to_save,
                "theta_c": theta_c,
                "theta_s": theta_s,
            },
        )

    return retained


# ---------------------------------------------------------------------------
# Likelihood API
# ---------------------------------------------------------------------------


@jax.jit
def _unnormalized_log_likelihood_jax(
    Y: Array,
    theta_c: Array,
    theta_s: Array,
) -> Array:
    p = theta_c.shape[0]
    i, j = jnp.triu_indices(p, k=1)
    differences = Y[:, j] - Y[:, i]
    terms = (
        theta_c[i, j] * jnp.cos(differences)
        + theta_s[i, j] * jnp.sin(differences)
    )
    return jnp.sum(terms, axis=1)


def unnormalized_log_likelihood(Y, thetas):
    """Return one unnormalized log likelihood value per sample."""
    Y = _validate_y(Y)
    theta_c = np.asarray(thetas[0], dtype=np.float64)
    theta_s = np.asarray(thetas[1], dtype=np.float64)
    if theta_c.shape != (Y.shape[1], Y.shape[1]):
        raise ValueError("theta matrices are incompatible with Y.")
    values = _unnormalized_log_likelihood_jax(
        jax.device_put(Y),
        jax.device_put(theta_c),
        jax.device_put(theta_s),
    )
    return np.asarray(jax.device_get(values), dtype=np.float64)


def unnormalized_likelihood(Y, thetas):
    """Return one unnormalized likelihood value per sample."""
    return np.exp(unnormalized_log_likelihood(Y, thetas))


def log_partition(thetas, n_samples=None):
    """Monte Carlo estimate of the log partition function."""
    theta_c = np.asarray(thetas[0], dtype=np.float64)
    theta_s = np.asarray(thetas[1], dtype=np.float64)
    if theta_c.shape != theta_s.shape:
        raise ValueError("theta_c and theta_s must have the same shape.")
    if theta_c.ndim != 2 or theta_c.shape[0] != theta_c.shape[1]:
        raise ValueError("theta matrices must be square.")

    p = theta_c.shape[0]
    if n_samples is None:
        n_samples = int(5e4 * np.log(p))
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    uniform_samples = np.random.uniform(
        -np.pi,
        np.pi,
        size=(int(n_samples), p),
    )
    log_uniform_density = -p * np.log(2.0 * np.pi)
    importance_log_weights = (
        unnormalized_log_likelihood(uniform_samples, thetas)
        - log_uniform_density
    )
    return float(sp.special.logsumexp(importance_log_weights) - np.log(n_samples))


def log_likelihood(Y, thetas, log_Z=None):
    """Return the summed log likelihood, matching the original function."""
    Y = _validate_y(Y)
    if log_Z is None:
        log_Z = log_partition(
            thetas,
            n_samples=int(5e4 * np.log(Y.shape[1])),
        )
    return float(np.sum(unnormalized_log_likelihood(Y, thetas) - log_Z))


__all__ = [
    "iso",
    "fit",
    "iso_sparse",
    "fit_sparse",
    "dynamic_iso",
    "gibbs",
    "unnormalized_likelihood",
    "unnormalized_log_likelihood",
    "log_likelihood",
    "log_partition",
]
