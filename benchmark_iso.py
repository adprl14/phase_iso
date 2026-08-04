#!/usr/bin/env python3
"""Benchmark one-stage and two-stage ISO estimators on a synthetic graph.

This script compares four fitted outputs on the same sampled dataset:

1. Original CVXPY regularized interaction-screening fit.
2. Original CVXPY two-stage fit:
      regularized screening -> active-set unregularized refit.
3. JAX regularized fit through iso_jax_dropin.fit.
4. JAX two-stage fit through iso_jax_dropin.fit_sparse:
      regularized screening -> active-set unregularized refit.

The two-stage procedures derive their active sets from their own regularized
fits. This matches the logic of ``iso.fit_sparse``: screen first, then refit
only the selected nodewise coefficients without the group-lasso penalty.

Runtime reporting separates:

- regularized screening;
- unregularized active-set refitting;
- total two-stage procedure time;
- JAX cold timings, which include compilation;
- JAX warm timings, which reuse compiled executables.

The benchmark is configurable in graph size, average degree, coupling range,
sample count, burn-in, thinning, and active-set threshold.

Expected files in the same directory:

    iso.py
    iso_jax_dropin.py

Example
-------
uv run \
  --with numpy \
  --with scipy \
  --with cvxpy \
  --with mosek \
  --with joblib \
  --with jax \
  --with matplotlib \
  python benchmark_iso_two_stage.py \
    --n-nodes 32 \
    --average-degree 4 \
    --n-samples 20000 \
    --burn-in 10000 \
    --thin 2 \
    --selection-threshold 1e-6 \
    --original-parallel \
    --output-dir iso_two_stage_results

Notes
-----
- ``--selection-threshold 0`` most literally matches ``iso.fit_sparse``, which
  selects entries whose post-screening magnitude is greater than zero.
  A small positive value such as 1e-6 can suppress numerical dust from a
  conic solver, especially when CVXPY falls back to SCS.
- ``--support-threshold`` is used only for reporting graph-recovery metrics.
  It does not change either estimator.
- The original code chooses MOSEK and falls back to SCS internally.
- The fitting timers exclude graph generation and Gibbs sampling.
- This file is a benchmark script only; importing it does not run the benchmark.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

# Enable float64 before importing JAX code that creates compiled functions.
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import cvxpy as cp
import jax
import jax.numpy as jnp
import matplotlib
from joblib import Parallel, delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import iso as iso_original

# Accept either the user's renamed drop-in module (iso_jax.py) or the
# distributed filename (iso_jax_dropin.py).
try:
    import iso_jax as iso_jax
except ImportError:
    import iso_jax_dropin as iso_jax

# The benchmark intentionally uses only the public, original-compatible API.
# No functions are imported from iso_jax_fast.


Array = jax.Array


@dataclass(frozen=True)
class ToyEdge:
    i: int
    j: int
    kappa: float
    mu: float


@dataclass(frozen=True)
class TimingResult:
    name: str
    seconds: float
    repeat: int


@dataclass(frozen=True)
class FitMetrics:
    method: str
    parameter_rmse: float
    theta_c_rmse: float
    theta_s_rmse: float
    kappa_rmse: float
    relative_parameter_error: float
    support_precision: float
    support_recall: float
    support_f1: float
    true_edge_phase_mae: float


@dataclass(frozen=True)
class RefitDiagnostics:
    iterations: np.ndarray
    relative_steps: np.ndarray
    lipschitz_estimates: np.ndarray
    active_degrees: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare regularized and two-stage CVXPY/JAX ISO estimators on "
            "a configurable synthetic circular graphical model."
        )
    )
    parser.add_argument("--n-nodes", type=int, default=24)
    parser.add_argument("--average-degree", type=float, default=3.0)
    parser.add_argument("--kappa-min", type=float, default=0.35)
    parser.add_argument("--kappa-max", type=float, default=0.90)
    parser.add_argument("--phase-range", type=float, default=math.pi)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--burn-in", type=int, default=5000)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--lam",
        type=float,
        default=None,
        help="Shared group-lasso penalty. Default matches iso.py.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help=(
            "Optional post-screening threshold passed to both regularized "
            "fits. The original rule zeros groups with magnitude < alpha/2."
        ),
    )
    parser.add_argument(
        "--selection-threshold",
        type=float,
        default=0.0,
        help=(
            "Regularized edge magnitude required to enter the unregularized "
            "refit. Zero most closely matches iso.fit_sparse."
        ),
    )
    parser.add_argument(
        "--support-threshold",
        type=float,
        default=0.05,
        help="Magnitude threshold used only for reported support metrics.",
    )
    parser.add_argument("--jax-tol", type=float, default=1e-7)
    parser.add_argument("--jax-maxiter", type=int, default=3000)
    parser.add_argument(
        "--jax-warm-repeats",
        type=int,
        default=3,
        help="Number of complete warm JAX two-stage repetitions.",
    )
    parser.add_argument(
        "--original-parallel",
        action="store_true",
        help="Parallelize nodewise CVXPY screening and refitting with joblib.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "iso_two_stage_results",
    )
    return parser.parse_args()


def default_lambda(n_samples: int, n_nodes: int) -> float:
    epsilon = 0.1
    return float(
        4.0 * np.sqrt(np.log(8.0 * n_nodes**2 / epsilon) / n_samples)
    )


def generate_random_connected_edges(
    n_nodes: int,
    *,
    average_degree: float,
    kappa_min: float,
    kappa_max: float,
    phase_range: float,
    seed: int,
) -> list[ToyEdge]:
    """Generate a sparse connected random graph with a fixed edge count."""
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2.")
    if average_degree <= 0:
        raise ValueError("average_degree must be positive.")
    if kappa_min <= 0 or kappa_max < kappa_min:
        raise ValueError("Require 0 < kappa_min <= kappa_max.")
    if not 0 <= phase_range <= np.pi:
        raise ValueError("phase_range must lie in [0, pi].")

    max_edges = n_nodes * (n_nodes - 1) // 2
    requested_edges = int(round(0.5 * n_nodes * average_degree))
    target_edges = min(max(requested_edges, n_nodes - 1), max_edges)

    rng = np.random.default_rng(seed)
    node_order = rng.permutation(n_nodes)
    edge_pairs: set[tuple[int, int]] = set()

    # Random recursive spanning tree guarantees connectivity.
    for position in range(1, n_nodes):
        child = int(node_order[position])
        parent = int(node_order[rng.integers(0, position)])
        edge_pairs.add(tuple(sorted((parent, child))))

    all_pairs = np.column_stack(np.triu_indices(n_nodes, k=1))
    rng.shuffle(all_pairs)
    for i_raw, j_raw in all_pairs:
        if len(edge_pairs) >= target_edges:
            break
        edge_pairs.add((int(i_raw), int(j_raw)))

    sorted_pairs = sorted(edge_pairs)
    kappas = rng.uniform(kappa_min, kappa_max, size=len(sorted_pairs))
    phases = rng.uniform(-phase_range, phase_range, size=len(sorted_pairs))
    return [
        ToyEdge(i=i, j=j, kappa=float(kappa), mu=float(mu))
        for (i, j), kappa, mu in zip(sorted_pairs, kappas, phases)
    ]


def build_model(
    n_nodes: int,
    edges: Sequence[ToyEdge],
) -> tuple[np.ndarray, np.ndarray]:
    """Construct symmetric theta_c and skew-symmetric theta_s matrices."""
    theta_c = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    theta_s = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    seen: set[tuple[int, int]] = set()
    for edge in edges:
        if not 0 <= edge.i < edge.j < n_nodes:
            raise ValueError(
                f"Each edge must satisfy 0 <= i < j < {n_nodes}; got {edge}."
            )
        if (edge.i, edge.j) in seen:
            raise ValueError(f"Duplicate edge: {(edge.i, edge.j)}")
        seen.add((edge.i, edge.j))

        cosine = edge.kappa * np.cos(edge.mu)
        sine = edge.kappa * np.sin(edge.mu)
        theta_c[edge.i, edge.j] = cosine
        theta_c[edge.j, edge.i] = cosine
        theta_s[edge.i, edge.j] = sine
        theta_s[edge.j, edge.i] = -sine

    return theta_c, theta_s


def sample_circular_graph_gibbs(
    theta_c: np.ndarray,
    theta_s: np.ndarray,
    *,
    n_samples: int,
    burn_in: int,
    thin: int,
    seed: int,
) -> np.ndarray:
    """Draw samples with incremental updates of all conditional fields."""
    theta_c = np.asarray(theta_c, dtype=np.float64)
    theta_s = np.asarray(theta_s, dtype=np.float64)

    if theta_c.shape != theta_s.shape:
        raise ValueError("theta_c and theta_s must have the same shape.")
    if theta_c.ndim != 2 or theta_c.shape[0] != theta_c.shape[1]:
        raise ValueError("theta matrices must be square.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if burn_in < 0:
        raise ValueError("burn_in must be nonnegative.")
    if thin <= 0:
        raise ValueError("thin must be positive.")
    if not np.allclose(theta_c, theta_c.T):
        raise ValueError("theta_c must be symmetric.")
    if not np.allclose(theta_s, -theta_s.T):
        raise ValueError("theta_s must be skew-symmetric.")

    rng = np.random.default_rng(seed)
    n_nodes = theta_c.shape[0]
    kappas = np.hypot(theta_c, theta_s)
    mus = np.arctan2(theta_s, theta_c)
    coupling_phasors = kappas * np.exp(-1j * mus)

    state = rng.uniform(-np.pi, np.pi, size=n_nodes)
    unit_state = np.exp(1j * state)
    conditional_fields = coupling_phasors @ unit_state

    samples = np.empty((n_samples, n_nodes), dtype=np.float64)
    total_sweeps = burn_in + n_samples * thin
    saved = 0

    for sweep in range(total_sweeps):
        for node in range(n_nodes):
            complex_field = conditional_fields[node]
            conditional_kappa = float(np.abs(complex_field))
            conditional_mu = float(np.angle(complex_field))

            old_value = unit_state[node]
            state[node] = rng.vonmises(conditional_mu, conditional_kappa)
            new_value = np.exp(1j * state[node])
            unit_state[node] = new_value
            conditional_fields += coupling_phasors[:, node] * (
                new_value - old_value
            )

        if (sweep + 1) % 256 == 0:
            conditional_fields = coupling_phasors @ unit_state

        if sweep >= burn_in and (sweep - burn_in) % thin == 0:
            samples[saved] = state
            saved += 1

    if saved != n_samples:
        raise RuntimeError(f"Expected {n_samples} samples but stored {saved}.")
    return samples


def active_sets_from_fit(
    estimate: Sequence[np.ndarray],
    *,
    selection_threshold: float,
) -> list[np.ndarray]:
    """Derive nodewise active sets from a symmetrized regularized estimate."""
    theta_c, theta_s = estimate
    kappa = np.hypot(theta_c, theta_s)
    active_sets: list[np.ndarray] = []
    for node in range(kappa.shape[0]):
        active = np.flatnonzero(kappa[node] > selection_threshold)
        active = active[active != node].astype(np.int32, copy=False)
        active_sets.append(active)
    return active_sets


def summarize_active_sets(active_sets: Sequence[np.ndarray]) -> dict[str, float]:
    degrees = np.asarray([len(active) for active in active_sets], dtype=np.int32)
    return {
        "directed_active_coefficients": int(np.sum(degrees)),
        "mean_active_degree": float(np.mean(degrees)),
        "min_active_degree": int(np.min(degrees)),
        "max_active_degree": int(np.max(degrees)),
    }


def _assemble_active_node_solutions(
    node_solutions: Sequence[np.ndarray],
    active_sets: Sequence[np.ndarray],
    n_nodes: int,
) -> list[np.ndarray]:
    """Assemble nodewise active-set solutions exactly as iso.fit_sparse."""
    theta_c_upper = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    theta_s_upper = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    for node, active in enumerate(active_sets):
        solution = np.asarray(node_solutions[node], dtype=np.float64)
        degree = len(active)
        if solution.shape != (2 * degree,):
            raise ValueError(
                f"Node {node}: expected solution shape {(2 * degree,)}, "
                f"got {solution.shape}."
            )
        for local_index, other_raw in enumerate(active):
            other = int(other_raw)
            if other < node:
                theta_c_upper[other, node] += solution[local_index]
                theta_s_upper[other, node] += solution[local_index + degree]
            elif other > node:
                theta_c_upper[node, other] += solution[local_index]
                theta_s_upper[node, other] += solution[local_index + degree]

    theta_c_upper *= 0.5
    theta_s_upper *= 0.5
    theta_c = theta_c_upper + theta_c_upper.T
    theta_s = theta_s_upper - theta_s_upper.T
    return [theta_c, theta_s]


def refit_active_sets_cvxpy(
    y: np.ndarray,
    active_sets: Sequence[np.ndarray],
    *,
    parallel: bool,
    verbose: bool = False,
) -> list[np.ndarray]:
    """Run the unregularized CVXPY refit without rerunning screening."""
    n_nodes = y.shape[1]

    def fit_one_node(node: int) -> np.ndarray:
        return iso_original.iso_sparse(
            y,
            active_sets[node],
            u=node,
            beta=None,
            alpha=None,
            verbose=verbose,
        )

    if parallel:
        try:
            n_jobs = int(os.environ["SLURM_CPUS_ON_NODE"])
        except (KeyError, ValueError):
            n_jobs = cpu_count()
        n_jobs = max(1, min(n_jobs, n_nodes))
        node_solutions = Parallel(n_jobs=n_jobs)(
            delayed(fit_one_node)(node) for node in range(n_nodes)
        )
    else:
        node_solutions = [fit_one_node(node) for node in range(n_nodes)]

    return _assemble_active_node_solutions(
        node_solutions,
        active_sets,
        n_nodes,
    )


# ---------------------------------------------------------------------------
# JAX active-set unregularized refit
# ---------------------------------------------------------------------------


def _node_design_from_trig(
    cos_y: Array,
    sin_y: Array,
    other_nodes: Array,
    node: Array,
) -> Array:
    """Construct the same oriented nodewise design matrix as iso.py."""
    cos_u = jax.lax.dynamic_slice_in_dim(cos_y, node, 1, axis=1)
    sin_u = jax.lax.dynamic_slice_in_dim(sin_y, node, 1, axis=1)
    cos_other = cos_y[:, other_nodes]
    sin_other = sin_y[:, other_nodes]

    cos_offset = cos_u * cos_other + sin_u * sin_other
    sin_offset = sin_u * cos_other - cos_u * sin_other
    orientation = jnp.where(other_nodes < node, 1.0, -1.0)
    sin_offset = sin_offset * orientation[None, :]
    return jnp.concatenate((cos_offset, sin_offset), axis=1)


def _masked_smooth_loss(
    theta: Array,
    design: Array,
) -> Array:
    return jnp.mean(jnp.exp(-(design @ theta)))


_masked_value_and_grad = jax.value_and_grad(_masked_smooth_loss)


@partial(jax.jit, static_argnames=("maxiter", "max_backtracks"))
def _solve_masked_unregularized_node(
    cos_y: Array,
    sin_y: Array,
    other_nodes: Array,
    node: Array,
    active_coordinate_mask: Array,
    tol: Array,
    maxiter: int,
    max_backtracks: int,
) -> tuple[Array, Array, Array, Array]:
    """Projected FISTA for an unregularized fit on a fixed active set.

    The optimization vector retains the full 2*(p-1) nodewise shape so all
    nodes reuse one compiled executable. Projection sets inactive cosine/sine
    coordinates to zero after every gradient step. This is equivalent to
    optimizing only over the selected coordinates.
    """
    design = _node_design_from_trig(cos_y, sin_y, other_nodes, node)
    coordinate_mask = active_coordinate_mask.astype(design.dtype)
    q = design.shape[1]

    x0 = jnp.zeros(q, dtype=design.dtype)
    y0 = x0
    t0 = jnp.array(1.0, dtype=design.dtype)
    lipschitz0 = jnp.array(1.0, dtype=design.dtype)
    error0 = jnp.array(jnp.inf, dtype=design.dtype)
    iteration0 = jnp.array(0, dtype=jnp.int32)

    def outer_condition(state: tuple[Array, ...]) -> Array:
        iteration, _x, _y, _t, _lipschitz, error = state
        return (iteration < maxiter) & (error > tol)

    def outer_step(state: tuple[Array, ...]) -> tuple[Array, ...]:
        iteration, x, accelerated, momentum, lipschitz, _error = state
        value, gradient = _masked_value_and_grad(accelerated, design)

        def candidate(current_lipschitz: Array) -> tuple[Array, Array]:
            proposed = (
                accelerated - gradient / current_lipschitz
            ) * coordinate_mask
            delta = proposed - accelerated
            quadratic_upper_bound = (
                value
                + jnp.vdot(gradient, delta).real
                + 0.5
                * current_lipschitz
                * jnp.vdot(delta, delta).real
            )
            accepted = (
                _masked_smooth_loss(proposed, design)
                <= quadratic_upper_bound + 1e-12
            )
            return proposed, accepted

        initial_proposed, initial_accepted = candidate(lipschitz)

        def backtrack_condition(bt_state: tuple[Array, ...]) -> Array:
            bt, _lip, _proposed, accepted = bt_state
            return (bt < max_backtracks) & (~accepted)

        def backtrack_step(bt_state: tuple[Array, ...]) -> tuple[Array, ...]:
            bt, current_lipschitz, _proposed, _accepted = bt_state
            new_lipschitz = 2.0 * current_lipschitz
            new_proposed, new_accepted = candidate(new_lipschitz)
            return bt + 1, new_lipschitz, new_proposed, new_accepted

        _, new_lipschitz, x_new, _ = jax.lax.while_loop(
            backtrack_condition,
            backtrack_step,
            (
                jnp.array(0, dtype=jnp.int32),
                lipschitz,
                initial_proposed,
                initial_accepted,
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

    iterations, solution, _y, _t, lipschitz, relative_step = (
        jax.lax.while_loop(
            outer_condition,
            outer_step,
            (
                iteration0,
                x0,
                y0,
                t0,
                lipschitz0,
                error0,
            ),
        )
    )
    return solution, iterations, relative_step, lipschitz


def refit_active_sets_jax(
    y: np.ndarray,
    active_sets: Sequence[np.ndarray],
    *,
    tol: float,
    maxiter: int,
    dtype: np.dtype = np.float64,
    return_diagnostics: bool = False,
):
    """Refit selected nodewise coefficients with no regularization."""
    y = np.asarray(y, dtype=dtype)
    if y.ndim != 2:
        raise ValueError("y must have shape (n_samples, n_nodes).")
    n_nodes = y.shape[1]
    if len(active_sets) != n_nodes:
        raise ValueError("active_sets must contain one array per node.")

    cos_y = jax.device_put(np.cos(y))
    sin_y = jax.device_put(np.sin(y))
    tol_device = jnp.asarray(tol, dtype=cos_y.dtype)

    full_node_solutions: list[Array] = []
    iterations: list[Array] = []
    relative_steps: list[Array] = []
    lipschitz_estimates: list[Array] = []
    active_degrees: list[int] = []

    for node in range(n_nodes):
        other_nodes_np = np.concatenate(
            (np.arange(node), np.arange(node + 1, n_nodes))
        ).astype(np.int32, copy=False)
        active_global = np.asarray(active_sets[node], dtype=np.int32)
        active_local = np.isin(other_nodes_np, active_global)
        coordinate_mask_np = np.concatenate((active_local, active_local))
        active_degrees.append(int(np.sum(active_local)))

        if not np.any(active_local):
            solution = jnp.zeros(
                2 * (n_nodes - 1),
                dtype=cos_y.dtype,
            )
            n_iter = jnp.asarray(0, dtype=jnp.int32)
            relative_step = jnp.asarray(0.0, dtype=cos_y.dtype)
            lipschitz = jnp.asarray(1.0, dtype=cos_y.dtype)
        else:
            solution, n_iter, relative_step, lipschitz = (
                _solve_masked_unregularized_node(
                    cos_y,
                    sin_y,
                    jax.device_put(other_nodes_np),
                    jnp.asarray(node, dtype=jnp.int32),
                    jax.device_put(coordinate_mask_np),
                    tol_device,
                    maxiter,
                    40,
                )
            )

        full_node_solutions.append(solution)
        iterations.append(n_iter)
        relative_steps.append(relative_step)
        lipschitz_estimates.append(lipschitz)

    full_solutions_np = np.asarray(
        jax.device_get(jnp.stack(full_node_solutions))
    )
    iterations_np = np.asarray(jax.device_get(jnp.stack(iterations)))
    relative_steps_np = np.asarray(
        jax.device_get(jnp.stack(relative_steps))
    )
    lipschitz_np = np.asarray(
        jax.device_get(jnp.stack(lipschitz_estimates))
    )

    compact_solutions: list[np.ndarray] = []
    for node in range(n_nodes):
        other_nodes_np = np.concatenate(
            (np.arange(node), np.arange(node + 1, n_nodes))
        )
        active_global = np.asarray(active_sets[node], dtype=np.int32)
        active_local = np.isin(other_nodes_np, active_global)
        d = n_nodes - 1
        cosine = full_solutions_np[node, :d][active_local]
        sine = full_solutions_np[node, d:][active_local]
        compact_solutions.append(np.concatenate((cosine, sine)))

    result = _assemble_active_node_solutions(
        compact_solutions,
        active_sets,
        n_nodes,
    )
    if not return_diagnostics:
        return result

    diagnostics = RefitDiagnostics(
        iterations=iterations_np,
        relative_steps=relative_steps_np,
        lipschitz_estimates=lipschitz_np,
        active_degrees=np.asarray(active_degrees, dtype=np.int32),
    )
    return result, diagnostics


# ---------------------------------------------------------------------------
# Timing, metrics, and output utilities
# ---------------------------------------------------------------------------


def force_numpy_materialization(value: object) -> None:
    if isinstance(value, np.ndarray):
        _ = value.shape
        return
    if isinstance(value, RefitDiagnostics):
        force_numpy_materialization(value.iterations)
        force_numpy_materialization(value.relative_steps)
        force_numpy_materialization(value.lipschitz_estimates)
        force_numpy_materialization(value.active_degrees)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            force_numpy_materialization(item)


def timed_call(function: Callable[[], object]) -> tuple[object, float]:
    gc.collect()
    start = time.perf_counter()
    result = function()
    force_numpy_materialization(result)
    return result, time.perf_counter() - start


def unpack_fit_result(
    result: object,
) -> tuple[list[np.ndarray], RefitDiagnostics | None]:
    diagnostics: RefitDiagnostics | None
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[1], RefitDiagnostics)
    ):
        matrices, diagnostics = result
    else:
        matrices, diagnostics = result, None

    if not isinstance(matrices, (list, tuple)) or len(matrices) != 2:
        raise TypeError("A fit must return [theta_c, theta_s].")
    return [
        np.asarray(matrices[0], dtype=np.float64),
        np.asarray(matrices[1], dtype=np.float64),
    ], diagnostics


def upper_triangle_parameters(
    theta_c: np.ndarray,
    theta_s: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, np.ndarray],
]:
    upper = np.triu_indices(theta_c.shape[0], k=1)
    cosine = theta_c[upper]
    sine = theta_s[upper]
    kappa = np.hypot(cosine, sine)
    mu = np.arctan2(sine, cosine)
    return cosine, sine, kappa, mu, upper


def circular_difference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (x - y)))


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def compute_metrics(
    method: str,
    estimate: Sequence[np.ndarray],
    truth: Sequence[np.ndarray],
    *,
    support_threshold: float,
) -> FitMetrics:
    true_c, true_s = truth
    est_c, est_s = estimate
    tc, ts, true_kappa, true_mu, _ = upper_triangle_parameters(true_c, true_s)
    ec, es, est_kappa, est_mu, _ = upper_triangle_parameters(est_c, est_s)

    true_parameters = np.concatenate((tc, ts))
    estimated_parameters = np.concatenate((ec, es))
    error = estimated_parameters - true_parameters

    true_support = true_kappa > 0.0
    estimated_support = est_kappa > support_threshold
    true_positive = int(np.sum(true_support & estimated_support))
    false_positive = int(np.sum(~true_support & estimated_support))
    false_negative = int(np.sum(true_support & ~estimated_support))

    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = float("nan")

    phase_errors = np.abs(
        circular_difference(est_mu[true_support], true_mu[true_support])
    )
    phase_mae = float(np.mean(phase_errors)) if phase_errors.size else float("nan")

    return FitMetrics(
        method=method,
        parameter_rmse=float(np.sqrt(np.mean(error**2))),
        theta_c_rmse=float(np.sqrt(np.mean((ec - tc) ** 2))),
        theta_s_rmse=float(np.sqrt(np.mean((es - ts) ** 2))),
        kappa_rmse=float(np.sqrt(np.mean((est_kappa - true_kappa) ** 2))),
        relative_parameter_error=float(
            np.linalg.norm(error) / max(np.linalg.norm(true_parameters), 1e-15)
        ),
        support_precision=precision,
        support_recall=recall,
        support_f1=f1,
        true_edge_phase_mae=phase_mae,
    )


def method_agreement_metrics(
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
) -> dict[str, float]:
    lc, ls, lk, lm, _ = upper_triangle_parameters(*left)
    rc, rs, rk, rm, _ = upper_triangle_parameters(*right)
    left_parameters = np.concatenate((lc, ls))
    right_parameters = np.concatenate((rc, rs))
    delta = right_parameters - left_parameters

    nonzero_for_phase = (lk > 1e-8) & (rk > 1e-8)
    phase_difference = np.abs(
        circular_difference(rm[nonzero_for_phase], lm[nonzero_for_phase])
    )
    return {
        "parameter_rmse": float(np.sqrt(np.mean(delta**2))),
        "maximum_absolute_parameter_difference": float(np.max(np.abs(delta))),
        "relative_parameter_difference": float(
            np.linalg.norm(delta) / max(np.linalg.norm(left_parameters), 1e-15)
        ),
        "kappa_rmse": float(np.sqrt(np.mean((rk - lk) ** 2))),
        "phase_mae_where_both_nonzero": (
            float(np.mean(phase_difference))
            if phase_difference.size
            else float("nan")
        ),
    }


def write_dataclass_table(path: Path, values: Iterable[object]) -> None:
    rows = [asdict(value) for value in values]
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_graph_table(path: Path, edges: Sequence[ToyEdge]) -> None:
    write_dataclass_table(path, edges)


def write_edge_table(
    path: Path,
    truth: Sequence[np.ndarray],
    estimates: dict[str, Sequence[np.ndarray]],
    *,
    support_threshold: float,
) -> None:
    tc, ts, tk, tm, upper = upper_triangle_parameters(*truth)
    extracted = {
        name: upper_triangle_parameters(*estimate)[:4]
        for name, estimate in estimates.items()
    }

    fieldnames = [
        "i",
        "j",
        "true_theta_c",
        "true_theta_s",
        "true_kappa",
        "true_mu",
        "true_edge",
    ]
    for name in estimates:
        fieldnames.extend(
            [
                f"{name}_theta_c",
                f"{name}_theta_s",
                f"{name}_kappa",
                f"{name}_mu",
                f"{name}_selected",
            ]
        )

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (i, j) in enumerate(zip(*upper)):
            row: dict[str, object] = {
                "i": int(i),
                "j": int(j),
                "true_theta_c": tc[index],
                "true_theta_s": ts[index],
                "true_kappa": tk[index],
                "true_mu": tm[index] if tk[index] > 0 else float("nan"),
                "true_edge": bool(tk[index] > 0),
            }
            for name, (cosine, sine, kappa, mu) in extracted.items():
                row.update(
                    {
                        f"{name}_theta_c": cosine[index],
                        f"{name}_theta_s": sine[index],
                        f"{name}_kappa": kappa[index],
                        f"{name}_mu": (
                            mu[index] if kappa[index] > 0 else float("nan")
                        ),
                        f"{name}_selected": bool(
                            kappa[index] > support_threshold
                        ),
                    }
                )
            writer.writerow(row)


def save_runtime_plot(
    cvx_regularized: float,
    cvx_two_stage: float,
    jax_regularized_cold: float,
    jax_two_stage_cold: float,
    jax_regularized_warm: Sequence[float],
    jax_two_stage_warm: Sequence[float],
    path: Path,
) -> None:
    names = [
        "CVXPY regularized",
        "CVXPY two-stage",
        "JAX regularized cold",
        "JAX two-stage cold",
        "JAX regularized warm",
        "JAX two-stage warm",
    ]
    values = [
        cvx_regularized,
        cvx_two_stage,
        jax_regularized_cold,
        jax_two_stage_cold,
        float(np.median(jax_regularized_warm)),
        float(np.median(jax_two_stage_warm)),
    ]
    figure = plt.figure(figsize=(10.0, 5.2))
    axis = figure.add_axes((0.10, 0.27, 0.87, 0.66))
    axis.bar(names, values)
    axis.set_ylabel("Wall-clock seconds")
    axis.set_title("Regularized and two-stage estimator runtime")
    axis.tick_params(axis="x", rotation=28)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_kappa_matrix(
    estimate: Sequence[np.ndarray],
    path: Path,
    title: str,
    common_limit: float,
) -> None:
    kappa = np.hypot(estimate[0], estimate[1])
    figure = plt.figure(figsize=(6.0, 5.2))
    axis = figure.add_axes((0.13, 0.12, 0.72, 0.78))
    image = axis.imshow(kappa, vmin=0.0, vmax=common_limit)
    axis.set_title(title)
    axis.set_xlabel("Node")
    axis.set_ylabel("Node")
    colorbar_axis = figure.add_axes((0.88, 0.12, 0.035, 0.78))
    figure.colorbar(image, cax=colorbar_axis)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def print_metric(metric: FitMetrics) -> None:
    print(f"\n{metric.method}")
    print("-" * len(metric.method))
    print(f"parameter RMSE:           {metric.parameter_rmse:.6g}")
    print(f"theta_c RMSE:             {metric.theta_c_rmse:.6g}")
    print(f"theta_s RMSE:             {metric.theta_s_rmse:.6g}")
    print(f"kappa RMSE:               {metric.kappa_rmse:.6g}")
    print(f"relative parameter error: {metric.relative_parameter_error:.6g}")
    print(f"support precision:        {metric.support_precision:.6g}")
    print(f"support recall:           {metric.support_recall:.6g}")
    print(f"support F1:               {metric.support_f1:.6g}")
    print(f"true-edge phase MAE:      {metric.true_edge_phase_mae:.6g} rad")


def main() -> None:
    args = parse_args()
    if args.n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if args.burn_in < 0:
        raise ValueError("burn_in must be nonnegative.")
    if args.thin <= 0:
        raise ValueError("thin must be positive.")
    if args.selection_threshold < 0:
        raise ValueError("selection_threshold must be nonnegative.")
    if args.support_threshold < 0:
        raise ValueError("support_threshold must be nonnegative.")
    if args.jax_warm_repeats <= 0:
        raise ValueError("jax_warm_repeats must be positive.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_seed = args.seed
    sampler_seed = args.seed + 1
    edges = generate_random_connected_edges(
        args.n_nodes,
        average_degree=args.average_degree,
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
        phase_range=args.phase_range,
        seed=graph_seed,
    )
    theta_c_true, theta_s_true = build_model(args.n_nodes, edges)
    truth = [theta_c_true, theta_s_true]
    degrees = np.count_nonzero(np.hypot(theta_c_true, theta_s_true), axis=1)
    realized_average_degree = float(np.mean(degrees))

    lam = args.lam
    if lam is None:
        lam = default_lambda(args.n_samples, args.n_nodes)

    print("Sampling circular graphical model")
    print(f"nodes:                {args.n_nodes}")
    print(f"true edges:           {len(edges)}")
    print(f"average degree:       {realized_average_degree:.3f}")
    print(f"samples:              {args.n_samples}")
    print(f"burn-in sweeps:       {args.burn_in}")
    print(f"thin:                 {args.thin}")
    print(f"lambda:               {lam:.8g}")
    print(f"alpha:                {args.alpha}")
    print(f"selection threshold:  {args.selection_threshold:.8g}")
    print(f"installed CVXPY solvers: {cp.installed_solvers()}")
    print(f"JAX API module:          {Path(iso_jax.__file__).resolve()}")

    y = sample_circular_graph_gibbs(
        theta_c_true,
        theta_s_true,
        n_samples=args.n_samples,
        burn_in=args.burn_in,
        thin=args.thin,
        seed=sampler_seed,
    )

    np.save(output_dir / "toy_samples.npy", y)
    write_graph_table(output_dir / "true_edges.csv", edges)
    np.savez(
        output_dir / "true_model.npz",
        theta_c=theta_c_true,
        theta_s=theta_s_true,
    )

    # ------------------------------------------------------------------
    # Original CVXPY: stage 1 and stage 2 timed separately.
    # ------------------------------------------------------------------
    print("\nCVXPY stage 1: regularized screening...")
    cvx_regularized_raw, cvx_regularized_seconds = timed_call(
        lambda: iso_original.fit(
            y,
            beta=None,
            alpha=args.alpha,
            lam=lam,
            parallel=args.original_parallel,
            verbose=False,
        )
    )
    cvx_regularized, _ = unpack_fit_result(cvx_regularized_raw)
    cvx_active_sets = active_sets_from_fit(
        cvx_regularized,
        selection_threshold=args.selection_threshold,
    )

    print("CVXPY two-stage fit via fit_sparse()...")
    cvx_refit_raw, cvx_two_stage_seconds = timed_call(
        lambda: iso_original.fit_sparse(
            y,
            beta=None,
            alpha=args.alpha,
            lam=lam,
            parallel=args.original_parallel,
            verbose=False,
        )
    )
    cvx_refit, _ = unpack_fit_result(cvx_refit_raw)
    cvx_refit_seconds = float("nan")

    # ------------------------------------------------------------------
    # JAX cold: compilation is included for both solver kernels.
    # ------------------------------------------------------------------
    print("\nJAX stage 1 cold: regularized screening...")
    jax_regularized_cold_raw, jax_regularized_cold_seconds = timed_call(
        lambda: iso_jax.fit(
            y,
            beta=None,
            alpha=args.alpha,
            lam=lam,
            parallel=args.original_parallel,
            verbose=False,
        )
    )
    jax_regularized_cold, _ = unpack_fit_result(
        jax_regularized_cold_raw
    )
    jax_screen_diagnostics = None
    jax_active_sets_cold = active_sets_from_fit(
        jax_regularized_cold,
        selection_threshold=args.selection_threshold,
    )

    print("JAX two-stage cold via fit_sparse()...")
    jax_refit_cold_raw, jax_two_stage_cold_seconds = timed_call(
        lambda: iso_jax.fit_sparse(
            y,
            beta=None,
            alpha=args.alpha,
            lam=lam,
            parallel=args.original_parallel,
            verbose=False,
        )
    )
    jax_refit_cold, _ = unpack_fit_result(jax_refit_cold_raw)
    jax_refit_cold_seconds = float("nan")
    jax_refit_diagnostics = None

    # ------------------------------------------------------------------
    # JAX warm repetitions: each repeat includes both stages.
    # ------------------------------------------------------------------
    jax_regularized_warm_seconds: list[float] = []
    jax_refit_warm_seconds: list[float] = []
    jax_two_stage_warm_seconds: list[float] = []
    jax_regularized_warm = jax_regularized_cold
    jax_refit_warm = jax_refit_cold
    jax_active_sets_warm = jax_active_sets_cold

    print("\nJAX warm two-stage repetitions...")
    for repeat in range(1, args.jax_warm_repeats + 1):
        screen_raw, screen_seconds = timed_call(
            lambda: iso_jax.fit(
                y,
                beta=None,
                alpha=args.alpha,
                lam=lam,
                parallel=args.original_parallel,
                verbose=False,
            )
        )
        jax_regularized_warm, _ = unpack_fit_result(screen_raw)
        screen_diagnostics = None
        jax_active_sets_warm = active_sets_from_fit(
            jax_regularized_warm,
            selection_threshold=args.selection_threshold,
        )

        two_stage_raw, total_seconds = timed_call(
            lambda: iso_jax.fit_sparse(
                y,
                beta=None,
                alpha=args.alpha,
                lam=lam,
                parallel=args.original_parallel,
                verbose=False,
            )
        )
        jax_refit_warm, _ = unpack_fit_result(two_stage_raw)

        jax_regularized_warm_seconds.append(screen_seconds)
        jax_refit_warm_seconds.append(float("nan"))
        jax_two_stage_warm_seconds.append(total_seconds)
        print(
            f"  repeat {repeat}: screening={screen_seconds:.6f} s, "
            f"two_stage_total={total_seconds:.6f} s"
        )

    estimates = {
        "cvxpy_regularized": cvx_regularized,
        "cvxpy_refit": cvx_refit,
        "jax_regularized": jax_regularized_warm,
        "jax_refit": jax_refit_warm,
    }
    metrics = [
        compute_metrics(
            name,
            estimate,
            truth,
            support_threshold=args.support_threshold,
        )
        for name, estimate in estimates.items()
    ]

    regularized_agreement = method_agreement_metrics(
        cvx_regularized,
        jax_regularized_warm,
    )
    refit_agreement = method_agreement_metrics(cvx_refit, jax_refit_warm)

    timings = [
        TimingResult("cvxpy_regularized", cvx_regularized_seconds, 1),
        TimingResult("cvxpy_refit_only", cvx_refit_seconds, 1),
        TimingResult("cvxpy_two_stage_total", cvx_two_stage_seconds, 1),
        TimingResult(
            "jax_regularized_cold",
            jax_regularized_cold_seconds,
            1,
        ),
        TimingResult("jax_refit_cold", jax_refit_cold_seconds, 1),
        TimingResult(
            "jax_two_stage_cold_total",
            jax_two_stage_cold_seconds,
            1,
        ),
    ]
    for repeat, (screen, refit, total) in enumerate(
        zip(
            jax_regularized_warm_seconds,
            jax_refit_warm_seconds,
            jax_two_stage_warm_seconds,
        ),
        start=1,
    ):
        timings.extend(
            [
                TimingResult("jax_regularized_warm", screen, repeat),
                TimingResult("jax_refit_warm", refit, repeat),
                TimingResult("jax_two_stage_warm_total", total, repeat),
            ]
        )

    warm_screen_median = float(np.median(jax_regularized_warm_seconds))
    warm_refit_median = float(np.nanmedian(jax_refit_warm_seconds))
    warm_total_median = float(np.median(jax_two_stage_warm_seconds))

    print("\nRuntime summary")
    print("---------------")
    print(f"CVXPY regularized:        {cvx_regularized_seconds:.6f} s")
    print(f"CVXPY refit only:         {cvx_refit_seconds} s (not measured separately; two-stage uses fit_sparse)")
    print(f"CVXPY two-stage total:    {cvx_two_stage_seconds:.6f} s")
    print(f"JAX regularized cold:     {jax_regularized_cold_seconds:.6f} s")
    print(f"JAX refit cold:           {jax_refit_cold_seconds} s (not measured separately; two-stage uses fit_sparse)")
    print(f"JAX two-stage cold total: {jax_two_stage_cold_seconds:.6f} s")
    print(f"JAX regularized warm med: {warm_screen_median:.6f} s")
    print(f"JAX refit warm median:    {warm_refit_median} s (not measured separately; two-stage uses fit_sparse)")
    print(f"JAX two-stage warm median:{warm_total_median: .6f} s")
    print(
        "JAX warm two-stage speedup: "
        f"{cvx_two_stage_seconds / warm_total_median:.3f}x"
    )

    print("\nActive-set summaries")
    print("--------------------")
    print("CVXPY:", summarize_active_sets(cvx_active_sets))
    print("JAX:  ", summarize_active_sets(jax_active_sets_warm))

    for metric in metrics:
        print_metric(metric)

    print("\nCVXPY versus JAX regularized agreement")
    print("----------------------------------------")
    for key, value in regularized_agreement.items():
        print(f"{key}: {value:.8g}")

    print("\nCVXPY versus JAX refit agreement")
    print("----------------------------------")
    for key, value in refit_agreement.items():
        print(f"{key}: {value:.8g}")

    np.savez(
        output_dir / "estimated_models.npz",
        theta_c_true=theta_c_true,
        theta_s_true=theta_s_true,
        theta_c_cvxpy_regularized=cvx_regularized[0],
        theta_s_cvxpy_regularized=cvx_regularized[1],
        theta_c_cvxpy_refit=cvx_refit[0],
        theta_s_cvxpy_refit=cvx_refit[1],
        theta_c_jax_regularized=jax_regularized_warm[0],
        theta_s_jax_regularized=jax_regularized_warm[1],
        theta_c_jax_refit=jax_refit_warm[0],
        theta_s_jax_refit=jax_refit_warm[1],
    )
    write_dataclass_table(output_dir / "estimation_metrics.csv", metrics)
    write_dataclass_table(output_dir / "runtime_measurements.csv", timings)
    write_edge_table(
        output_dir / "edge_estimates.csv",
        truth,
        estimates,
        support_threshold=args.support_threshold,
    )

    active_set_rows: list[dict[str, object]] = []
    for method, active_sets in (
        ("cvxpy", cvx_active_sets),
        ("jax", jax_active_sets_warm),
    ):
        for node, active in enumerate(active_sets):
            active_set_rows.append(
                {
                    "method": method,
                    "node": node,
                    "active_degree": len(active),
                    "active_nodes": " ".join(str(int(value)) for value in active),
                }
            )
    with (output_dir / "active_sets.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "node", "active_degree", "active_nodes"],
        )
        writer.writeheader()
        writer.writerows(active_set_rows)

    summary = {
        "configuration": {
            "n_nodes": args.n_nodes,
            "requested_average_degree": args.average_degree,
            "realized_average_degree": realized_average_degree,
            "n_true_edges": len(edges),
            "kappa_min": args.kappa_min,
            "kappa_max": args.kappa_max,
            "phase_range": args.phase_range,
            "n_samples": args.n_samples,
            "burn_in": args.burn_in,
            "thin": args.thin,
            "seed": args.seed,
            "lam": lam,
            "alpha": args.alpha,
            "selection_threshold": args.selection_threshold,
            "support_threshold": args.support_threshold,
            "jax_tol": args.jax_tol,
            "jax_maxiter": args.jax_maxiter,
            "jax_warm_repeats": args.jax_warm_repeats,
            "original_parallel": args.original_parallel,
            "installed_cvxpy_solvers": cp.installed_solvers(),
        },
        "active_sets": {
            "cvxpy": summarize_active_sets(cvx_active_sets),
            "jax": summarize_active_sets(jax_active_sets_warm),
        },
        "runtime_seconds": {
            "cvxpy_regularized": cvx_regularized_seconds,
            "cvxpy_refit_only": cvx_refit_seconds,
            "cvxpy_two_stage_total": cvx_two_stage_seconds,
            "jax_regularized_cold": jax_regularized_cold_seconds,
            "jax_refit_cold": jax_refit_cold_seconds,
            "jax_two_stage_cold_total": jax_two_stage_cold_seconds,
            "jax_regularized_warm": jax_regularized_warm_seconds,
            "jax_refit_warm": jax_refit_warm_seconds,
            "jax_two_stage_warm_total": jax_two_stage_warm_seconds,
            "jax_regularized_warm_median": warm_screen_median,
            "jax_refit_warm_median": warm_refit_median,
            "jax_two_stage_warm_median": warm_total_median,
            "jax_warm_two_stage_speedup": (
                cvx_two_stage_seconds / warm_total_median
            ),
        },
        "metrics": {metric.method: asdict(metric) for metric in metrics},
        "agreement": {
            "regularized": regularized_agreement,
            "refit": refit_agreement,
        },
        "jax_screening_diagnostics": None,
        "jax_refit_diagnostics": (
            {
                "iterations": jax_refit_diagnostics.iterations.tolist(),
                "relative_steps": jax_refit_diagnostics.relative_steps.tolist(),
                "lipschitz_estimates": (
                    jax_refit_diagnostics.lipschitz_estimates.tolist()
                ),
                "active_degrees": jax_refit_diagnostics.active_degrees.tolist(),
            }
            if isinstance(jax_refit_diagnostics, RefitDiagnostics)
            else None
        ),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)

    common_kappa_limit = max(
        float(np.max(np.hypot(estimate[0], estimate[1])))
        for estimate in [truth, *estimates.values()]
    )
    save_kappa_matrix(
        truth,
        output_dir / "kappa_true.png",
        "True edge magnitudes",
        common_kappa_limit,
    )
    for name, estimate in estimates.items():
        save_kappa_matrix(
            estimate,
            output_dir / f"kappa_{name}.png",
            name.replace("_", " ").title(),
            common_kappa_limit,
        )

    save_runtime_plot(
        cvx_regularized_seconds,
        cvx_two_stage_seconds,
        jax_regularized_cold_seconds,
        jax_two_stage_cold_seconds,
        jax_regularized_warm_seconds,
        jax_two_stage_warm_seconds,
        output_dir / "runtime_comparison.png",
    )

    print(f"\nSaved benchmark outputs to: {output_dir}")


if __name__ == "__main__":
    main()
