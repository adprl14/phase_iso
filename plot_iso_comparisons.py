#!/usr/bin/env python3
"""Generate y=x comparison plots from an ISO benchmark result file.

This script does not sample data or fit either estimator. It reads the
``estimated_models.npz`` written by ``benchmark_iso_two_stage.py`` and creates:

- each estimate against the known truth;
- all estimates against truth on a shared plot;
- pairwise estimator-versus-estimator plots;
- a manifest listing every generated figure.

Example
-------
uv run --with numpy --with matplotlib python plot_iso_comparisons.py \
    --results-dir ./iso_two_stage_results
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Mapping

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_KEYS = {
    "cvxpy_regularized": ("theta_c_cvxpy_regularized", "theta_s_cvxpy_regularized"),
    "cvxpy_refit": ("theta_c_cvxpy_refit", "theta_s_cvxpy_refit"),
    "jax_regularized": ("theta_c_jax_regularized", "theta_s_jax_regularized"),
    "jax_refit": ("theta_c_jax_refit", "theta_s_jax_refit"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create y=x plots from estimated_models.npz."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("iso_two_stage_results"),
        help="Directory containing estimated_models.npz.",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Explicit path to estimated_models.npz; overrides --results-dir.",
    )
    parser.add_argument(
        "--phase-threshold",
        type=float,
        default=1e-6,
        help="Minimum magnitude required for estimator-vs-estimator phase plots.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def upper_triangle(theta_c: np.ndarray, theta_s: np.ndarray) -> dict[str, np.ndarray]:
    if theta_c.shape != theta_s.shape:
        raise ValueError("theta_c and theta_s shapes differ.")
    if theta_c.ndim != 2 or theta_c.shape[0] != theta_c.shape[1]:
        raise ValueError("Parameter matrices must be square.")
    upper = np.triu_indices(theta_c.shape[0], k=1)
    cosine = np.asarray(theta_c[upper], dtype=np.float64)
    sine = np.asarray(theta_s[upper], dtype=np.float64)
    return {
        "theta_c": cosine,
        "theta_s": sine,
        "kappa": np.hypot(cosine, sine),
        "mu": np.arctan2(sine, cosine),
    }


def finite_limits(arrays: list[np.ndarray], *, phase: bool = False) -> tuple[float, float]:
    if phase:
        return -float(np.pi), float(np.pi)
    finite_parts = [np.asarray(a).ravel()[np.isfinite(np.asarray(a).ravel())] for a in arrays]
    finite_parts = [a for a in finite_parts if a.size]
    if not finite_parts:
        return -1.0, 1.0
    values = np.concatenate(finite_parts)
    lower = float(np.min(values))
    upper = float(np.max(values))
    if np.isclose(lower, upper):
        pad = 1.0 if np.isclose(lower, 0.0) else max(1e-6, 0.05 * abs(lower))
    else:
        pad = 0.05 * (upper - lower)
    return lower - pad, upper + pad


def save_identity_scatter(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_label: str,
    y_label: str,
    title: str,
    path: Path,
    dpi: int,
    phase: bool = False,
) -> None:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    figure = plt.figure(figsize=(6.4, 5.4))
    axis = figure.add_axes((0.14, 0.13, 0.80, 0.78))
    axis.scatter(x, y, s=14, alpha=0.7)
    lower, upper = finite_limits([x, y], phase=phase)
    axis.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.25)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def save_multi_identity_scatter(
    truth: np.ndarray,
    estimates: Mapping[str, np.ndarray],
    *,
    quantity: str,
    path: Path,
    dpi: int,
    mask: np.ndarray | None = None,
    phase: bool = False,
) -> None:
    truth = np.asarray(truth, dtype=np.float64).ravel()
    if mask is None:
        mask = np.ones(truth.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool).ravel()

    figure = plt.figure(figsize=(7.2, 6.0))
    axis = figure.add_axes((0.14, 0.13, 0.81, 0.78))
    plotted_arrays = [truth[mask]]
    for name, values in estimates.items():
        values = np.asarray(values, dtype=np.float64).ravel()
        valid = mask & np.isfinite(truth) & np.isfinite(values)
        axis.scatter(truth[valid], values[valid], s=13, alpha=0.62, label=name)
        plotted_arrays.append(values[valid])

    lower, upper = finite_limits(plotted_arrays, phase=phase)
    axis.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.25)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(f"True {quantity}")
    axis.set_ylabel(f"Estimated {quantity}")
    axis.set_title(f"All estimators versus true {quantity}")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def load_models(npz_path: Path) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Result file does not exist: {npz_path}")
    with np.load(npz_path) as archive:
        required = {"theta_c_true", "theta_s_true"}
        for pair in METHOD_KEYS.values():
            required.update(pair)
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"Missing arrays in {npz_path}: {missing}")

        truth = upper_triangle(archive["theta_c_true"], archive["theta_s_true"])
        methods = {
            name: upper_triangle(archive[c_key], archive[s_key])
            for name, (c_key, s_key) in METHOD_KEYS.items()
        }
    return truth, methods


def generate_plots(
    npz_path: Path,
    output_dir: Path,
    *,
    phase_threshold: float,
    dpi: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    truth, methods = load_models(npz_path)
    created: list[Path] = []

    # One combined truth-versus-all-estimators plot per parameterization.
    for quantity in ("theta_c", "theta_s", "kappa"):
        path = output_dir / f"comparison_true_vs_all_estimates_{quantity}.png"
        save_multi_identity_scatter(
            truth[quantity],
            {name: values[quantity] for name, values in methods.items()},
            quantity=quantity,
            path=path,
            dpi=dpi,
        )
        created.append(path)

    true_edge_mask = truth["kappa"] > 0.0
    path = output_dir / "comparison_true_vs_all_estimates_mu_true_edges.png"
    save_multi_identity_scatter(
        truth["mu"],
        {name: values["mu"] for name, values in methods.items()},
        quantity="mu",
        path=path,
        dpi=dpi,
        mask=true_edge_mask,
        phase=True,
    )
    created.append(path)

    # Individual truth-versus-estimator plots.
    for method_name, estimate in methods.items():
        for quantity in ("theta_c", "theta_s", "kappa"):
            path = output_dir / f"comparison_true_vs_{method_name}_{quantity}.png"
            save_identity_scatter(
                truth[quantity],
                estimate[quantity],
                x_label=f"True {quantity}",
                y_label=f"{method_name} {quantity}",
                title=f"{method_name} versus truth: {quantity}",
                path=path,
                dpi=dpi,
            )
            created.append(path)

        path = output_dir / f"comparison_true_vs_{method_name}_mu_true_edges.png"
        save_identity_scatter(
            truth["mu"][true_edge_mask],
            estimate["mu"][true_edge_mask],
            x_label="True mu",
            y_label=f"{method_name} mu",
            title=f"{method_name} versus truth: phase on true edges",
            path=path,
            dpi=dpi,
            phase=True,
        )
        created.append(path)

    # Every estimator pair against one another.
    for (left_name, left), (right_name, right) in combinations(methods.items(), 2):
        for quantity in ("theta_c", "theta_s", "kappa"):
            path = output_dir / f"comparison_{left_name}_vs_{right_name}_{quantity}.png"
            save_identity_scatter(
                left[quantity],
                right[quantity],
                x_label=f"{left_name} {quantity}",
                y_label=f"{right_name} {quantity}",
                title=f"{right_name} versus {left_name}: {quantity}",
                path=path,
                dpi=dpi,
            )
            created.append(path)

        phase_mask = (left["kappa"] > phase_threshold) & (right["kappa"] > phase_threshold)
        path = output_dir / f"comparison_{left_name}_vs_{right_name}_mu_selected_edges.png"
        save_identity_scatter(
            left["mu"][phase_mask],
            right["mu"][phase_mask],
            x_label=f"{left_name} mu",
            y_label=f"{right_name} mu",
            title=f"{right_name} versus {left_name}: phase on shared selected edges",
            path=path,
            dpi=dpi,
            phase=True,
        )
        created.append(path)

    manifest = output_dir / "comparison_plot_manifest.txt"
    manifest.write_text("".join(f"{path.name}\n" for path in created))
    return created


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    npz_path = args.npz.resolve() if args.npz is not None else results_dir / "estimated_models.npz"
    output_dir = results_dir
    created = generate_plots(
        npz_path,
        output_dir,
        phase_threshold=args.phase_threshold,
        dpi=args.dpi,
    )
    print(f"Read estimates from: {npz_path}")
    print(f"Generated {len(created)} y=x comparison plots in: {output_dir}")
    for path in created:
        print(path.name)
    print(f"Manifest: {output_dir / 'comparison_plot_manifest.txt'}")


if __name__ == "__main__":
    main()
