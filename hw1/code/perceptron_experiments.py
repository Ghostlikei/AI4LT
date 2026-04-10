#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


RADIUS_BOUND = 1.0
BASE_SEED = 20260410


def perceptron_online(x: np.ndarray, y: np.ndarray) -> tuple[int, np.ndarray]:
    """Run one online pass and count mistakes."""
    w = np.zeros(x.shape[1], dtype=float)
    mistakes = 0
    for x_t, y_t in zip(x, y):
        if y_t * float(np.dot(w, x_t)) <= 0.0:
            w += y_t * x_t
            mistakes += 1
    return mistakes, w


def perceptron_until_converged(
    x: np.ndarray, y: np.ndarray, max_epochs: int = 50
) -> tuple[int, np.ndarray]:
    """Repeated passes used only for deterministic sanity checks."""
    w = np.zeros(x.shape[1], dtype=float)
    total_mistakes = 0
    for _ in range(max_epochs):
        epoch_mistakes = 0
        for x_t, y_t in zip(x, y):
            if y_t * float(np.dot(w, x_t)) <= 0.0:
                w += y_t * x_t
                epoch_mistakes += 1
        total_mistakes += epoch_mistakes
        if epoch_mistakes == 0:
            return total_mistakes, w
    raise RuntimeError("Perceptron did not converge during the sanity check.")


def _orthogonal_unit_vector(
    rng: np.random.Generator, dimension: int, separator: np.ndarray
) -> np.ndarray:
    if dimension == 1:
        return np.zeros(1, dtype=float)

    candidate = rng.normal(size=dimension)
    candidate -= float(np.dot(candidate, separator)) * separator
    norm = float(np.linalg.norm(candidate))
    if norm < 1e-12:
        candidate = np.zeros(dimension, dtype=float)
        candidate[1] = 1.0
        return candidate
    return candidate / norm


def generate_margin_sequence(
    n_examples: int,
    dimension: int,
    gamma: float,
    radius_bound: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a separable sequence with a known margin gamma and norm bound R.

    We fix the true separator to u = e_1. Every point has signed margin exactly
    gamma, and its norm is at most radius_bound.
    """
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if gamma > radius_bound:
        raise ValueError("gamma cannot exceed the norm bound")
    if dimension < 1:
        raise ValueError("dimension must be at least 1")

    separator = np.zeros(dimension, dtype=float)
    separator[0] = 1.0

    labels = rng.choice(np.array([-1, 1], dtype=int), size=n_examples)
    points = np.empty((n_examples, dimension), dtype=float)
    max_orthogonal_radius = math.sqrt(max(radius_bound**2 - gamma**2, 0.0))

    for index, label in enumerate(labels):
        base_point = label * gamma * separator
        if dimension == 1 or max_orthogonal_radius == 0.0:
            points[index] = base_point
            continue

        orthogonal_direction = _orthogonal_unit_vector(rng, dimension, separator)
        orthogonal_radius = max_orthogonal_radius * math.sqrt(rng.uniform(0.0, 1.0))
        points[index] = base_point + orthogonal_radius * orthogonal_direction

    return points, labels, separator


def validate_generator(
    points: np.ndarray,
    labels: np.ndarray,
    separator: np.ndarray,
    gamma: float,
    radius_bound: float,
) -> tuple[float, float]:
    signed_margins = labels * (points @ separator)
    norms = np.linalg.norm(points, axis=1)
    min_margin = float(np.min(signed_margins))
    max_norm = float(np.max(norms))

    if min_margin < gamma - 1e-9:
        raise AssertionError(f"Generator violated the target margin: {min_margin} < {gamma}")
    if max_norm > radius_bound + 1e-9:
        raise AssertionError(
            f"Generator violated the norm bound: {max_norm} > {radius_bound}"
        )
    return min_margin, max_norm


def run_sanity_checks() -> None:
    # A single mistaken positive example from zero weights should copy x into w.
    x_single = np.array([[2.0, -1.0]])
    y_single = np.array([1])
    mistakes, weights = perceptron_online(x_single, y_single)
    assert mistakes == 1
    assert np.allclose(weights, x_single[0])

    # The generator should honor the declared margin and norm bound.
    rng = np.random.default_rng(BASE_SEED)
    x_gen, y_gen, separator = generate_margin_sequence(
        n_examples=256, dimension=10, gamma=0.2, radius_bound=RADIUS_BOUND, rng=rng
    )
    min_margin, max_norm = validate_generator(
        x_gen, y_gen, separator, gamma=0.2, radius_bound=RADIUS_BOUND
    )
    assert min_margin >= 0.2 - 1e-9
    assert max_norm <= RADIUS_BOUND + 1e-9

    # On a tiny separable dataset, repeated passes should end with zero training errors.
    x_tiny = np.array(
        [
            [1.5, 0.2],
            [1.2, -0.1],
            [-1.0, 0.3],
            [-1.4, -0.5],
        ]
    )
    y_tiny = np.array([1, 1, -1, -1])
    _, final_w = perceptron_until_converged(x_tiny, y_tiny)
    assert np.all(y_tiny * (x_tiny @ final_w) > 0.0)


def run_trials(
    *,
    gamma: float,
    dimension: int,
    n_examples: int,
    n_trials: int,
    radius_bound: float,
    seed_offset: int,
) -> np.ndarray:
    mistake_counts = np.empty(n_trials, dtype=float)
    for trial in range(n_trials):
        rng = np.random.default_rng(BASE_SEED + seed_offset + trial)
        x, y, separator = generate_margin_sequence(
            n_examples=n_examples,
            dimension=dimension,
            gamma=gamma,
            radius_bound=radius_bound,
            rng=rng,
        )
        validate_generator(x, y, separator, gamma=gamma, radius_bound=radius_bound)
        mistake_counts[trial], _ = perceptron_online(x, y)
    return mistake_counts


def summarize_trials(mistakes: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(mistakes)),
        "std": float(np.std(mistakes, ddof=0)),
        "min": float(np.min(mistakes)),
        "max": float(np.max(mistakes)),
    }


def run_gamma_sweep() -> dict[str, object]:
    gamma_values = np.array([0.50, 0.35, 0.25, 0.18, 0.14, 0.11, 0.09, 0.07, 0.055])
    means = []
    stds = []
    trial_lengths = []
    bounds = []

    for index, gamma in enumerate(gamma_values):
        theoretical_bound = (RADIUS_BOUND**2) / (gamma**2)
        n_examples = max(300, int(math.ceil(8.0 * theoretical_bound)))
        mistakes = run_trials(
            gamma=float(gamma),
            dimension=25,
            n_examples=n_examples,
            n_trials=48,
            radius_bound=RADIUS_BOUND,
            seed_offset=10_000 * index,
        )
        summary = summarize_trials(mistakes)
        means.append(summary["mean"])
        stds.append(summary["std"])
        trial_lengths.append(n_examples)
        bounds.append(theoretical_bound)

    mean_array = np.array(means)
    inv_gamma_sq = 1.0 / (gamma_values**2)
    slope, intercept = np.polyfit(np.log(inv_gamma_sq), np.log(mean_array), deg=1)

    return {
        "gamma_values": gamma_values.tolist(),
        "inverse_gamma_sq": inv_gamma_sq.tolist(),
        "trial_lengths": trial_lengths,
        "mean_mistakes": mean_array.tolist(),
        "std_mistakes": stds,
        "theoretical_bounds": bounds,
        "loglog_slope": float(slope),
        "loglog_intercept": float(intercept),
        "mean_ratio_to_bound": float(np.mean(mean_array / np.array(bounds))),
        "max_ratio_to_bound": float(np.max(mean_array / np.array(bounds))),
    }


def run_dimension_sweep() -> dict[str, object]:
    dimensions = np.array([2, 5, 10, 20, 50, 100, 200])
    gamma = 0.08
    theoretical_bound = (RADIUS_BOUND**2) / (gamma**2)
    n_examples = max(400, int(math.ceil(10.0 * theoretical_bound)))
    means = []
    stds = []

    for index, dimension in enumerate(dimensions):
        mistakes = run_trials(
            gamma=gamma,
            dimension=int(dimension),
            n_examples=n_examples,
            n_trials=36,
            radius_bound=RADIUS_BOUND,
            seed_offset=200_000 + 10_000 * index,
        )
        summary = summarize_trials(mistakes)
        means.append(summary["mean"])
        stds.append(summary["std"])

    mean_array = np.array(means)
    return {
        "dimensions": dimensions.tolist(),
        "gamma": gamma,
        "trial_length": n_examples,
        "theoretical_bound": theoretical_bound,
        "mean_mistakes": mean_array.tolist(),
        "std_mistakes": stds,
        "relative_range": float((np.max(mean_array) - np.min(mean_array)) / np.mean(mean_array)),
    }


def run_small_gamma_sweep() -> dict[str, object]:
    gamma_values = np.array([0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03])
    means = []
    stds = []
    bounds = []
    trial_lengths = []

    for index, gamma in enumerate(gamma_values):
        theoretical_bound = (RADIUS_BOUND**2) / (gamma**2)
        n_examples = max(500, int(math.ceil(6.0 * theoretical_bound)))
        mistakes = run_trials(
            gamma=float(gamma),
            dimension=25,
            n_examples=n_examples,
            n_trials=24,
            radius_bound=RADIUS_BOUND,
            seed_offset=400_000 + 10_000 * index,
        )
        summary = summarize_trials(mistakes)
        means.append(summary["mean"])
        stds.append(summary["std"])
        bounds.append(theoretical_bound)
        trial_lengths.append(n_examples)

    return {
        "gamma_values": gamma_values.tolist(),
        "trial_lengths": trial_lengths,
        "mean_mistakes": means,
        "std_mistakes": stds,
        "theoretical_bounds": bounds,
    }


def _save_gamma_scaling_plot(summary: dict[str, object], output_path: Path) -> None:
    x_values = np.array(summary["inverse_gamma_sq"], dtype=float)
    mean_mistakes = np.array(summary["mean_mistakes"], dtype=float)
    std_mistakes = np.array(summary["std_mistakes"], dtype=float)
    bounds = np.array(summary["theoretical_bounds"], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_values, mean_mistakes, "o-", linewidth=2, label="Average mistakes")
    ax.fill_between(
        x_values,
        np.maximum(mean_mistakes - std_mistakes, 0.0),
        mean_mistakes + std_mistakes,
        alpha=0.15,
        label="one standard deviation",
    )
    ax.plot(x_values, bounds, "s--", linewidth=2, label=r"theory bound $R^2/\gamma^2$")
    ax.set_xlabel(r"$1/\gamma^2$")
    ax.set_ylabel("mistakes")
    ax.set_title("Perceptron mistakes versus inverse margin squared")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_bound_ratio_plot(summary: dict[str, object], output_path: Path) -> None:
    gamma_values = np.array(summary["gamma_values"], dtype=float)
    mean_mistakes = np.array(summary["mean_mistakes"], dtype=float)
    bounds = np.array(summary["theoretical_bounds"], dtype=float)
    ratios = mean_mistakes / bounds

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gamma_values, ratios, "o-", linewidth=2, color="#b85450")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"margin $\gamma$")
    ax.set_ylabel(r"average mistake ratio $M / (R^2/\gamma^2)$")
    ax.set_title("How loose is the theoretical bound?")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_dimension_plot(summary: dict[str, object], output_path: Path) -> None:
    dimensions = np.array(summary["dimensions"], dtype=float)
    mean_mistakes = np.array(summary["mean_mistakes"], dtype=float)
    std_mistakes = np.array(summary["std_mistakes"], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        dimensions,
        mean_mistakes,
        yerr=std_mistakes,
        fmt="o-",
        linewidth=2,
        capsize=4,
        color="#2f6db3",
    )
    ax.set_xscale("log")
    ax.set_xlabel("dimension d")
    ax.set_ylabel("mistakes")
    ax.set_title("Perceptron mistakes are nearly dimension-independent at fixed margin")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_small_gamma_plot(summary: dict[str, object], output_path: Path) -> None:
    gamma_values = np.array(summary["gamma_values"], dtype=float)
    mean_mistakes = np.array(summary["mean_mistakes"], dtype=float)
    bounds = np.array(summary["theoretical_bounds"], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gamma_values, mean_mistakes, "o-", linewidth=2, label="Average mistakes")
    ax.plot(gamma_values, bounds, "s--", linewidth=2, label=r"theory bound $R^2/\gamma^2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"margin $\gamma$")
    ax.set_ylabel("mistakes")
    ax.set_title(r"Behavior as $\gamma \to 0$")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_figures(summary: dict[str, object], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    _save_gamma_scaling_plot(summary["gamma_sweep"], figures_dir / "mistakes_vs_inv_gamma2.png")
    _save_bound_ratio_plot(summary["gamma_sweep"], figures_dir / "bound_ratio_vs_gamma.png")
    _save_dimension_plot(summary["dimension_sweep"], figures_dir / "mistakes_vs_dimension.png")
    _save_small_gamma_plot(summary["small_gamma_sweep"], figures_dir / "mistakes_small_gamma.png")


def main() -> None:
    run_sanity_checks()

    assignment_dir = Path(__file__).resolve().parent.parent
    figures_dir = assignment_dir / "figures"
    summary_path = Path(__file__).resolve().with_name("perceptron_summary.json")

    summary = {
        "radius_bound": RADIUS_BOUND,
        "base_seed": BASE_SEED,
        "gamma_sweep": run_gamma_sweep(),
        "dimension_sweep": run_dimension_sweep(),
        "small_gamma_sweep": run_small_gamma_sweep(),
    }

    save_figures(summary, figures_dir)
    summary_path.write_text(json.dumps(summary, indent=2))

    gamma_summary = summary["gamma_sweep"]
    dimension_summary = summary["dimension_sweep"]

    print("Perceptron experiments completed.")
    print(f"Summary written to: {summary_path}")
    print(f"Figures written to: {figures_dir}")
    print(
        "log-log slope of mean mistakes versus 1/gamma^2: "
        f"{gamma_summary['loglog_slope']:.3f}"
    )
    print(
        "average ratio to the bound R^2/gamma^2: "
        f"{gamma_summary['mean_ratio_to_bound']:.3f}"
    )
    print(
        "relative range across tested dimensions: "
        f"{dimension_summary['relative_range']:.3f}"
    )


if __name__ == "__main__":
    main()
