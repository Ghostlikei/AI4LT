from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
SUMMARY_PATH = Path(__file__).with_name("experiment_summary.json")


def predict_sign(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.where(X @ w >= 0.0, 1, -1)


def agreement(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(predict_sign(w, X) == y))


def hinge_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.maximum(1.0 - y * (X @ w), 0.0)))


def support_size(w: np.ndarray, tol: float = 1e-8) -> int:
    return int(np.sum(np.abs(w) > tol))


def feasible_correct_subset(
    X: np.ndarray,
    y: np.ndarray,
    subset: tuple[int, ...],
    support: tuple[int, ...],
) -> tuple[bool, np.ndarray | None]:
    d = X.shape[1]
    if not support:
        if all(y[i] == 1 for i in subset):
            return True, np.zeros(d)
        return False, None

    support_list = list(support)
    A_ub = []
    b_ub = []
    for i in subset:
        row = X[i, support_list]
        if y[i] == 1:
            A_ub.append(-row)
            b_ub.append(0.0)
        else:
            # Strict negativity can be scaled to margin -1 on a finite sample.
            A_ub.append(row)
            b_ub.append(-1.0)

    res = linprog(
        c=np.zeros(len(support_list)),
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        bounds=[(None, None)] * len(support_list),
        method="highs",
    )
    if not res.success:
        return False, None

    w = np.zeros(d)
    w[support_list] = res.x
    return True, w


def exact_sparse_agreement_search(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
) -> tuple[np.ndarray, float, int]:
    m, d = X.shape
    best_correct = int(np.sum(y == 1))
    best_w = np.zeros(d)
    lp_calls = 0

    for support_size_value in range(1, k + 1):
        for support in itertools.combinations(range(d), support_size_value):
            for correct_count in range(m, best_correct, -1):
                found = False
                for subset in itertools.combinations(range(m), correct_count):
                    lp_calls += 1
                    feasible, candidate_w = feasible_correct_subset(X, y, subset, support)
                    if feasible:
                        best_correct = correct_count
                        best_w = candidate_w
                        found = True
                        break
                if found:
                    break

    return best_w, best_correct / m, lp_calls


def hinge_l1_lp(X: np.ndarray, y: np.ndarray, radius: float) -> tuple[np.ndarray, float]:
    m, d = X.shape
    nvars = 2 * d + m

    c = np.concatenate([np.zeros(2 * d), np.ones(m) / m])
    A_ub = []
    b_ub = []

    for i in range(m):
        row = np.zeros(nvars)
        row[:d] = -y[i] * X[i]
        row[d : 2 * d] = y[i] * X[i]
        row[2 * d + i] = -1.0
        A_ub.append(row)
        b_ub.append(-1.0)

    row = np.zeros(nvars)
    row[: 2 * d] = 1.0
    A_ub.append(row)
    b_ub.append(radius)

    res = linprog(
        c=c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=[(0.0, None)] * nvars,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Hinge LP failed: {res.message}")

    z = res.x
    w = z[:d] - z[d : 2 * d]
    return w, float(z[2 * d :].mean())


def make_sparse_dataset(
    seed: int,
    d: int,
    m: int,
    k_true: int,
    flip_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(m, d))
    w_star = np.zeros(d)
    support = rng.choice(d, size=min(k_true, d), replace=False)
    w_star[support] = rng.normal(loc=0.0, scale=1.3, size=len(support))
    y = predict_sign(w_star, X + 0.0)  # same sign convention as the homework

    noisy_scores = X @ w_star + 0.45 * rng.normal(size=m)
    y = np.where(noisy_scores >= 0.0, 1, -1)

    flip_count = max(1, int(round(m * flip_fraction)))
    flip_idx = rng.choice(m, size=flip_count, replace=False)
    y[flip_idx] *= -1
    return X, y


def make_validation_split(seed: int, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w_star = np.zeros(d)
    w_star[[1, 7]] = [1.5, -1.2]

    X_train = rng.normal(size=(10, d))
    y_train = np.where(X_train @ w_star + 0.5 * rng.normal(size=10) >= 0.0, 1, -1)
    flip_idx = rng.choice(10, size=2, replace=False)
    y_train[flip_idx] *= -1

    X_valid = rng.normal(size=(200, d))
    y_valid = np.where(X_valid @ w_star + 0.5 * rng.normal(size=200) >= 0.0, 1, -1)
    return X_train, y_train, X_valid, y_valid


def summarize_curve(xs: list[int], runtimes: list[list[float]], agreements: list[list[float]]) -> list[dict]:
    summary = []
    for x_value, runtime_values, agreement_values in zip(xs, runtimes, agreements):
        runtime_arr = np.array(runtime_values)
        agreement_arr = np.array(agreement_values)
        summary.append(
            {
                "x": x_value,
                "runtime_median": float(np.median(runtime_arr)),
                "runtime_min": float(np.min(runtime_arr)),
                "runtime_max": float(np.max(runtime_arr)),
                "agreement_median": float(np.median(agreement_arr)),
            }
        )
    return summary


def plot_runtime_curve(
    xs: list[int],
    exact_curve: list[dict],
    hinge_curve: list[dict],
    xlabel: str,
    title: str,
    output_path: Path,
) -> None:
    exact_median = [row["runtime_median"] for row in exact_curve]
    exact_low = [row["runtime_median"] - row["runtime_min"] for row in exact_curve]
    exact_high = [row["runtime_max"] - row["runtime_median"] for row in exact_curve]
    hinge_median = [row["runtime_median"] for row in hinge_curve]
    hinge_low = [row["runtime_median"] - row["runtime_min"] for row in hinge_curve]
    hinge_high = [row["runtime_max"] - row["runtime_median"] for row in hinge_curve]

    plt.figure(figsize=(7.2, 4.5))
    plt.errorbar(
        xs,
        exact_median,
        yerr=[exact_low, exact_high],
        marker="o",
        linewidth=2,
        capsize=4,
        label="Exact sparse 0/1 search",
    )
    plt.errorbar(
        xs,
        hinge_median,
        yerr=[hinge_low, hinge_high],
        marker="s",
        linewidth=2,
        capsize=4,
        label="Hinge LP with $\\ell_1$ constraint",
    )
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Runtime (seconds, log scale)")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_comparison_panels(summary: dict, output_path: Path) -> None:
    gap = summary["outlier_gap"]
    improper = summary["improper_demo"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

    axes[0].bar(
        ["Exact 0/1", "Hinge LP"],
        [gap["exact_train_agreement"], gap["hinge_train_agreement"]],
        color=["#2c7fb8", "#d95f0e"],
    )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Training agreement")
    axes[0].set_title("Outlier Dataset")
    axes[0].text(
        0.02,
        0.05,
        (
            f"exact hinge loss = {gap['exact_hinge_loss']:.3f}\n"
            f"hinge LP loss = {gap['hinge_hinge_loss']:.3f}"
        ),
        transform=axes[0].transAxes,
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    axes[1].bar(
        ["Exact 0/1", "Hinge LP"],
        [improper["exact_support_size"], improper["hinge_support_size"]],
        color=["#2c7fb8", "#d95f0e"],
    )
    axes[1].axhline(improper["k"], color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Number of nonzero coordinates")
    axes[1].set_title("Representative Noisy Dataset")
    axes[1].text(
        0.02,
        0.05,
        (
            f"exact valid. agreement = {improper['exact_validation_agreement']:.3f}\n"
            f"hinge valid. agreement = {improper['hinge_validation_agreement']:.3f}"
        ),
        transform=axes[1].transAxes,
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_dimension_sweep() -> tuple[list[dict], list[dict]]:
    d_values = [4, 6, 8, 10, 12]
    seeds = [0, 1, 2]
    k = 2
    m = 10
    radius = 2.5

    exact_runtimes = []
    exact_agreements = []
    hinge_runtimes = []
    hinge_agreements = []

    for d in d_values:
        exact_rt_row = []
        exact_ag_row = []
        hinge_rt_row = []
        hinge_ag_row = []
        for seed in seeds:
            X, y = make_sparse_dataset(seed, d, m, k_true=2)

            start = time.perf_counter()
            exact_w, exact_agreement, _ = exact_sparse_agreement_search(X, y, k)
            exact_rt_row.append(time.perf_counter() - start)
            exact_ag_row.append(exact_agreement)

            start = time.perf_counter()
            hinge_w, _ = hinge_l1_lp(X, y, radius)
            hinge_rt_row.append(time.perf_counter() - start)
            hinge_ag_row.append(agreement(hinge_w, X, y))

        exact_runtimes.append(exact_rt_row)
        exact_agreements.append(exact_ag_row)
        hinge_runtimes.append(hinge_rt_row)
        hinge_agreements.append(hinge_ag_row)

    exact_curve = summarize_curve(d_values, exact_runtimes, exact_agreements)
    hinge_curve = summarize_curve(d_values, hinge_runtimes, hinge_agreements)
    return exact_curve, hinge_curve


def run_sparsity_sweep() -> tuple[list[dict], list[dict]]:
    k_values = [1, 2, 3]
    seeds = [10, 11, 12]
    d = 12
    m = 10
    radius = 2.5

    exact_runtimes = []
    exact_agreements = []
    hinge_runtimes = []
    hinge_agreements = []

    for k in k_values:
        exact_rt_row = []
        exact_ag_row = []
        hinge_rt_row = []
        hinge_ag_row = []
        for seed in seeds:
            X, y = make_sparse_dataset(seed, d, m, k_true=k)

            start = time.perf_counter()
            exact_w, exact_agreement, _ = exact_sparse_agreement_search(X, y, k)
            exact_rt_row.append(time.perf_counter() - start)
            exact_ag_row.append(exact_agreement)

            start = time.perf_counter()
            hinge_w, _ = hinge_l1_lp(X, y, radius)
            hinge_rt_row.append(time.perf_counter() - start)
            hinge_ag_row.append(agreement(hinge_w, X, y))

        exact_runtimes.append(exact_rt_row)
        exact_agreements.append(exact_ag_row)
        hinge_runtimes.append(hinge_rt_row)
        hinge_agreements.append(hinge_ag_row)

    exact_curve = summarize_curve(k_values, exact_runtimes, exact_agreements)
    hinge_curve = summarize_curve(k_values, hinge_runtimes, hinge_agreements)
    return exact_curve, hinge_curve


def run_outlier_gap() -> dict:
    X = np.array([[1.0]] * 5 + [[-12.0]] * 2)
    y = np.ones(len(X), dtype=int)

    exact_w, exact_train_agreement, _ = exact_sparse_agreement_search(X, y, k=1)
    hinge_w, hinge_objective = hinge_l1_lp(X, y, radius=2.0)

    return {
        "exact_weights": exact_w.tolist(),
        "hinge_weights": hinge_w.tolist(),
        "exact_train_agreement": exact_train_agreement,
        "hinge_train_agreement": agreement(hinge_w, X, y),
        "exact_hinge_loss": hinge_loss(exact_w, X, y),
        "hinge_hinge_loss": hinge_objective,
        "exact_support_size": support_size(exact_w),
        "hinge_support_size": support_size(hinge_w),
    }


def run_improper_demo() -> dict:
    d = 10
    k = 2
    X_train, y_train, X_valid, y_valid = make_validation_split(seed=0, d=d)

    exact_w, exact_train_agreement, _ = exact_sparse_agreement_search(X_train, y_train, k)
    hinge_w, hinge_objective = hinge_l1_lp(X_train, y_train, radius=2.5)

    return {
        "d": d,
        "k": k,
        "exact_train_agreement": exact_train_agreement,
        "hinge_train_agreement": agreement(hinge_w, X_train, y_train),
        "exact_validation_agreement": agreement(exact_w, X_valid, y_valid),
        "hinge_validation_agreement": agreement(hinge_w, X_valid, y_valid),
        "exact_support_size": support_size(exact_w),
        "hinge_support_size": support_size(hinge_w),
        "exact_hinge_loss": hinge_loss(exact_w, X_train, y_train),
        "hinge_hinge_loss": hinge_objective,
        "exact_weights": exact_w.tolist(),
        "hinge_weights": hinge_w.tolist(),
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    exact_dim, hinge_dim = run_dimension_sweep()
    exact_k, hinge_k = run_sparsity_sweep()
    outlier_gap = run_outlier_gap()
    improper_demo = run_improper_demo()

    summary = {
        "runtime_by_dimension": {
            "x_values": [row["x"] for row in exact_dim],
            "exact_curve": exact_dim,
            "hinge_curve": hinge_dim,
            "support_candidates": [
                sum(math.comb(row["x"], j) for j in range(0, 2 + 1))
                for row in exact_dim
            ],
        },
        "runtime_by_sparsity": {
            "x_values": [row["x"] for row in exact_k],
            "exact_curve": exact_k,
            "hinge_curve": hinge_k,
            "support_candidates": [
                sum(math.comb(12, j) for j in range(0, row["x"] + 1))
                for row in exact_k
            ],
        },
        "outlier_gap": outlier_gap,
        "improper_demo": improper_demo,
    }

    plot_runtime_curve(
        xs=summary["runtime_by_dimension"]["x_values"],
        exact_curve=exact_dim,
        hinge_curve=hinge_dim,
        xlabel="Ambient dimension $d$",
        title="Runtime Scaling with Dimension ($k=2$, $m=10$)",
        output_path=FIG_DIR / "runtime_vs_dimension.png",
    )
    plot_runtime_curve(
        xs=summary["runtime_by_sparsity"]["x_values"],
        exact_curve=exact_k,
        hinge_curve=hinge_k,
        xlabel="Sparsity budget $k$",
        title="Runtime Scaling with Sparsity ($d=12$, $m=10$)",
        output_path=FIG_DIR / "runtime_vs_sparsity.png",
    )
    plot_comparison_panels(summary, FIG_DIR / "comparison_panels.png")

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
