from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
SUMMARY_PATH = Path(__file__).with_name("fixedboost_summary.json")


def sign_matrix_from_labeled_sample(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    signed = y[:, None] * X
    return np.concatenate([signed, -signed], axis=1).astype(float)


def normalized_margin(A: np.ndarray, w: np.ndarray) -> float:
    l1 = float(np.sum(w))
    if l1 == 0.0:
        return 0.0
    return float(np.min(A @ w) / l1)


def fixedboost(A: np.ndarray, rounds: int, eta: float) -> tuple[list[dict], np.ndarray]:
    n, m = A.shape
    D = np.ones(n) / n
    scores = np.zeros(n)
    w = np.zeros(m)
    history: list[dict] = []

    for t in range(1, rounds + 1):
        correlations = A.T @ D
        j = int(np.argmax(correlations))
        w[j] += eta
        scores += eta * A[:, j]

        weights_sum = float(np.sum(w))
        history.append(
            {
                "round": t,
                "selected_index": j,
                "best_correlation": float(correlations[j]),
                "best_edge": float(correlations[j] / 2.0),
                "l1_norm": weights_sum,
                "min_raw_margin": float(np.min(scores)),
                "normalized_margin": float(np.min(scores) / weights_sum),
                "exp_loss": float(np.mean(np.exp(-scores))),
            }
        )

        D = np.exp(-scores)
        D /= D.sum()

    return history, w


def optimal_margin_lp(A: np.ndarray) -> dict:
    n, m = A.shape
    c = np.zeros(m + 1)
    c[-1] = -1.0

    A_ub = np.zeros((n, m + 1))
    A_ub[:, :m] = -A
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(n)

    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, None)] * m + [(-1.0, 1.0)]
    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Optimal-margin LP failed: {res.message}")

    u = res.x[:m]
    rho = float(res.x[-1])
    return {
        "optimal_margin": rho,
        "weights": u.tolist(),
    }


def first_round_at_margin(history: list[dict], target: float) -> int | None:
    for row in history:
        if row["normalized_margin"] >= target:
            return int(row["round"])
    return None


def summarize_instance(
    name: str,
    A: np.ndarray,
    history: list[dict],
    opt: dict,
    eta: float,
    meta: dict,
) -> dict:
    rho = float(opt["optimal_margin"])
    final = history[-1]
    n = A.shape[0]
    return {
        "name": name,
        "eta": eta,
        "n": int(A.shape[0]),
        "m": int(A.shape[1]),
        "optimal_margin": rho,
        "half_opt_round": first_round_at_margin(history, 0.5 * rho),
        "ninety_percent_round": first_round_at_margin(history, 0.9 * rho),
        "theorem_scale_log_over_gamma4": float(math.log(max(n, 2)) / max(rho**4, 1e-12)),
        "final_normalized_margin": final["normalized_margin"],
        "final_gap_to_optimum": float(rho - final["normalized_margin"]),
        "final_l1_norm": final["l1_norm"],
        "final_exp_loss": final["exp_loss"],
        "history": history,
        "lp_weights": opt["weights"],
        "meta": meta,
    }


def easy_majority_instance() -> tuple[np.ndarray, dict]:
    patterns = np.array(
        [[x1, x2, x3] for x1 in (-1, 1) for x2 in (-1, 1) for x3 in (-1, 1)],
        dtype=float,
    )
    y = np.where(np.sum(patterns, axis=1) >= 0.0, 1, -1)
    A = sign_matrix_from_labeled_sample(patterns, y)
    meta = {
        "description": "All 8 Boolean patterns in dimension 3 with labels sign(x1 + x2 + x3).",
        "dimension": 3,
        "sample_size": 8,
    }
    return A, meta


def hard_sign_matrix(m: int) -> np.ndarray:
    rows = ["0"] + [item for r in range(1, m + 1) for item in (f"{r}+", f"{r}-")]
    cols = []

    base = []
    for name in rows:
        if name == "0":
            base.append(1)
        else:
            base.append(1 if name[-1] == "+" else -1)
    cols.append(base)

    for rr in range(1, m + 1):
        col = []
        for name in rows:
            if name == "0":
                col.append(1)
            else:
                r = int(name[:-1])
                sign = name[-1]
                if r == rr:
                    col.append(-1 if sign == "+" else 1)
                else:
                    col.append(1 if sign == "+" else -1)
        cols.append(col)

    for rr in range(1, m + 1):
        col = []
        for name in rows:
            if name == "0":
                col.append(-1)
            else:
                r = int(name[:-1])
                sign = name[-1]
                if r < rr:
                    col.append(-1)
                elif r == rr:
                    col.append(1)
                else:
                    col.append(1 if sign == "+" else -1)
        cols.append(col)

    return np.array(cols, dtype=float).T


def hard_margin_instance(m: int = 3) -> tuple[np.ndarray, dict]:
    M = hard_sign_matrix(m)
    q = [1] + [2 ** (r - 1) for r in range(1, m + 1) for _ in (0, 1)]
    X = np.repeat(M, q, axis=0)
    y = np.ones(X.shape[0], dtype=int)
    A = sign_matrix_from_labeled_sample(X, y)
    meta = {
        "description": "Homework 6 hard sign-matrix construction repeated with weights q.",
        "m": m,
        "dimension": int(M.shape[1]),
        "sample_size": int(X.shape[0]),
        "row_weights": q,
    }
    return A, meta


def plot_histories(easy: dict, hard: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    for col, summary in enumerate((easy, hard)):
        rounds = [row["round"] for row in summary["history"]]
        margins = [row["normalized_margin"] for row in summary["history"]]
        l1_norms = [row["l1_norm"] for row in summary["history"]]
        edges = [row["best_edge"] for row in summary["history"]]
        rho = summary["optimal_margin"]

        axes[0, col].plot(rounds, margins, color="#1f77b4", linewidth=2, label="FixedBoost")
        axes[0, col].axhline(rho, color="#d95f0e", linestyle="--", linewidth=2, label="Optimal margin")
        axes[0, col].axhline(0.5 * rho, color="#2ca25f", linestyle=":", linewidth=1.8, label="Half optimum")
        axes[0, col].set_title(f"{summary['name']} margin")
        axes[0, col].set_xlabel("Round")
        axes[0, col].set_ylabel("Normalized margin")
        axes[0, col].legend(loc="lower right", fontsize=8)

        axes[1, col].plot(rounds, l1_norms, color="#1f77b4", linewidth=2, label=r"$\|w_t\|_1$")
        axes[1, col].plot(rounds, edges, color="#756bb1", linewidth=1.8, label="Best edge")
        axes[1, col].set_title(f"{summary['name']} growth")
        axes[1, col].set_xlabel("Round")
        axes[1, col].legend(loc="upper left", fontsize=8)

    fig.suptitle("FixedBoost on Large- and Small-Margin Coordinate Instances")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    eta = 0.1

    easy_A, easy_meta = easy_majority_instance()
    easy_opt = optimal_margin_lp(easy_A)
    easy_history, _ = fixedboost(easy_A, rounds=120, eta=eta)
    easy_summary = summarize_instance("easy_majority", easy_A, easy_history, easy_opt, eta, easy_meta)

    hard_A, hard_meta = hard_margin_instance(m=3)
    hard_opt = optimal_margin_lp(hard_A)
    hard_history, _ = fixedboost(hard_A, rounds=1500, eta=eta)
    hard_summary = summarize_instance("hard_sign_matrix", hard_A, hard_history, hard_opt, eta, hard_meta)

    plot_histories(easy_summary, hard_summary, FIG_DIR / "fixedboost_margin_comparison.png")

    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "easy_instance": easy_summary,
                "hard_instance": hard_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
