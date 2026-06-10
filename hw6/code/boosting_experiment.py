from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
SUMMARY_PATH = Path(__file__).with_name("boosting_summary.json")


def predict_linear(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.where(X @ w >= 0.0, 1, -1)


def exponential_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.exp(-y * (X @ w))))


def normalized_margin(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    l1 = float(np.sum(np.abs(w)))
    if l1 == 0.0:
        return 0.0
    return float(np.min(y * (X @ w)) / l1)


def support_size(w: np.ndarray, tol: float = 1e-10) -> int:
    return int(np.sum(np.abs(w) > tol))


def adaboost_coordinate(
    X: np.ndarray,
    y: np.ndarray,
    rounds: int,
) -> tuple[list[dict], np.ndarray, float]:
    n, d = X.shape
    D = np.ones(n) / n
    F = np.zeros(n)
    w = np.zeros(d)
    history: list[dict] = []

    start = time.perf_counter()
    for t in range(rounds):
        correlations = (D * y) @ X
        j = int(np.argmax(np.abs(correlations)))
        sigma = 1.0 if correlations[j] >= 0.0 else -1.0
        h = sigma * X[:, j]

        edge = float(np.sum(D * y * h) / 2.0)
        eps = 0.5 - edge
        eps = min(max(eps, 1e-12), 1.0 - 1e-12)
        alpha = 0.5 * math.log((1.0 - eps) / eps)

        F += alpha * h
        w[j] += alpha * sigma
        preds = np.where(F >= 0.0, 1, -1)

        history.append(
            {
                "round": t + 1,
                "train_error": float(np.mean(preds != y)),
                "exp_loss": float(np.mean(np.exp(-y * F))),
                "edge": edge,
                "support": support_size(w),
                "norm_margin": normalized_margin(w, X, y),
                "alpha": alpha,
                "coordinate": j,
                "sign": int(sigma),
            }
        )

        D *= np.exp(-alpha * y * h)
        D /= D.sum()

    return history, w, time.perf_counter() - start


def hinge_l1_lp(X: np.ndarray, y: np.ndarray, radius: float) -> tuple[np.ndarray, dict]:
    n, d = X.shape
    nvars = 2 * d + n
    c = np.concatenate([np.zeros(2 * d), np.ones(n) / n])

    A_ub = []
    b_ub = []
    for i in range(n):
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

    start = time.perf_counter()
    res = linprog(
        c=c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=[(0.0, None)] * nvars,
        method="highs",
    )
    runtime = time.perf_counter() - start
    if not res.success:
        raise RuntimeError(f"Hinge LP failed: {res.message}")

    z = res.x
    w = z[:d] - z[d : 2 * d]
    metrics = {
        "train_error": float(np.mean(predict_linear(w, X) != y)),
        "exp_loss": exponential_loss(w, X, y),
        "support": support_size(w),
        "norm_margin": normalized_margin(w, X, y),
        "hinge_objective": float(z[2 * d :].mean()),
        "runtime": runtime,
        "l1_norm": float(np.sum(np.abs(w))),
        "radius": radius,
        "weights": w.tolist(),
    }
    return w, metrics


def easy_sparse_margin_dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    d = 20
    n = 400
    X = rng.choice([-1, 1], size=(n, d))
    y = np.where(X[:, 0] + X[:, 1] + X[:, 2] >= 0, 1, -1)
    info = {
        "name": "easy_sparse_margin",
        "d": d,
        "n": n,
        "s_true": 3,
        "description": "Uniform random sign vectors with labels sign(x1 + x2 + x3).",
    }
    return X.astype(float), y.astype(int), info


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


def hard_construction_dataset(m: int = 3) -> tuple[np.ndarray, np.ndarray, dict]:
    M = hard_sign_matrix(m)
    q = [1] + [2 ** (r - 1) for r in range(1, m + 1) for _ in (0, 1)]
    X = np.repeat(M, q, axis=0)
    y = np.ones(X.shape[0], dtype=int)
    s = 2 * m + 1

    carry = [2 ** (m - r) for r in range(1, m + 1)]
    flips = [2**m - 2 ** (m - r) for r in range(1, m + 1)]
    base = (2 - m) * 2**m - 1
    witness = np.array([base] + flips + carry, dtype=float)

    info = {
        "name": "hard_construction",
        "m": m,
        "s": s,
        "n": int(X.shape[0]),
        "weights_q": q,
        "column_weighted_sum": (np.array(q, dtype=float) @ M).tolist(),
        "witness_weights": witness.tolist(),
        "witness_linf": float(np.max(np.abs(witness))),
        "description": "Part A.3 sign-matrix construction duplicated according to the q-weights.",
    }
    return X, y, info


def first_zero_round(history: list[dict]) -> int | None:
    for row in history:
        if row["train_error"] == 0.0:
            return int(row["round"])
    return None


def summarize_run(
    X: np.ndarray,
    y: np.ndarray,
    history: list[dict],
    final_w: np.ndarray,
    runtime: float,
) -> dict:
    return {
        "zero_train_error_round": first_zero_round(history),
        "final_train_error": history[-1]["train_error"],
        "final_exp_loss": history[-1]["exp_loss"],
        "final_edge": history[-1]["edge"],
        "final_support": history[-1]["support"],
        "final_norm_margin": history[-1]["norm_margin"],
        "runtime": runtime,
        "weights": final_w.tolist(),
        "history": history,
        "final_l1_norm": float(np.sum(np.abs(final_w))),
        "final_prediction_error": float(np.mean(predict_linear(final_w, X) != y)),
    }


def plot_run(summary: dict, hinge: dict, output_path: Path, title: str) -> None:
    rounds = [row["round"] for row in summary["history"]]
    train_error = [row["train_error"] for row in summary["history"]]
    exp_loss = [row["exp_loss"] for row in summary["history"]]
    edge = [row["edge"] for row in summary["history"]]
    support = [row["support"] for row in summary["history"]]
    norm_margin = [row["norm_margin"] for row in summary["history"]]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    ax = axes.ravel()

    ax[0].plot(rounds, train_error, color="#1f77b4", linewidth=2)
    ax[0].axhline(hinge["train_error"], color="#d95f0e", linestyle="--", linewidth=1.8)
    ax[0].set_title("Training Error")
    ax[0].set_xlabel("Round")
    ax[0].set_ylim(-0.02, 1.02)

    ax[1].plot(rounds, exp_loss, color="#1f77b4", linewidth=2)
    ax[1].axhline(hinge["exp_loss"], color="#d95f0e", linestyle="--", linewidth=1.8)
    ax[1].set_title("Exponential Loss")
    ax[1].set_xlabel("Round")
    ax[1].set_yscale("log")

    ax[2].plot(rounds, edge, color="#1f77b4", linewidth=2)
    ax[2].set_title("Observed Edge")
    ax[2].set_xlabel("Round")

    ax[3].plot(rounds, support, color="#1f77b4", linewidth=2)
    ax[3].axhline(hinge["support"], color="#d95f0e", linestyle="--", linewidth=1.8)
    ax[3].set_title("Support Size")
    ax[3].set_xlabel("Round")

    ax[4].plot(rounds, norm_margin, color="#1f77b4", linewidth=2)
    ax[4].axhline(hinge["norm_margin"], color="#d95f0e", linestyle="--", linewidth=1.8)
    ax[4].set_title("Normalized Margin")
    ax[4].set_xlabel("Round")

    ax[5].axis("off")
    ax[5].text(
        0.02,
        0.98,
        (
            f"AdaBoost zero-error round: {summary['zero_train_error_round']}\n"
            f"AdaBoost runtime: {summary['runtime']:.4f}s\n"
            f"Hinge runtime: {hinge['runtime']:.4f}s\n"
            f"Hinge objective: {hinge['hinge_objective']:.4f}\n"
            "Blue: AdaBoost, Orange dashed: hinge baseline"
        ),
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X_easy, y_easy, easy_info = easy_sparse_margin_dataset()
    easy_history, easy_w, easy_runtime = adaboost_coordinate(X_easy, y_easy, rounds=50)
    _, easy_hinge = hinge_l1_lp(X_easy, y_easy, radius=3.0)
    easy_summary = summarize_run(X_easy, y_easy, easy_history, easy_w, easy_runtime)

    X_hard, y_hard, hard_info = hard_construction_dataset(m=3)
    hard_history, hard_w, hard_runtime = adaboost_coordinate(X_hard, y_hard, rounds=200)
    _, hard_hinge = hinge_l1_lp(X_hard, y_hard, radius=float(hard_info["s"]))
    hard_summary = summarize_run(X_hard, y_hard, hard_history, hard_w, hard_runtime)

    plot_run(
        easy_summary,
        easy_hinge,
        FIG_DIR / "easy_round_metrics.png",
        "Easy Sparse-Margin Task",
    )
    plot_run(
        hard_summary,
        hard_hinge,
        FIG_DIR / "hard_round_metrics.png",
        "Hard Sign-Matrix Construction",
    )

    summary = {
        "easy_dataset": easy_info,
        "easy_adaboost": easy_summary,
        "easy_hinge_l1": easy_hinge,
        "hard_dataset": hard_info,
        "hard_adaboost": hard_summary,
        "hard_hinge_l1": hard_hinge,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
