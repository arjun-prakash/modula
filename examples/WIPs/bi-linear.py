"""Extragradient vs Muon-EG on a regularized bilinear game.

This script compares a baseline Extragradient/Mirror-Prox optimizer against a Muon-style
Extragradient that replaces raw gradients with normalized "dualized" directions. We track both
last-iterate and averaged (ergodic) metrics so the same machinery works for strongly
convex–concave (μ>0) and purely bilinear (μ=0) games.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


Array = jnp.ndarray
Trajectory = Dict[str, np.ndarray]


@dataclass
class Problem:
    matrix: Array
    mu: float
    radius: float


def build_problem(dim: int, mu: float, radius: float, seed: int) -> Problem:
    key = jax.random.PRNGKey(seed)
    matrix = jax.random.normal(key, shape=(dim, dim), dtype=jnp.float32)
    return Problem(matrix=matrix, mu=mu, radius=radius)


def compute_loss(x: Array, y: Array, problem: Problem) -> Array:
    bilinear = jnp.dot(x, problem.matrix @ y)
    reg_term = 0.5 * problem.mu * (jnp.dot(x, x) - jnp.dot(y, y))
    return bilinear + reg_term


def compute_gradients(x: Array, y: Array, problem: Problem) -> Array:
    grad_x = problem.matrix @ y + problem.mu * x
    grad_y = problem.matrix.T @ x - problem.mu * y
    return grad_x, grad_y


def project_ball(vector: Array, radius: float) -> Array:
    norm = jnp.linalg.norm(vector)
    scale = jnp.where(norm > radius, radius / norm, 1.0)
    return vector * scale


def dualize_vector(grad: Array, budget: float) -> Array:
    norm = jnp.linalg.norm(grad)
    safe_budget = jnp.asarray(budget, dtype=grad.dtype)
    direction = jnp.where(norm > 0.0, grad * (safe_budget / norm), jnp.zeros_like(grad))
    return direction


def evaluate_state(x: Array, y: Array, problem: Problem):
    loss = compute_loss(x, y, problem)
    grad_x, grad_y = compute_gradients(x, y, problem)
    residual = jnp.sqrt(jnp.sum(jnp.square(grad_x)) + jnp.sum(jnp.square(grad_y)))
    return loss, residual, grad_x, grad_y


def record_state(
    records: List[Dict[str, np.ndarray]],
    step_index: int,
    x: Array,
    y: Array,
    sum_x: Array,
    sum_y: Array,
    problem: Problem,
):
    loss, residual, _, _ = evaluate_state(x, y, problem)
    sum_x = sum_x + x
    sum_y = sum_y + y
    denom = float(step_index + 1)
    avg_x = sum_x / denom
    avg_y = sum_y / denom
    avg_loss, avg_residual, _, _ = evaluate_state(avg_x, avg_y, problem)

    records.append(
        {
            "loss": float(loss),
            "avg_loss": float(avg_loss),
            "residual": float(residual),
            "avg_residual": float(avg_residual),
            "x": np.asarray(x),
            "y": np.asarray(y),
            "x_avg": np.asarray(avg_x),
            "y_avg": np.asarray(avg_y),
        }
    )
    return sum_x, sum_y


def stack_records(records: List[Dict[str, np.ndarray]]) -> Trajectory:
    scalar_keys = ["loss", "avg_loss", "residual", "avg_residual"]
    vector_keys = ["x", "y", "x_avg", "y_avg"]

    trajectory: Trajectory = {
        key: np.array([entry[key] for entry in records], dtype=np.float32)
        for key in scalar_keys
    }
    for key in vector_keys:
        trajectory[key] = np.stack([entry[key] for entry in records], axis=0)
    trajectory["steps"] = np.arange(len(records), dtype=np.int32)
    return trajectory


def extragradient(
    x: Array,
    y: Array,
    steps: int,
    step_size: float,
    problem: Problem,
) -> Trajectory:
    records: List[Dict[str, np.ndarray]] = []
    sum_x = jnp.zeros_like(x)
    sum_y = jnp.zeros_like(y)

    for step in range(steps):
        sum_x, sum_y = record_state(records, step, x, y, sum_x, sum_y, problem)

        grad_x, grad_y = compute_gradients(x, y, problem)
        x_half = project_ball(x - step_size * grad_x, problem.radius)
        y_half = project_ball(y + step_size * grad_y, problem.radius)

        grad_x_half, grad_y_half = compute_gradients(x_half, y_half, problem)
        x = project_ball(x - step_size * grad_x_half, problem.radius)
        y = project_ball(y + step_size * grad_y_half, problem.radius)

    record_state(records, steps, x, y, sum_x, sum_y, problem)
    return stack_records(records)


def muon_extragradient(
    x: Array,
    y: Array,
    steps: int,
    step_size: float,
    problem: Problem,
    budget_x: float,
    budget_y: float,
) -> Trajectory:
    records: List[Dict[str, np.ndarray]] = []
    sum_x = jnp.zeros_like(x)
    sum_y = jnp.zeros_like(y)

    for step in range(steps):
        sum_x, sum_y = record_state(records, step, x, y, sum_x, sum_y, problem)

        grad_x, grad_y = compute_gradients(x, y, problem)
        dir_x = dualize_vector(grad_x, budget_x)
        dir_y = dualize_vector(grad_y, budget_y)

        x_half = project_ball(x - step_size * dir_x, problem.radius)
        y_half = project_ball(y + step_size * dir_y, problem.radius)

        grad_x_half, grad_y_half = compute_gradients(x_half, y_half, problem)
        dir_x_half = dualize_vector(grad_x_half, budget_x)
        dir_y_half = dualize_vector(grad_y_half, budget_y)

        x = project_ball(x - step_size * dir_x_half, problem.radius)
        y = project_ball(y + step_size * dir_y_half, problem.radius)

    record_state(records, steps, x, y, sum_x, sum_y, problem)
    return stack_records(records)


def run_experiment(
    *,
    dim: int,
    steps: int,
    step_size: float,
    mu: float,
    radius: float,
    budget_x: float,
    budget_y: float,
    seed: int,
) -> Dict[str, Trajectory]:
    problem = build_problem(dim=dim, mu=mu, radius=radius, seed=seed)

    key = jax.random.PRNGKey(seed + 1)
    key_x, key_y = jax.random.split(key)

    init_x = project_ball(jax.random.normal(key_x, (dim,), dtype=jnp.float32), radius)
    init_y = project_ball(jax.random.normal(key_y, (dim,), dtype=jnp.float32), radius)

    baseline = extragradient(init_x, init_y, steps, step_size, problem)
    muon = muon_extragradient(init_x, init_y, steps, step_size, problem, budget_x, budget_y)

    return {
        "extragradient": baseline,
        "muon_extragradient": muon,
        "problem": {
            "matrix": np.asarray(problem.matrix),
            "mu": float(problem.mu),
            "radius": float(problem.radius),
            "budget_x": float(budget_x),
            "budget_y": float(budget_y),
        },
    }


def plot_results(results: Dict[str, Trajectory], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "bilinear_muon_eg.png"

    methods = ["extragradient", "muon_extragradient"]
    colors = plt.get_cmap("tab10")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax_residual, ax_avg_residual, ax_loss, ax_phase = axes.flat

    for idx, method in enumerate(methods):
        traj = results[method]
        color = colors(idx)
        steps = traj["steps"]

        ax_residual.plot(steps, traj["residual"], color=color, label=method)
        ax_avg_residual.plot(steps, traj["avg_residual"], color=color, label=method)
        ax_loss.plot(steps, np.abs(traj["loss"]), color=color, label=method)

        ax_phase.plot(traj["x"][:, 0], traj["y"][:, 0], color=color, label=method)
        ax_phase.scatter(traj["x"][0, 0], traj["y"][0, 0], color=color, marker="o", edgecolor="white", zorder=3)
        ax_phase.scatter(traj["x"][-1, 0], traj["y"][-1, 0], color=color, marker="x", s=60, zorder=3)

    for axis in (ax_residual, ax_avg_residual, ax_loss):
        axis.set_yscale("log")
        axis.set_xlabel("Step")
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.legend()

    ax_residual.set_title("Last-iterate residual ‖F(z_t)‖")
    ax_avg_residual.set_title("Averaged residual ‖F(ż_t)‖")
    ax_loss.set_title("Last-iterate objective gap |L(x_t, y_t)|")

    ax_phase.set_title("First-coordinate phase plot")
    ax_phase.set_xlabel("x[0]")
    ax_phase.set_ylabel("y[0]")
    ax_phase.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
    ax_phase.axvline(0.0, color="black", linewidth=0.5, alpha=0.4)
    ax_phase.scatter(0.0, 0.0, color="black", marker="+", s=80, label="nash", zorder=10)
    ax_phase.grid(True, linestyle="--", alpha=0.3)
    ax_phase.legend()

    fig.suptitle("Extragradient vs Muon-EG on a μ-strongly convex–concave game")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    return plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=5, help="Dimensionality of the player vectors")
    parser.add_argument("--steps", type=int, default=400, help="Number of extragradient steps")
    parser.add_argument("--lr", type=float, default=0.5, help="Extragradient step size (η)")
    parser.add_argument("--mu", type=float, default=0.1, help="Strong convex-concave regularization μ")
    parser.add_argument("--radius", type=float, default=2.0, help="ℓ₂ radius for the projection ball")
    parser.add_argument("--budget-x", type=float, default=1.0, help="Muon budget for the minimizer (x)")
    parser.add_argument("--budget-y", type=float, default=1.0, help="Muon budget for the maximizer (y)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for B and initialization")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./plots"),
        help="Directory to store the comparison figure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = run_experiment(
        dim=args.dim,
        steps=args.steps,
        step_size=args.lr,
        mu=args.mu,
        radius=args.radius,
        budget_x=args.budget_x,
        budget_y=args.budget_y,
        seed=args.seed,
    )

    plot_path = plot_results(results, args.output_dir)

    for method in ("extragradient", "muon_extragradient"):
        traj = results[method]
        final_residual = traj["residual"][-1]
        final_avg_residual = traj["avg_residual"][-1]
        print(
            f"{method:16s} | residual={final_residual:9.3e} | avg_residual={final_avg_residual:9.3e}"
        )

    print(f"Saved comparison figure to {plot_path}")


if __name__ == "__main__":
    main()
