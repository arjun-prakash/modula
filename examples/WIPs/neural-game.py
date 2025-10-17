"""Neuralized matrix game with Extra-Gradient and manifold Muon layers.

This script instantiates a two-player zero-sum matrix game in which each player
observes a context vector and chooses a mixed strategy through a small neural
network. The hidden feature blocks live on the Stiefel manifold (enforced via
Muon dualization and matrix-sign retraction) while the strategy heads are kept
on the probability simplex using the newly added ``ProbDist`` layer.

We compare three optimisation regimes on a shared Extra-Gradient backbone:

``eg``
    vanilla Extra-Gradient (no manifold control on either layer group).
``stiefel``
    Muon updates + Stiefel retraction for feature blocks, plain heads.
``stiefel_simplex``
    Muon updates for features *and* simplex-constrained heads.

The training loop logs the expected payoff, policy entropies, spectral norms of
the feature maps, and the wall-clock time per optimisation step so the impact
of each manifold constraint can be visualised afterwards.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from modula.atom import Linear, ProbDist


Array = jnp.ndarray


# Fixed 3x3 zero-sum payoff matrix (rock-paper-scissors style).
PAYOFF_MATRIX = jnp.asarray(
    (
        (0.0, -1.0, 1.0),
        (1.0, 0.0, -1.0),
        (-1.0, 1.0, 0.0),
    ),
    dtype=jnp.float32,
)

LOG_EPS = 1e-8


def compute_loss_from_probs(probs_x: Array, probs_y: Array, epsilon: float) -> Tuple[Array, Array, Array, Array]:
    payoff = jnp.einsum("bi,ij,bj->b", probs_x, PAYOFF_MATRIX, probs_y)
    entropy_x = -jnp.sum(probs_x * jnp.log(jnp.clip(probs_x, LOG_EPS, 1.0)), axis=-1)
    entropy_y = -jnp.sum(probs_y * jnp.log(jnp.clip(probs_y, LOG_EPS, 1.0)), axis=-1)
    loss = payoff + epsilon * (entropy_x - entropy_y)
    return jnp.mean(loss), jnp.mean(payoff), jnp.mean(entropy_x), jnp.mean(entropy_y)


def best_response_y(probs_x: Array, epsilon: float) -> Array:
    payoffs = jnp.einsum("bi,ij->bj", probs_x, PAYOFF_MATRIX)
    if epsilon > 0.0:
        logits = -payoffs / epsilon
        return jax.nn.softmax(logits, axis=-1)
    indices = jnp.argmax(-payoffs, axis=-1)
    return jax.nn.one_hot(indices, probs_x.shape[-1])


def best_response_x(probs_y: Array, epsilon: float) -> Array:
    payoffs = jnp.einsum("ij,bj->bi", PAYOFF_MATRIX, probs_y)
    if epsilon > 0.0:
        logits = payoffs / epsilon
        return jax.nn.softmax(logits, axis=-1)
    indices = jnp.argmax(payoffs, axis=-1)
    return jax.nn.one_hot(indices, probs_y.shape[-1])


def kl_divergence(p_new: Array, p_old: Array) -> float:
    log_new = jnp.log(jnp.clip(p_new, LOG_EPS, 1.0))
    log_old = jnp.log(jnp.clip(p_old, LOG_EPS, 1.0))
    divergence = jnp.sum(p_new * (log_new - log_old), axis=-1)
    return float(jnp.mean(divergence))


def stiefel_violation(weight: Array) -> float:
    w = np.asarray(weight)
    gram = w.T @ w
    identity = np.eye(gram.shape[0], dtype=w.dtype)
    fro_norm = np.linalg.norm(gram - identity, ord="fro")
    denom = np.sqrt(w.shape[0] * w.shape[1])
    return float(fro_norm / max(denom, 1e-12))


def spectral_and_condition(weight: Array) -> Tuple[float, float]:
    singulars = np.linalg.svd(np.asarray(weight), compute_uv=False)
    sigma_max = float(np.max(singulars))
    sigma_min = float(np.min(singulars))
    cond = sigma_max / max(sigma_min, 1e-6)
    return sigma_max, cond


@dataclass
class GameConfig:
    """Hyperparameters and defaults for the neural game."""

    input_dim: int = 8
    hidden_dim: int = 16
    num_actions: int = 3
    epsilon: float = 5e-3
    batch_size: int = 128
    eval_batch_size: int = 512
    steps: int = 1000
    learning_rates: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8)
    log_every: int = 50
    stiefel_budget: float = 1.0
    simplex_budget: float = 0.1
    stiefel_alpha: float = 1e-2
    stiefel_beta: float = 0.9
    activation: str = "relu"


@dataclass
class PlayerManifold:
    """Holds manifold-aware atoms for a player's layers."""

    feature1: Linear
    feature2: Linear
    head: ProbDist


Params = Tuple[Array, Array, Array]


def build_activation(name: str):
    if name == "relu":
        return jax.nn.relu
    if name == "tanh":
        return jnp.tanh
    raise ValueError(f"Unsupported activation: {name}")


def initialise_player(manifold: PlayerManifold, key: Array, mode: str) -> Params:
    """Sample initial weights and (optionally) retract to the manifolds."""

    key_f1, key_f2, key_head = jax.random.split(key, 3)
    w1 = manifold.feature1.initialize(key_f1)[0]
    w2 = manifold.feature2.initialize(key_f2)[0]
    head = manifold.head.initialize(key_head)[0]

    if mode in {"stiefel", "stiefel_simplex"}:
        w1 = manifold.feature1.retract([w1])[0]
        w2 = manifold.feature2.retract([w2])[0]

    if mode in {"stiefel", "stiefel_simplex"}:
        head = manifold.head.retract([head])[0]

    return w1, w2, head


def forward_logits(params: Params, contexts: Array, activation_fn) -> Array:
    """Compute logits for a batch of contexts using a two-layer network."""

    w1, w2, head = params
    hidden1 = activation_fn(contexts @ w1.T)
    hidden2 = activation_fn(hidden1 @ w2.T)
    logits = hidden2 @ head.T
    return logits


def game_statistics(logits_x: Array, logits_y: Array, epsilon: float) -> Tuple[Array, Array]:
    """Return (loss, metrics) given player logits on a mini-batch."""

    probs_x = jax.nn.softmax(logits_x, axis=-1)
    probs_y = jax.nn.softmax(logits_y, axis=-1)

    payoff = jnp.einsum("bi,ij,bj->b", probs_x, PAYOFF_MATRIX, probs_y)
    mean_payoff = jnp.mean(payoff)

    entropy_x = -jnp.sum(probs_x * jnp.log(jnp.clip(probs_x, 1e-7, 1.0)), axis=-1)
    entropy_y = -jnp.sum(probs_y * jnp.log(jnp.clip(probs_y, 1e-7, 1.0)), axis=-1)

    mean_entropy_x = jnp.mean(entropy_x)
    mean_entropy_y = jnp.mean(entropy_y)

    loss = mean_payoff + epsilon * (mean_entropy_x - mean_entropy_y)
    aux = jnp.stack([mean_payoff, mean_entropy_x, mean_entropy_y])
    return loss, aux


def make_loss_fn(config: GameConfig):
    activation_fn = build_activation(config.activation)

    def loss_fn(params_x: Params, params_y: Params, contexts: Array):
        logits_x = forward_logits(params_x, contexts, activation_fn)
        logits_y = forward_logits(params_y, contexts, activation_fn)
        return game_statistics(logits_x, logits_y, config.epsilon)

    return jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))


def make_eval_fn(config: GameConfig):
    activation_fn = build_activation(config.activation)

    @jax.jit
    def eval_fn(params_x: Params, params_y: Params, contexts: Array):
        logits_x = forward_logits(params_x, contexts, activation_fn)
        logits_y = forward_logits(params_y, contexts, activation_fn)
        return game_statistics(logits_x, logits_y, config.epsilon)

    return eval_fn


def simplex_muon_direction(grad: Array, budget: float) -> Array:
    """Project gradients onto the simplex tangent with an ℓ∞ budget."""

    centered = grad - jnp.mean(grad, axis=-1, keepdims=True)
    max_abs = jnp.max(jnp.abs(centered), axis=-1, keepdims=True)
    scale = jnp.where(max_abs > 0, budget / max_abs, 0.0)
    return centered * scale


def apply_step(
    params: Params,
    grads: Params,
    manifold: PlayerManifold,
    step_size: float,
    mode: str,
    role: str,
    config: GameConfig,
    state: Optional[Tuple[list, list]],
) -> Tuple[Params, Optional[Tuple[list, list]]]:
    """Apply one manifold-aware update for a single player."""

    w1, w2, head = params
    g1, g2, g_head = grads

    step_scale = -step_size if role == "min" else step_size

    state1, state2 = (state if state is not None else (None, None))

    if mode in {"stiefel", "stiefel_simplex"}:
        tangent1_list, state1_next = manifold.feature1.online_dual_ascent(
            state1,
            [w1],
            [g1],
            target_norm=config.stiefel_budget,
            alpha=config.stiefel_alpha,
            beta=config.stiefel_beta,
        )
        tangent2_list, state2_next = manifold.feature2.online_dual_ascent(
            state2,
            [w2],
            [g2],
            target_norm=config.stiefel_budget,
            alpha=config.stiefel_alpha,
            beta=config.stiefel_beta,
        )
        dir1 = tangent1_list[0]
        dir2 = tangent2_list[0]
    else:
        dir1, dir2 = g1, g2
        state1_next, state2_next = state1, state2

    w1_next = w1 + step_scale * dir1
    w2_next = w2 + step_scale * dir2

    if mode in {"stiefel", "stiefel_simplex"}:
        w1_next = manifold.feature1.retract([w1_next])[0]
        w2_next = manifold.feature2.retract([w2_next])[0]

    if mode == "stiefel_simplex":
        dir_head = simplex_muon_direction(g_head, config.simplex_budget)
        head_next = head + step_scale * dir_head
        head_next = manifold.head.retract([head_next])[0]
    elif mode == "stiefel":
        head_next = head + step_scale * g_head
        head_next = manifold.head.retract([head_next])[0]
    else:
        head_next = head + step_scale * g_head

    if mode in {"stiefel", "stiefel_simplex"}:
        next_state = (state1_next, state2_next)
    else:
        next_state = state

    return (w1_next, w2_next, head_next), next_state


def spectral_norm(matrix: Array) -> float:
    values = np.linalg.svd(np.asarray(matrix), compute_uv=False)
    return float(values.max())


def train_single_run(
    mode: str,
    step_size: float,
    seed: int,
    config: GameConfig,
) -> Dict[str, np.ndarray]:
    """Run the neural game under a specific manifold regime."""

    manifolds = PlayerManifold(
        feature1=Linear(config.hidden_dim, config.input_dim),
        feature2=Linear(config.hidden_dim, config.hidden_dim),
        head=ProbDist(config.num_actions, config.hidden_dim),
    )

    loss_grad_fn = make_loss_fn(config)
    activation_fn = build_activation(config.activation)

    rng = jax.random.PRNGKey(seed)
    rng, key_init, key_eval = jax.random.split(rng, 3)
    key_x, key_y = jax.random.split(key_init)
    params_x = initialise_player(manifolds, key_x, mode)
    params_y = initialise_player(manifolds, key_y, mode)

    eval_contexts = jax.random.normal(key_eval, (config.eval_batch_size, config.input_dim))

    state_x: Optional[Tuple[list, list]] = None
    state_y: Optional[Tuple[list, list]] = None
    if mode in {"stiefel", "stiefel_simplex"}:
        state_x = (
            manifolds.feature1.init_dual_state([params_x[0]]),
            manifolds.feature2.init_dual_state([params_x[1]]),
        )
        state_y = (
            manifolds.feature1.init_dual_state([params_y[0]]),
            manifolds.feature2.init_dual_state([params_y[1]]),
        )

    log_steps = []
    log_loss = []
    log_payoff = []
    log_entropy_x = []
    log_entropy_y = []
    log_spec_x = []
    log_spec_y = []
    log_step_time = []
    log_gap_last = []
    log_gap_avg = []
    log_kl_x = []
    log_kl_y = []
    log_stiefel_violation_x = []
    log_stiefel_violation_y = []
    log_cond_x = []
    log_cond_y = []

    step_times_window: list[float] = []

    prev_probs_x: Optional[Array] = None
    prev_probs_y: Optional[Array] = None
    avg_probs_x: Optional[Array] = None
    avg_probs_y: Optional[Array] = None
    avg_count = 0

    progress_desc = f"{mode} η={step_size:.3g}"
    for step in trange(config.steps, leave=False, desc=progress_desc):
        rng, batch_key = jax.random.split(rng)
        contexts = jax.random.normal(batch_key, (config.batch_size, config.input_dim))

        start = time.perf_counter()
        (loss_val, aux), grads = loss_grad_fn(params_x, params_y, contexts)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), (loss_val, aux, grads))
        elapsed = time.perf_counter() - start

        step_times_window.append(elapsed)
        if len(step_times_window) > config.log_every:
            step_times_window.pop(0)

        grads_x, grads_y = grads

        params_x_half, _ = apply_step(
            params_x,
            grads_x,
            manifolds,
            step_size,
            mode,
            "min",
            config,
            state_x,
        )
        params_y_half, _ = apply_step(
            params_y,
            grads_y,
            manifolds,
            step_size,
            mode,
            "max",
            config,
            state_y,
        )

        (loss_half, aux_half), grads_half = loss_grad_fn(params_x_half, params_y_half, contexts)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), (loss_half, aux_half, grads_half))

        grads_x_half, grads_y_half = grads_half

        params_x, state_x = apply_step(
            params_x,
            grads_x_half,
            manifolds,
            step_size,
            mode,
            "min",
            config,
            state_x,
        )
        params_y, state_y = apply_step(
            params_y,
            grads_y_half,
            manifolds,
            step_size,
            mode,
            "max",
            config,
            state_y,
        )

        if step % config.log_every == 0 or step == config.steps - 1:
            logits_x_eval = forward_logits(params_x, eval_contexts, activation_fn)
            logits_y_eval = forward_logits(params_y, eval_contexts, activation_fn)
            probs_x_eval = jax.nn.softmax(logits_x_eval, axis=-1)
            probs_y_eval = jax.nn.softmax(logits_y_eval, axis=-1)

            loss_mean, payoff_mean, entropy_x_mean, entropy_y_mean = compute_loss_from_probs(
                probs_x_eval,
                probs_y_eval,
                config.epsilon,
            )

            if prev_probs_x is None:
                kl_x = 0.0
            else:
                kl_x = kl_divergence(probs_x_eval, prev_probs_x)
            if prev_probs_y is None:
                kl_y = 0.0
            else:
                kl_y = kl_divergence(probs_y_eval, prev_probs_y)

            y_br = best_response_y(probs_x_eval, config.epsilon)
            x_br = best_response_x(probs_y_eval, config.epsilon)
            upper_loss, _, _, _ = compute_loss_from_probs(probs_x_eval, y_br, config.epsilon)
            lower_loss, _, _, _ = compute_loss_from_probs(x_br, probs_y_eval, config.epsilon)
            gap_last = float(upper_loss - lower_loss)

            if avg_probs_x is None:
                avg_probs_x = probs_x_eval
                avg_probs_y = probs_y_eval
                avg_count = 1
                gap_avg = gap_last
            else:
                new_count = avg_count + 1
                avg_probs_x = (avg_probs_x * avg_count + probs_x_eval) / new_count
                avg_probs_y = (avg_probs_y * avg_count + probs_y_eval) / new_count
                avg_count = new_count
                y_br_avg = best_response_y(avg_probs_x, config.epsilon)
                x_br_avg = best_response_x(avg_probs_y, config.epsilon)
                upper_avg, _, _, _ = compute_loss_from_probs(
                    avg_probs_x, y_br_avg, config.epsilon
                )
                lower_avg, _, _, _ = compute_loss_from_probs(
                    x_br_avg, avg_probs_y, config.epsilon
                )
                gap_avg = float(upper_avg - lower_avg)

            prev_probs_x = probs_x_eval
            prev_probs_y = probs_y_eval

            violations_x = [stiefel_violation(params_x[0]), stiefel_violation(params_x[1])]
            violations_y = [stiefel_violation(params_y[0]), stiefel_violation(params_y[1])]
            spec_vals_x = []
            spec_vals_y = []
            cond_vals_x = []
            cond_vals_y = []
            for weight in params_x[:2]:
                sigma_max, cond_val = spectral_and_condition(weight)
                spec_vals_x.append(sigma_max)
                cond_vals_x.append(cond_val)
            for weight in params_y[:2]:
                sigma_max, cond_val = spectral_and_condition(weight)
                spec_vals_y.append(sigma_max)
                cond_vals_y.append(cond_val)

            spec_x = float(np.mean(spec_vals_x))
            spec_y = float(np.mean(spec_vals_y))
            cond_x = float(np.mean(cond_vals_x))
            cond_y = float(np.mean(cond_vals_y))
            violation_x = float(np.mean(violations_x))
            violation_y = float(np.mean(violations_y))

            log_steps.append(step)
            log_loss.append(float(loss_mean))
            log_payoff.append(float(payoff_mean))
            log_entropy_x.append(float(entropy_x_mean))
            log_entropy_y.append(float(entropy_y_mean))
            log_spec_x.append(spec_x)
            log_spec_y.append(spec_y)
            log_cond_x.append(cond_x)
            log_cond_y.append(cond_y)
            log_stiefel_violation_x.append(violation_x)
            log_stiefel_violation_y.append(violation_y)
            log_gap_last.append(gap_last)
            log_gap_avg.append(gap_avg)
            log_kl_x.append(kl_x)
            log_kl_y.append(kl_y)

            avg_step_time = float(np.mean(step_times_window)) if step_times_window else 0.0
            log_step_time.append(avg_step_time)

    return {
        "mode": mode,
        "step_size": step_size,
        "steps": np.asarray(log_steps, dtype=np.int32),
        "loss": np.asarray(log_loss, dtype=np.float32),
        "payoff": np.asarray(log_payoff, dtype=np.float32),
        "entropy_x": np.asarray(log_entropy_x, dtype=np.float32),
        "entropy_y": np.asarray(log_entropy_y, dtype=np.float32),
        "spectral_x": np.asarray(log_spec_x, dtype=np.float32),
        "spectral_y": np.asarray(log_spec_y, dtype=np.float32),
        "stiefel_violation_x": np.asarray(log_stiefel_violation_x, dtype=np.float32),
        "stiefel_violation_y": np.asarray(log_stiefel_violation_y, dtype=np.float32),
        "cond_x": np.asarray(log_cond_x, dtype=np.float32),
        "cond_y": np.asarray(log_cond_y, dtype=np.float32),
        "gap_last": np.asarray(log_gap_last, dtype=np.float32),
        "gap_avg": np.asarray(log_gap_avg, dtype=np.float32),
        "kl_x": np.asarray(log_kl_x, dtype=np.float32),
        "kl_y": np.asarray(log_kl_y, dtype=np.float32),
        "step_time": np.asarray(log_step_time, dtype=np.float32),
    }


def save_plots(all_results: Dict[str, np.ndarray], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = list(all_results.values())
    if not runs:
        return

    def label_for(run: Dict[str, np.ndarray]) -> str:
        return f"{run['mode']} η={run['step_size']}"

    # Saddle gap plots (last iterate vs averaged)
    fig_gap, axes_gap = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    ax_gap_last, ax_gap_avg = axes_gap
    for run in runs:
        label = label_for(run)
        ax_gap_last.plot(run["steps"], run["gap_last"], label=label)
        ax_gap_avg.plot(run["steps"], run["gap_avg"], label=label)
    ax_gap_last.set_title("Saddle gap (last iterate)")
    ax_gap_avg.set_title("Saddle gap (averaged)")
    for ax in axes_gap:
        ax.set_xlabel("training step")
        ax.set_ylabel("gap")
        ax.grid(True, linestyle="--", alpha=0.3)
    handles, labels = ax_gap_last.get_legend_handles_labels()
    if handles:
        fig_gap.legend(handles, labels, loc="upper center", ncol=3)
    fig_gap.tight_layout(rect=(0, 0, 1, 0.9))
    fig_gap.savefig(output_dir / "gap_metrics.png", dpi=300)
    plt.close(fig_gap)

    # Hidden-layer diagnostics (Stiefel violation, spectral norm, condition number)
    fig_hidden, axes_hidden = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    titles = [
        "Stiefel violation",
        "Spectral norm",
        "Condition number",
    ]
    metrics = [
        ("stiefel_violation_x", "stiefel_violation_y"),
        ("spectral_x", "spectral_y"),
        ("cond_x", "cond_y"),
    ]
    for axis, title, (key_x, key_y) in zip(axes_hidden, titles, metrics):
        for run in runs:
            steps = run["steps"]
            label = label_for(run)
            axis.plot(steps, run[key_x], label=f"{label} (X)")
            axis.plot(steps, run[key_y], linestyle="--", label=f"{label} (Y)")
        axis.set_title(title)
        axis.set_xlabel("training step")
        axis.grid(True, linestyle="--", alpha=0.3)
    handles, labels = axes_hidden[0].get_legend_handles_labels()
    if handles:
        fig_hidden.legend(handles, labels, loc="upper center", ncol=3)
    fig_hidden.tight_layout(rect=(0, 0, 1, 0.9))
    fig_hidden.savefig(output_dir / "hidden_diagnostics.png", dpi=300)
    plt.close(fig_hidden)

    # Head diagnostics: KL and entropy trajectories
    fig_heads, axes_heads = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    (ax_kl_x, ax_kl_y), (ax_ent_x, ax_ent_y) = axes_heads
    for run in runs:
        label = label_for(run)
        steps = run["steps"]
        ax_kl_x.plot(steps, run["kl_x"], label=label)
        ax_kl_y.plot(steps, run["kl_y"], label=label)
        ax_ent_x.plot(steps, run["entropy_x"], label=label)
        ax_ent_y.plot(steps, run["entropy_y"], label=label)

    ax_kl_x.set_title("KL(x_{t+1} || x_t)")
    ax_kl_y.set_title("KL(y_{t+1} || y_t)")
    ax_ent_x.set_title("Entropy H(x)")
    ax_ent_y.set_title("Entropy H(y)")
    for ax in axes_heads.flat:
        ax.set_xlabel("training step")
        ax.grid(True, linestyle="--", alpha=0.3)
    handles, labels = ax_kl_x.get_legend_handles_labels()
    if handles:
        fig_heads.legend(handles, labels, loc="upper center", ncol=3)
    fig_heads.tight_layout(rect=(0, 0, 1, 0.9))
    fig_heads.savefig(output_dir / "head_diagnostics.png", dpi=300)
    plt.close(fig_heads)

    # Step-time comparison
    fig_time, ax_time = plt.subplots(figsize=(6, 4))
    for run in runs:
        ax_time.plot(run["steps"], run["step_time"], label=label_for(run))
    ax_time.set_title("Step time (s)")
    ax_time.set_xlabel("training step")
    ax_time.set_ylabel("seconds")
    ax_time.grid(True, linestyle="--", alpha=0.3)
    handles, labels = ax_time.get_legend_handles_labels()
    if handles:
        fig_time.legend(handles, labels, loc="upper center", ncol=3)
    fig_time.tight_layout(rect=(0, 0, 1, 0.9))
    fig_time.savefig(output_dir / "step_time.png", dpi=300)
    plt.close(fig_time)

    # Stable learning-rate range summary (final gap vs learning rate)
    lr_summary: Dict[str, list[Tuple[float, float]]] = {}
    for run in runs:
        mode = run["mode"]
        lr_summary.setdefault(mode, []).append((run["step_size"], float(run["gap_last"][-1])))

    fig_lr, ax_lr = plt.subplots(figsize=(6, 4))
    for mode, entries in lr_summary.items():
        entries.sort(key=lambda item: item[0])
        rates, final_gaps = zip(*entries)
        ax_lr.plot(rates, final_gaps, marker="o", label=mode)
    ax_lr.set_xscale("log")
    ax_lr.set_xlabel("learning rate")
    ax_lr.set_ylabel("final gap")
    ax_lr.set_title("Stable learning-rate range")
    ax_lr.grid(True, linestyle="--", alpha=0.3)
    handles, labels = ax_lr.get_legend_handles_labels()
    if handles:
        fig_lr.legend(handles, labels, loc="upper center")
    fig_lr.tight_layout(rect=(0, 0, 1, 0.92))
    fig_lr.savefig(output_dir / "lr_stability.png", dpi=300)
    plt.close(fig_lr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["eg", "stiefel", "stiefel_simplex"],
        choices=["eg", "stiefel", "stiefel_simplex"],
        help="Optimisation regimes to evaluate.",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=None,
        help="Learning rate grid to sweep. Defaults to the Toy B values.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Training steps per run.")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed base.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/results/neural-game"),
        help="Where to store plots and logs.",
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "tanh"],
        default="relu",
        help="Hidden activation function.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Logging cadence in optimisation steps.",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Optional path to dump raw metrics as JSON.")

    args = parser.parse_args()

    config = GameConfig(activation=args.activation)
    if args.learning_rates is not None:
        config = GameConfig(
            **{**config.__dict__, "learning_rates": tuple(args.learning_rates)},
        )
    if args.steps is not None:
        config.steps = args.steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.log_every is not None:
        config.log_every = args.log_every

    all_results: Dict[str, np.ndarray] = {}

    run_idx = 0
    for mode in args.modes:
        for lr in config.learning_rates:
            print(f"Running mode={mode} lr={lr:.3f}")
            result = train_single_run(mode, lr, seed=args.seed + run_idx, config=config)
            key = f"{mode}_lr{lr:g}"
            all_results[key] = result
            run_idx += 1

    save_plots(all_results, args.output_dir)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as fp:
            json.dump({key: {k: v.tolist() for k, v in value.items()} for key, value in all_results.items()}, fp, indent=2)


if __name__ == "__main__":
    main()
