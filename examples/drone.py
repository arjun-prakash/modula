import argparse
import json
from pathlib import Path
from typing import Dict, List

import gymnasium
from gymnasium import spaces
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

import PyFlyt.gym_envs  # noqa: F401

from modula.atom import Linear, matrix_sign
from modula.bond import ReLU

METHOD_CHOICES = ("descent", "dualize", "manifold_online")
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
EPS = 1e-6


def build_policy(input_dim: int, action_dim: int, width: int):
    model = Linear(action_dim * 2, width)
    model @= ReLU() @ Linear(width, width)
    model @= ReLU() @ Linear(width, input_dim)
    model.jit()
    return model


def prepare_action_transform(action_space: spaces.Box):
    if not isinstance(action_space, spaces.Box):
        raise ValueError("Drone example expects a continuous Box action space.")

    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    finite_mask = np.isfinite(low) & np.isfinite(high)

    scale = np.ones_like(low, dtype=np.float32)
    bias = np.zeros_like(low, dtype=np.float32)
    scale[finite_mask] = 0.5 * (high[finite_mask] - low[finite_mask])
    bias[finite_mask] = 0.5 * (high[finite_mask] + low[finite_mask])

    scale = np.where(scale == 0.0, 1.0, scale)
    log_scale_sum = float(np.sum(np.log(scale)))

    return {
        "scale": jnp.asarray(scale),
        "bias": jnp.asarray(bias),
        "log_scale_sum": jnp.asarray(log_scale_sum, dtype=jnp.float32),
    }


def policy_distribution(model, weights, inputs):
    logits = model(inputs, weights)
    mean, log_std = jnp.split(logits, 2, axis=-1)
    log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
    return mean, log_std


def gaussian_tanh_log_prob(mean, log_std, pre_tanh, log_scale_sum):
    std = jnp.exp(log_std)
    normal_log_prob = -0.5 * (((pre_tanh - mean) / std) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
    normal_log_prob = jnp.sum(normal_log_prob, axis=-1)
    squashed = jnp.tanh(pre_tanh)
    correction = jnp.sum(jnp.log(1.0 - squashed ** 2 + EPS), axis=-1)
    return normal_log_prob - correction - log_scale_sum


def compute_discounted_returns(rewards: List[float], gamma: float) -> jnp.ndarray:
    if not rewards:
        return jnp.zeros((0,), dtype=jnp.float32)

    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        returns.append(running)
    returns.reverse()

    returns_arr = jnp.asarray(returns, dtype=jnp.float32)
    mean = jnp.mean(returns_arr)
    std = jnp.std(returns_arr)
    denom = jnp.where(std > 1e-8, std, 1.0)
    return (returns_arr - mean) / denom


def sample_action(model, weights, observation, key, transform):
    mean, log_std = policy_distribution(model, weights, observation)
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, shape=mean.shape)
    pre_tanh = mean + noise * std
    squashed = jnp.tanh(pre_tanh)
    action = transform["bias"] + transform["scale"] * squashed
    return np.asarray(action, dtype=np.float32), pre_tanh


def compute_episode_gradients(model, weights, observations, pre_tanh_actions, returns, log_scale_sum):
    def loss_fn(w):
        mean, log_std = policy_distribution(model, w, observations)
        log_probs = gaussian_tanh_log_prob(mean, log_std, pre_tanh_actions, log_scale_sum)
        return -jnp.sum(log_probs * returns)

    loss_value, grads = jax.value_and_grad(loss_fn)(weights)
    return loss_value, grads


def train_single_run(
    model,
    base_key,
    method: str,
    learning_rate: float,
    episodes: int,
    max_episode_steps,
    gamma: float,
    target_norm: float,
    env_name: str,
    action_transform,
    dual_alpha: float,
    dual_beta: float,
):
    key_init, key_loop = jax.random.split(base_key)
    weights = model.initialize(key_init)
    dual_state = model.init_dual_state(weights) if method == "manifold_online" else None

    env = gymnasium.make(env_name)
    try:
        max_steps = max_episode_steps
        if max_steps is None or max_steps <= 0:
            max_steps = getattr(env.spec, "max_episode_steps", None)
        if max_steps is None:
            max_steps = 1000
        max_steps = int(max_steps)

        episodic_returns: List[float] = []
        loss_history: List[float] = []

        description = f"{method} lr={learning_rate:.3g}"
        key_loop, seed_key = jax.random.split(key_loop)
        base_env_seed = int(jax.random.randint(seed_key, shape=(), minval=0, maxval=1_000_000))

        for episode in trange(episodes, leave=False, desc=description):
            obs, _ = env.reset(seed=base_env_seed + episode)
            episode_observations: List[jnp.ndarray] = []
            episode_pre_tanh: List[jnp.ndarray] = []
            episode_rewards: List[float] = []

            for _ in range(max_steps):
                observation = jnp.asarray(obs, dtype=jnp.float32).reshape(-1)
                key_loop, action_key = jax.random.split(key_loop)
                action, pre_tanh = sample_action(model, weights, observation, action_key, action_transform)

                next_obs, reward, terminated, truncated, _ = env.step(action)
                episode_observations.append(observation)
                episode_pre_tanh.append(pre_tanh)
                episode_rewards.append(float(reward))

                obs = next_obs
                if terminated or truncated:
                    break

            episodic_returns.append(float(np.sum(episode_rewards)))
            if not episode_rewards:
                continue

            observations = jnp.stack(episode_observations, axis=0)
            pre_tanh_actions = jnp.stack(episode_pre_tanh, axis=0)
            returns = compute_discounted_returns(episode_rewards, gamma)

            loss_value, grad_weights = compute_episode_gradients(
                model,
                weights,
                observations,
                pre_tanh_actions,
                returns,
                action_transform["log_scale_sum"],
            )
            loss_history.append(float(loss_value))

            if method == "manifold_online":
                tangents, dual_state = model.online_dual_ascent(
                    dual_state,
                    weights,
                    grad_weights,
                    target_norm=target_norm,
                    alpha=dual_alpha,
                    beta=dual_beta,
                )
                weights = [w - learning_rate * t for w, t in zip(weights, tangents)]
                weights = [matrix_sign(weight_matrix) for weight_matrix in weights]
            elif method == "dualize":
                directions = model.dualize(grad_weights, target_norm=target_norm)
                weights = [w - learning_rate * direction for w, direction in zip(weights, directions)]
            elif method == "descent":
                weights = [w - learning_rate * grad for w, grad in zip(weights, grad_weights)]
            else:
                raise ValueError(f"Unknown training method: {method}")

        final_loss = loss_history[-1] if loss_history else float("nan")
        return {
            "episodic_returns": episodic_returns,
            "loss_history": loss_history,
            "final_loss": float(final_loss),
        }
    finally:
        env.close()


def plot_returns(best_runs: Dict[str, Dict[str, object]], plots_dir: Path) -> None:
    if not best_runs:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("tab10")

    for idx, (method, run) in enumerate(best_runs.items()):
        returns = np.asarray(run["episodic_returns"], dtype=np.float32)
        if returns.size == 0:
            continue
        episodes = np.arange(1, returns.size + 1)
        ax.plot(
            episodes,
            returns,
            label=f"{method} lr={run['learning_rate']:.3g}",
            color=cmap(idx % 10),
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return (undiscounted)")
    ax.set_title("Drone REINFORCE episodic returns")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(plots_dir / "drone_reinforce_returns.png", dpi=300)
    plt.close(fig)


def save_results(results, best_runs, args, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_steps_cfg = int(args.max_episode_steps) if args.max_episode_steps > 0 else None

    payload = {
        "config": {
            "env_name": args.env_name,
            "learning_rates": [float(lr) for lr in args.learning_rates],
            "episodes": int(args.episodes),
            "max_episode_steps": max_steps_cfg,
            "gamma": float(args.gamma),
            "seed": int(args.seed),
            "target_norm": float(args.target_norm),
            "hidden_width": int(args.hidden_width),
            "methods": list(args.methods),
            "dual_alpha": float(args.dual_alpha),
            "dual_beta": float(args.dual_beta),
        },
        "methods": {},
    }

    for method, runs in results.items():
        payload["methods"][method] = {
            "runs": [
                {
                    "learning_rate": float(entry["learning_rate"]),
                    "average_return": float(entry["average_return"]),
                    "final_return": float(entry["final_return"]),
                    "final_loss": float(entry["final_loss"]),
                }
                for entry in runs
            ]
        }
        best = best_runs.get(method)
        if best is not None:
            payload["methods"][method]["best"] = {
                "learning_rate": float(best["learning_rate"]),
                "average_return": float(best["average_return"]),
                "final_return": float(best["final_return"]),
                "final_loss": float(best["final_loss"]),
                "episodic_returns": [float(x) for x in best["episodic_returns"]],
                "loss_history": [float(x) for x in best["loss_history"]],
            }

    with output_path.open("w") as fp:
        json.dump(payload, fp, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="REINFORCE comparison on PyFlyt QuadX hover task."
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="PyFlyt/QuadX-Hover",
        help="Gymnasium environment id to train on.",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e-6],
        help="Learning rates to sweep per method.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of training episodes per learning rate.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=0,
        help="Maximum steps per episode (0 uses the environment default).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for returns.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Base PRNG seed."
    )
    parser.add_argument(
        "--target-norm",
        type=float,
        default=1.0,
        help="Target norm used by dualize/manifold updates.",
    )
    parser.add_argument(
        "--hidden-width",
        type=int,
        default=128,
        help="Hidden layer width of the policy network.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=list(METHOD_CHOICES),
        choices=METHOD_CHOICES,
        help="Optimization methods to compare.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("results/drone_reinforce_results.json"),
        help="Where to store aggregate metrics.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("plots"),
        help="Directory for plot outputs.",
    )
    parser.add_argument(
        "--dual-alpha",
        type=float,
        default=2e-5,
        help="Alpha parameter for manifold_online dual ascent.",
    )
    parser.add_argument(
        "--dual-beta",
        type=float,
        default=0.9,
        help="Beta parameter for manifold_online dual ascent.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    preview_env = gymnasium.make(args.env_name)
    try:
        obs_space = preview_env.observation_space
        if not isinstance(obs_space, spaces.Box):
            raise ValueError("Drone example expects a Box observation space.")
        action_space = preview_env.action_space
        if not isinstance(action_space, spaces.Box):
            raise ValueError("Drone example expects a continuous Box action space.")

        input_dim = int(np.prod(obs_space.shape))
        action_dim = int(np.prod(action_space.shape))
        action_transform = prepare_action_transform(action_space)
    finally:
        preview_env.close()

    model = build_policy(input_dim, action_dim, args.hidden_width)

    results: Dict[str, List[Dict[str, float]]] = {method: [] for method in args.methods}
    best_runs: Dict[str, Dict[str, object]] = {}

    base_key = jax.random.PRNGKey(args.seed)
    max_episode_steps = args.max_episode_steps if args.max_episode_steps > 0 else None

    for method_idx, method in enumerate(args.methods):
        method_key = jax.random.fold_in(base_key, method_idx)
        for lr_idx, learning_rate in enumerate(args.learning_rates):
            run_key = jax.random.fold_in(method_key, lr_idx)
            run_data = train_single_run(
                model,
                run_key,
                method,
                learning_rate,
                args.episodes,
                max_episode_steps,
                args.gamma,
                args.target_norm,
                args.env_name,
                action_transform,
                args.dual_alpha,
                args.dual_beta,
            )

            episodic_returns = run_data["episodic_returns"]
            average_return = float(np.mean(episodic_returns)) if episodic_returns else float("nan")
            final_return = float(episodic_returns[-1]) if episodic_returns else float("nan")

            entry = {
                "learning_rate": float(learning_rate),
                "average_return": average_return,
                "final_return": final_return,
                "final_loss": float(run_data["final_loss"]),
            }
            results[method].append(entry)

            best = best_runs.get(method)
            if best is None or final_return > best["final_return"]:
                best_runs[method] = {
                    "learning_rate": float(learning_rate),
                    "episodic_returns": episodic_returns,
                    "loss_history": run_data["loss_history"],
                    "average_return": average_return,
                    "final_return": final_return,
                    "final_loss": float(run_data["final_loss"]),
                }

            print(
                f"[{method}] lr={learning_rate:.3g}: avg return={average_return:.2f} "
                f"| final return={final_return:.2f} | loss={run_data['final_loss']:.4f}"
            )

    plot_returns(best_runs, args.plots_dir)
    save_results(results, best_runs, args, args.results_path)


if __name__ == "__main__":
    main()
