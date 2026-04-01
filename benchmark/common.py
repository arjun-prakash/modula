import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import jax
import jax.numpy as jnp
import jax.tree_util
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import tqdm

from examples.data.cifar10 import load_cifar10
from examples.data.cifar100 import load_cifar100
from modula.abstract import CompositeModule, TupleModule
from modula.atom import Conv2D, Linear
from modula.bond import Flatten, MaxPool2D, ReLU

METHOD_CHOICES = ("adam", "manifold", "manifold_online", "manifold_admm")
FEATURE_DIM = 4 * 4 * 128


@dataclass
class DatasetBundle:
    name: str
    num_classes: int
    train_inputs: jnp.ndarray
    train_targets: jnp.ndarray
    train_labels: jnp.ndarray
    test_inputs: jnp.ndarray
    test_targets: jnp.ndarray
    test_labels: jnp.ndarray


def one_hot(labels, num_classes, dtype=jnp.float32):
    return jnp.array(labels[:, None] == jnp.arange(num_classes), dtype)


def canonicalize_methods(methods: Sequence[str]) -> List[str]:
    canonical: List[str] = []
    for method in methods:
        normalized = method.lower()
        if normalized not in METHOD_CHOICES:
            raise ValueError(f"Unknown method: {method}")
        if normalized not in canonical:
            canonical.append(normalized)
    return canonical


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    dataset_name: str,
    default_learning_rates: Sequence[float],
    default_steps: int,
    default_batch_size: int,
    default_eval_every: int,
    default_results_path: Path,
    default_plots_dir: Path,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=list(default_learning_rates),
        help="Learning rates to sweep",
    )
    parser.add_argument("--steps", type=int, default=default_steps, help="Training steps per learning rate")
    parser.add_argument("--batch-size", type=int, default=default_batch_size, help="Mini-batch size")
    parser.add_argument("--eval-every", type=int, default=default_eval_every, help="Metric logging interval")
    parser.add_argument(
        "--eval-train-samples",
        type=int,
        default=1000,
        help="Train samples used for periodic eval (0 for the full train set)",
    )
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    parser.add_argument("--target-norm", type=float, default=1.0, help="Target norm for manifold-family methods")
    parser.add_argument("--dual-alpha", type=float, default=2e-5, help="Alpha for manifold_online")
    parser.add_argument("--dual-beta", type=float, default=0.9, help="Beta for manifold_online")
    parser.add_argument("--admm-steps", type=int, default=10, help="ADMM inner steps for manifold_admm")
    parser.add_argument("--admm-rho", type=float, default=4.0, help="ADMM penalty for manifold_admm")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=list(METHOD_CHOICES),
        choices=METHOD_CHOICES,
        help="Training methods to evaluate",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=default_results_path,
        help="Path to save benchmark summary metrics",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=default_plots_dir,
        help="Directory for plot outputs",
    )
    parser.add_argument("--synthetic-data", action="store_true", help="Use synthetic CIFAR-shaped data")
    parser.add_argument("--smoke-test", action="store_true", help="Force a tiny pass-through benchmark run")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=f"{dataset_name}-benchmark",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity (username or team)",
    )
    return parser


def apply_smoke_test_overrides(args) -> None:
    if not args.smoke_test:
        return
    args.learning_rates = [float(args.learning_rates[0] if args.learning_rates else 1e-2)]
    args.steps = min(int(args.steps), 2)
    args.batch_size = min(int(args.batch_size), 8)
    args.eval_every = 1
    args.eval_train_samples = 16 if int(args.eval_train_samples) == 0 else min(int(args.eval_train_samples), 16)


def _synthetic_split(num_examples: int, num_classes: int, rng: np.random.Generator):
    labels = np.arange(num_examples, dtype=np.int32) % num_classes
    rows = np.linspace(0.0, 1.0, 32, dtype=np.float32)[None, :, None, None]
    cols = np.linspace(0.0, 1.0, 32, dtype=np.float32)[None, None, :, None]
    base = (labels.astype(np.float32) / max(num_classes - 1, 1))[:, None, None, None]
    ch1 = ((labels % 10).astype(np.float32) / 9.0)[:, None, None, None]
    ch2 = ((labels // 10).astype(np.float32) / max((num_classes - 1) // 10, 1))[:, None, None, None]

    images = np.concatenate(
        [
            np.broadcast_to(base + 0.25 * rows, (num_examples, 32, 32, 1)),
            np.broadcast_to(ch1 + 0.20 * cols, (num_examples, 32, 32, 1)),
            np.broadcast_to(ch2 + 0.15 * (rows + cols), (num_examples, 32, 32, 1)),
        ],
        axis=-1,
    )
    noise = rng.normal(loc=0.0, scale=0.03, size=images.shape).astype(np.float32)
    images = np.clip(images + noise, 0.0, 1.0).astype(np.float32)
    return images, labels


def _make_synthetic_dataset(name: str, num_classes: int, *, smoke_test: bool, seed: int) -> DatasetBundle:
    train_size = 64 if smoke_test else 256
    test_size = 32 if smoke_test else 128
    rng = np.random.default_rng(seed)
    train_images, train_labels = _synthetic_split(train_size, num_classes, rng)
    test_images, test_labels = _synthetic_split(test_size, num_classes, rng)
    return DatasetBundle(
        name=name,
        num_classes=num_classes,
        train_inputs=jnp.asarray(train_images, dtype=jnp.float32),
        train_targets=one_hot(jnp.asarray(train_labels, dtype=jnp.int32), num_classes),
        train_labels=jnp.asarray(train_labels, dtype=jnp.int32),
        test_inputs=jnp.asarray(test_images, dtype=jnp.float32),
        test_targets=one_hot(jnp.asarray(test_labels, dtype=jnp.int32), num_classes),
        test_labels=jnp.asarray(test_labels, dtype=jnp.int32),
    )


def prepare_dataset(dataset_name: str, *, synthetic_data: bool = False, smoke_test: bool = False, seed: int = 0):
    if dataset_name == "cifar10":
        loader = load_cifar10
        num_classes = 10
    elif dataset_name == "cifar100":
        loader = load_cifar100
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if synthetic_data:
        return _make_synthetic_dataset(dataset_name, num_classes, smoke_test=smoke_test, seed=seed)

    train_images, train_labels, test_images, test_labels = loader(normalize=True)
    train_labels = jnp.asarray(train_labels, dtype=jnp.int32)
    test_labels = jnp.asarray(test_labels, dtype=jnp.int32)

    return DatasetBundle(
        name=dataset_name,
        num_classes=num_classes,
        train_inputs=jnp.asarray(train_images, dtype=jnp.float32),
        train_targets=one_hot(train_labels, num_classes),
        train_labels=train_labels,
        test_inputs=jnp.asarray(test_images, dtype=jnp.float32),
        test_targets=one_hot(test_labels, num_classes),
        test_labels=test_labels,
    )


def get_batch(key, X, y, batch_size):
    if X.shape[0] == 0:
        raise ValueError("Cannot sample from an empty dataset")
    replace = X.shape[0] < batch_size
    idx = jax.random.choice(key, X.shape[0], shape=(batch_size,), replace=replace)
    return X[idx], y[idx]


def build_cifar_trunk():
    trunk = Flatten()
    trunk @= MaxPool2D(pool_size=2)
    trunk @= ReLU() @ Conv2D(64, 128, kernel_size=3)
    trunk @= MaxPool2D(pool_size=2)
    trunk @= ReLU() @ Conv2D(32, 64, kernel_size=3)
    trunk @= MaxPool2D(pool_size=2)
    trunk @= ReLU() @ Conv2D(3, 32, kernel_size=3)
    trunk.jit()
    trunk.admm_dual_ascent = jax.jit(trunk.admm_dual_ascent, static_argnames=("steps", "rho"))
    return trunk


def build_classifier_head(num_classes: int):
    head = Linear(num_classes, FEATURE_DIM)
    head.jit()
    return head


def build_cifar_models(num_classes: int):
    return build_cifar_trunk(), build_classifier_head(num_classes)


def make_predict_fn(trunk, head):
    return jax.jit(lambda trunk_w, head_w, inputs: head(trunk(inputs, trunk_w), head_w))


def tree_l2_norm(tree) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return 0.0
    squared_norm = sum(jnp.sum(jnp.square(jnp.asarray(leaf, dtype=jnp.float32))) for leaf in leaves)
    return float(jnp.sqrt(squared_norm))


def _iter_weighted_atoms(module) -> List[Any]:
    atoms = int(getattr(module, "atoms", 0) or 0)
    if atoms == 0:
        return []

    children = getattr(module, "children", ())
    if not children:
        return [module]

    ordered_atoms: List[Any] = []
    for child in children:
        ordered_atoms.extend(_iter_weighted_atoms(child))
    return ordered_atoms


def _iter_atom_target_norms(module, target_norm: float) -> List[float]:
    atoms = int(getattr(module, "atoms", 0) or 0)
    if atoms == 0:
        return []

    if isinstance(module, CompositeModule):
        if module.mass <= 0:
            return [0.0] * atoms
        m0, m1 = module.children
        return (
            _iter_atom_target_norms(m0, target_norm * m0.mass / module.mass / m1.sensitivity)
            + _iter_atom_target_norms(m1, target_norm * m1.mass / module.mass)
        )

    if isinstance(module, TupleModule):
        if module.mass <= 0:
            return [0.0] * atoms
        norms: List[float] = []
        for child in module.children:
            norms.extend(_iter_atom_target_norms(child, target_norm * child.mass / module.mass))
        return norms

    children = getattr(module, "children", ())
    if not children:
        return [float(target_norm)] * atoms

    norms: List[float] = []
    for child in children:
        norms.extend(_iter_atom_target_norms(child, target_norm))
    return norms


def _manifold_update_scale(atom) -> float:
    if isinstance(atom, Conv2D):
        return float((atom.k ** 2) * np.sqrt(atom.d_out / atom.d_in))
    if isinstance(atom, Linear):
        return float(np.sqrt(atom.fanout / atom.fanin))
    raise ValueError(f"Unsupported manifold benchmark atom type: {type(atom).__name__}")


def _scale_manifold_tangents(module, tangents, *, target_norm: float):
    atoms = _iter_weighted_atoms(module)
    atom_target_norms = _iter_atom_target_norms(module, target_norm)

    if len(atoms) != len(tangents) or len(atoms) != len(atom_target_norms):
        raise ValueError(
            f"Mismatch between atom metadata ({len(atoms)}), tangents ({len(tangents)}), and target norms ({len(atom_target_norms)})"
        )

    scaled_tangents = []
    for atom, tangent, atom_target_norm in zip(atoms, tangents, atom_target_norms):
        scale = _manifold_update_scale(atom) * float(atom_target_norm)
        scaled_tangents.append(jnp.asarray(scale, dtype=tangent.dtype) * tangent)
    return scaled_tangents


def _weight_to_stiefel_matrix(atom, weight):
    array = jnp.asarray(weight, dtype=jnp.float32)
    if isinstance(atom, Conv2D):
        return atom._flatten_kernel(array)
    if isinstance(atom, Linear):
        return array
    return None


def _stiefel_target_scale(atom):
    if isinstance(atom, (Conv2D, Linear)):
        return jnp.asarray(1.0, dtype=jnp.float32)
    return None


def stiefel_deviation(atom, weight) -> float | None:
    matrix = _weight_to_stiefel_matrix(atom, weight)
    scale = _stiefel_target_scale(atom)
    if matrix is None or scale is None:
        return None

    rows, cols = matrix.shape
    if rows >= cols:
        gram = matrix.T @ matrix
        eye = jnp.eye(cols, dtype=matrix.dtype)
    else:
        gram = matrix @ matrix.T
        eye = jnp.eye(rows, dtype=matrix.dtype)

    residual = gram - (scale**2) * eye
    denom = jnp.sqrt(jnp.asarray(residual.size, dtype=matrix.dtype))
    denom = jnp.maximum(denom, jnp.asarray(1e-12, dtype=matrix.dtype))
    return float(jnp.linalg.norm(residual) / denom)


def compute_trunk_stiefel_metrics(trunk, trunk_weights) -> Dict[str, float]:
    atoms = _iter_weighted_atoms(trunk)
    deviations: List[float] = []
    metrics: Dict[str, float] = {}

    for layer_idx, (atom, weight) in enumerate(zip(atoms, trunk_weights)):
        deviation = stiefel_deviation(atom, weight)
        if deviation is None:
            continue
        deviations.append(deviation)
        metrics[f"trunk_stiefel_deviation_layer_{layer_idx}"] = float(deviation)

    if deviations:
        metrics["trunk_stiefel_deviation_mean"] = float(np.mean(deviations))
        metrics["trunk_stiefel_deviation_max"] = float(np.max(deviations))
    else:
        metrics["trunk_stiefel_deviation_mean"] = 0.0
        metrics["trunk_stiefel_deviation_max"] = 0.0

    return metrics


def compute_accuracy(predict_fn, trunk_weights, head_weights, inputs, labels, *, batch_size: int = 1024) -> float:
    total = inputs.shape[0]
    correct = 0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        logits = predict_fn(trunk_weights, head_weights, inputs[start:end])
        predictions = jnp.argmax(logits, axis=1)
        correct += int(jnp.sum(predictions == labels[start:end]))

    return 100.0 * correct / total


def _loss_fn(predict_fn, trunk_weights, head_weights, inputs, targets):
    logits = predict_fn(trunk_weights, head_weights, inputs)
    return jnp.mean((logits - targets) ** 2)


def train_single_run(
    trunk,
    head,
    dataset: DatasetBundle,
    *,
    batch_size: int,
    steps: int,
    learning_rate: float,
    eval_every: int,
    eval_train_samples: int,
    seed: int,
    method: str,
    target_norm: float,
    dual_alpha: float,
    dual_beta: float,
    admm_steps: int,
    admm_rho: float,
    logger=None,
    show_progress: bool = True,
):
    key = jax.random.PRNGKey(seed)
    key, trunk_key, head_key = jax.random.split(key, 3)

    trunk_weights = trunk.initialize(trunk_key)
    head_weights = head.initialize(head_key)
    predict_fn = make_predict_fn(trunk, head)
    loss_and_grad = jax.jit(jax.value_and_grad(lambda tw, hw, x, y: _loss_fn(predict_fn, tw, hw, x, y), argnums=(0, 1)))

    head_optimizer = optax.adam(learning_rate)
    head_opt_state = head_optimizer.init(head_weights)

    trunk_optimizer = None
    trunk_opt_state = None
    if method == "adam":
        trunk_optimizer = optax.adam(learning_rate)
        trunk_opt_state = trunk_optimizer.init(trunk_weights)

    dual_state = trunk.init_dual_state(trunk_weights) if method == "manifold_online" else None
    progress = tqdm(range(steps), desc=f"{method} lr={learning_rate:.3g}", leave=False, disable=not show_progress)
    start_time = time.perf_counter()
    last_loss = 0.0

    for step in progress:
        key, batch_key = jax.random.split(key)
        batch_inputs, batch_targets = get_batch(batch_key, dataset.train_inputs, dataset.train_targets, batch_size)
        loss_value, (trunk_grads, head_grads) = loss_and_grad(trunk_weights, head_weights, batch_inputs, batch_targets)
        last_loss = float(loss_value)

        head_updates, head_opt_state = head_optimizer.update(head_grads, head_opt_state, params=head_weights)
        head_weights = optax.apply_updates(head_weights, head_updates)

        if method == "adam":
            trunk_updates, trunk_opt_state = trunk_optimizer.update(trunk_grads, trunk_opt_state, params=trunk_weights)
            trunk_update_norm = tree_l2_norm(trunk_updates)
            trunk_weights = optax.apply_updates(trunk_weights, trunk_updates)
        elif method == "manifold":
            tangents = trunk.dual_ascent(trunk_weights, trunk_grads, target_norm=1.0)
            scaled_tangents = _scale_manifold_tangents(trunk, tangents, target_norm=target_norm)
            trunk_update_norm = tree_l2_norm(scaled_tangents)
            trunk_weights = [weight - learning_rate * tangent for weight, tangent in zip(trunk_weights, scaled_tangents)]
            trunk_weights = trunk.retract(trunk_weights)
        elif method == "manifold_online":
            tangents, dual_state = trunk.online_dual_ascent(
                dual_state,
                trunk_weights,
                trunk_grads,
                target_norm=1.0,
                alpha=dual_alpha,
                beta=dual_beta,
            )
            scaled_tangents = _scale_manifold_tangents(trunk, tangents, target_norm=target_norm)
            trunk_update_norm = tree_l2_norm(scaled_tangents)
            trunk_weights = [weight - learning_rate * tangent for weight, tangent in zip(trunk_weights, scaled_tangents)]
            trunk_weights = trunk.retract(trunk_weights)
        elif method == "manifold_admm":
            tangents = trunk.admm_dual_ascent(
                trunk_weights,
                trunk_grads,
                target_norm=1.0,
                steps=admm_steps,
                rho=admm_rho,
            )
            scaled_tangents = _scale_manifold_tangents(trunk, tangents, target_norm=target_norm)
            trunk_update_norm = tree_l2_norm(scaled_tangents)
            trunk_weights = [weight - learning_rate * tangent for weight, tangent in zip(trunk_weights, scaled_tangents)]
            trunk_weights = trunk.retract(trunk_weights)
        else:
            raise ValueError(f"Unknown method: {method}")

        head_grad_norm = tree_l2_norm(head_grads)
        head_update_norm = tree_l2_norm(head_updates)
        trunk_grad_norm = tree_l2_norm(trunk_grads)

        if step % eval_every == 0 or step == steps - 1:
            if eval_train_samples and dataset.train_inputs.shape[0] > eval_train_samples:
                eval_train_inputs = dataset.train_inputs[:eval_train_samples]
                eval_train_labels = dataset.train_labels[:eval_train_samples]
            else:
                eval_train_inputs = dataset.train_inputs
                eval_train_labels = dataset.train_labels

            train_acc = compute_accuracy(
                predict_fn,
                trunk_weights,
                head_weights,
                eval_train_inputs,
                eval_train_labels,
            )
            test_acc = compute_accuracy(
                predict_fn,
                trunk_weights,
                head_weights,
                dataset.test_inputs,
                dataset.test_labels,
            )

            elapsed_time_so_far = time.perf_counter() - start_time
            stiefel_metrics = compute_trunk_stiefel_metrics(trunk, trunk_weights)

            if logger is not None:
                logger.log(
                    {
                        "step": int(step),
                        "loss": float(loss_value),
                        "train_accuracy": float(train_acc),
                        "test_accuracy": float(test_acc),
                        "trunk_grad_norm": float(trunk_grad_norm),
                        "head_grad_norm": float(head_grad_norm),
                        "trunk_update_norm": float(trunk_update_norm),
                        "head_update_norm": float(head_update_norm),
                        "elapsed_time_seconds": float(elapsed_time_so_far),
                        "seconds_per_step_so_far": float(elapsed_time_so_far / max(step + 1, 1)),
                        **stiefel_metrics,
                    }
                )

            progress.set_description(
                f"{method} lr={learning_rate:.3g} | loss={float(loss_value):.4f} | train={train_acc:.2f}% | test={test_acc:.2f}%"
            )

    elapsed_time = time.perf_counter() - start_time
    final_train_accuracy = compute_accuracy(
        predict_fn,
        trunk_weights,
        head_weights,
        dataset.train_inputs,
        dataset.train_labels,
    )
    final_test_accuracy = compute_accuracy(
        predict_fn,
        trunk_weights,
        head_weights,
        dataset.test_inputs,
        dataset.test_labels,
    )

    final_stiefel_metrics = compute_trunk_stiefel_metrics(trunk, trunk_weights)
    result = {
        "train_accuracy": float(final_train_accuracy),
        "test_accuracy": float(final_test_accuracy),
        "final_loss": float(last_loss),
        "training_time_seconds": float(elapsed_time),
        "seconds_per_step": float(elapsed_time / max(steps, 1)),
        **final_stiefel_metrics,
    }

    if logger is not None:
        logger.log(result)

    return result


def make_run_config(args, *, dataset_name: str, num_classes: int, method: str, learning_rate: float) -> Dict[str, Any]:
    return {
        "dataset": dataset_name,
        "num_classes": int(num_classes),
        "method": method,
        "learning_rate": float(learning_rate),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "eval_every": int(args.eval_every),
        "eval_train_samples": int(args.eval_train_samples),
        "seed": int(args.seed),
        "target_norm": float(args.target_norm),
        "dual_alpha": float(args.dual_alpha),
        "dual_beta": float(args.dual_beta),
        "admm_steps": int(args.admm_steps),
        "admm_rho": float(args.admm_rho),
        "synthetic_data": bool(args.synthetic_data),
        "smoke_test": bool(args.smoke_test),
    }


def make_results_config(args, *, dataset_name: str, num_classes: int) -> Dict[str, Any]:
    return {
        "dataset": dataset_name,
        "num_classes": int(num_classes),
        "learning_rates": [float(rate) for rate in args.learning_rates],
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "eval_every": int(args.eval_every),
        "eval_train_samples": int(args.eval_train_samples),
        "seed": int(args.seed),
        "target_norm": float(args.target_norm),
        "dual_alpha": float(args.dual_alpha),
        "dual_beta": float(args.dual_beta),
        "admm_steps": int(args.admm_steps),
        "admm_rho": float(args.admm_rho),
        "methods": list(args.methods),
        "synthetic_data": bool(args.synthetic_data),
        "smoke_test": bool(args.smoke_test),
    }


def save_results(
    dataset_name: str,
    config: Mapping[str, Any],
    results: Mapping[str, Sequence[Mapping[str, Any]]],
    best_runs: Mapping[str, Mapping[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": dataset_name, "config": dict(config), "methods": {}}

    for method, runs in results.items():
        payload["methods"][method] = {
            "runs": [dict(run) for run in runs],
        }
        if method in best_runs:
            payload["methods"][method]["best"] = dict(best_runs[method])

    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def plot_best_accuracy_vs_runtime(best_runs: Mapping[str, Mapping[str, Any]], plots_dir: Path, dataset_name: str) -> None:
    if not best_runs:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    color_map = plt.get_cmap("tab10")

    for idx, (method, run) in enumerate(best_runs.items()):
        runtime = float(run["training_time_seconds"])
        accuracy = float(run["test_accuracy"])
        ax.scatter(runtime, accuracy, s=80, color=color_map(idx % 10), label=method)
        ax.annotate(method, (runtime, accuracy), textcoords="offset points", xytext=(6, 6))

    ax.set_xlabel("Runtime (seconds)")
    ax.set_ylabel("Best test accuracy (%)")
    ax.set_title(f"{dataset_name.upper()} best accuracy vs runtime")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(plots_dir / f"{dataset_name}_best_accuracy_vs_runtime.png", dpi=300)
    plt.close(fig)


def print_run_summary(method: str, run_result: Mapping[str, Any]) -> None:
    print(
        f"[{method}] lr={run_result['learning_rate']:.3g}: "
        f"train acc={run_result['train_accuracy']:.2f}% | "
        f"test acc={run_result['test_accuracy']:.2f}% | "
        f"loss={run_result['final_loss']:.4f} | "
        f"time={run_result['training_time_seconds']:.2f}s"
    )
