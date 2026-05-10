import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import jax
import jax.numpy as jnp
import jax.tree_util
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import tqdm

from examples.data.cifar10 import load_cifar10
from benchmark.scaling import (
    SCALING_CHOICES,
    iter_weighted_atoms,
    manifold_directions,
    muon_directions,
)
from modula.atom import Linear

MANIFOLD_METHODS = ("manifold", "manifold_online", "manifold_admm")
METHOD_CHOICES = (
    "adam",
    "adamw",
    "sgd",
    "muon",
    *MANIFOLD_METHODS,
)
MUON_SCALING_CHOICES = SCALING_CHOICES
LOSS_CHOICES = ("cross_entropy", "mse")
CIFAR_NORMALIZATION_CHOICES = ("minus_one_one", "zero_one")


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
    parser.add_argument(
        "--epochs",
        type=float,
        default=None,
        help="Training epochs per learning rate; overrides --steps when set",
    )
    parser.add_argument("--batch-size", type=int, default=default_batch_size, help="Mini-batch size")
    parser.add_argument(
        "--cifar-normalization",
        type=str,
        default="minus_one_one",
        choices=CIFAR_NORMALIZATION_CHOICES,
        help="CIFAR input normalization; minus_one_one matches torchvision Normalize((0.5,), (0.5,))",
    )
    parser.add_argument("--eval-every", type=int, default=default_eval_every, help="Metric logging interval")
    parser.add_argument(
        "--loss",
        type=str,
        default="cross_entropy",
        choices=LOSS_CHOICES,
        help="Training loss for classifier benchmarks",
    )
    parser.add_argument(
        "--eval-train-samples",
        type=int,
        default=1000,
        help="Train samples used for periodic eval (0 for the full train set)",
    )
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    parser.add_argument(
        "--adam-weight-decay",
        type=float,
        default=0.01,
        help="Decoupled weight decay for AdamW baseline",
    )
    parser.add_argument(
        "--sgd-momentum",
        type=float,
        default=0.9,
        help="Momentum coefficient for SGD baseline",
    )
    parser.add_argument("--dual-alpha", type=float, default=2e-5, help="Alpha for manifold_online")
    parser.add_argument("--dual-beta", type=float, default=0.9, help="Beta for manifold_online")
    parser.add_argument("--admm-steps", type=int, default=10, help="ADMM inner steps for manifold_admm")
    parser.add_argument("--admm-rho", type=float, default=4.0, help="ADMM penalty for manifold_admm")
    parser.add_argument(
        "--manifold-momentum",
        type=float,
        default=0.9,
        help="Momentum coefficient for manifold-family trunk updates",
    )
    parser.add_argument(
        "--manifold-weight-decay",
        type=float,
        default=0.01,
        help="Decoupled weight decay for manifold-family trunk updates",
    )
    parser.add_argument(
        "--manifold-scaling",
        type=str,
        default="fan_ratio",
        choices=MUON_SCALING_CHOICES,
        help="Scaling rule for manifold-family trunk updates",
    )
    parser.add_argument(
        "--muon-scaling",
        type=str,
        default="fan_ratio",
        choices=MUON_SCALING_CHOICES,
        help="Scaling rule for Muon trunk updates",
    )
    parser.add_argument(
        "--muon-momentum",
        type=float,
        default=0.9,
        help="Momentum coefficient for Muon trunk updates",
    )
    parser.add_argument(
        "--muon-weight-decay",
        type=float,
        default=0.01,
        help="Decoupled weight decay for Muon trunk updates",
    )
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
        "--wandb-group",
        type=str,
        default=None,
        help="Weights & Biases group name (defaults to the benchmark group)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity (username or team)",
    )
    return parser


def validate_common_arguments(parser: argparse.ArgumentParser, args) -> None:
    if args.epochs is not None and args.epochs <= 0.0:
        parser.error("--epochs must be positive")
    if args.sgd_momentum < 0.0:
        parser.error("--sgd-momentum must be nonnegative")


def apply_smoke_test_overrides(args) -> None:
    if not args.smoke_test:
        return
    args.learning_rates = [float(args.learning_rates[0] if args.learning_rates else 1e-2)]
    args.steps = min(int(args.steps), 2)
    args.batch_size = min(int(args.batch_size), 8)
    args.eval_every = 1
    args.eval_train_samples = 16 if int(args.eval_train_samples) == 0 else min(int(args.eval_train_samples), 16)


def resolve_training_steps(args, dataset: DatasetBundle) -> None:
    if args.epochs is None:
        return

    train_size = int(dataset.train_inputs.shape[0])
    batch_size = int(args.batch_size)
    args.steps = max(1, int(math.ceil(float(args.epochs) * train_size / batch_size)))


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


def _make_synthetic_dataset(
    name: str,
    num_classes: int,
    *,
    smoke_test: bool,
    seed: int,
    cifar_normalization: str,
) -> DatasetBundle:
    train_size = 64 if smoke_test else 256
    test_size = 32 if smoke_test else 128
    rng = np.random.default_rng(seed)
    train_images, train_labels = _synthetic_split(train_size, num_classes, rng)
    test_images, test_labels = _synthetic_split(test_size, num_classes, rng)
    return DatasetBundle(
        name=name,
        num_classes=num_classes,
        train_inputs=normalize_cifar_images(train_images, normalization=cifar_normalization),
        train_targets=one_hot(jnp.asarray(train_labels, dtype=jnp.int32), num_classes),
        train_labels=jnp.asarray(train_labels, dtype=jnp.int32),
        test_inputs=normalize_cifar_images(test_images, normalization=cifar_normalization),
        test_targets=one_hot(jnp.asarray(test_labels, dtype=jnp.int32), num_classes),
        test_labels=jnp.asarray(test_labels, dtype=jnp.int32),
    )


def normalize_cifar_images(images, *, normalization: str):
    images = jnp.asarray(images, dtype=jnp.float32)
    if normalization == "zero_one":
        return images
    if normalization == "minus_one_one":
        return 2.0 * images - 1.0
    raise ValueError(f"Unknown CIFAR normalization: {normalization}")


def prepare_dataset(
    dataset_name: str,
    *,
    synthetic_data: bool = False,
    smoke_test: bool = False,
    seed: int = 0,
    cifar_normalization: str = "minus_one_one",
):
    if dataset_name == "cifar10":
        loader = load_cifar10
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if synthetic_data:
        return _make_synthetic_dataset(
            dataset_name,
            num_classes,
            smoke_test=smoke_test,
            seed=seed,
            cifar_normalization=cifar_normalization,
        )

    train_images, train_labels, test_images, test_labels = loader(normalize=True)
    train_labels = jnp.asarray(train_labels, dtype=jnp.int32)
    test_labels = jnp.asarray(test_labels, dtype=jnp.int32)

    return DatasetBundle(
        name=dataset_name,
        num_classes=num_classes,
        train_inputs=normalize_cifar_images(train_images, normalization=cifar_normalization),
        train_targets=one_hot(train_labels, num_classes),
        train_labels=train_labels,
        test_inputs=normalize_cifar_images(test_images, normalization=cifar_normalization),
        test_targets=one_hot(test_labels, num_classes),
        test_labels=test_labels,
    )


def get_batch(key, X, y, batch_size):
    if X.shape[0] == 0:
        raise ValueError("Cannot sample from an empty dataset")
    replace = X.shape[0] < batch_size
    idx = jax.random.choice(key, X.shape[0], shape=(batch_size,), replace=replace)
    return X[idx], y[idx]


def make_predict_fn(trunk, head):
    return jax.jit(lambda trunk_w, head_w, inputs: head(trunk(inputs, trunk_w), head_w))


def tree_l2_norm(tree) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return 0.0
    squared_norm = sum(jnp.sum(jnp.square(jnp.asarray(leaf, dtype=jnp.float32))) for leaf in leaves)
    return float(jnp.sqrt(squared_norm))


def _iter_weighted_atoms(module) -> List[Any]:
    return iter_weighted_atoms(module)


def _manifold_directions(module, tangents, *, scaling: str):
    return manifold_directions(module, tangents, scaling=scaling)


def _muon_directions(module, grads, *, scaling: str):
    return muon_directions(module, grads, scaling=scaling)


def _weight_to_stiefel_matrix(atom, weight):
    array = jnp.asarray(weight, dtype=jnp.float32)
    if isinstance(atom, Linear):
        return array
    return None


def _stiefel_target_scale(atom):
    if isinstance(atom, Linear):
        radius = float(getattr(atom, "stiefel_radius", 1.0))
        return jnp.asarray(radius, dtype=jnp.float32)
    return None


def _linear_rms_to_rms_norm(atom, matrix) -> float | None:
    if not isinstance(atom, Linear):
        return None

    array = jnp.asarray(matrix, dtype=jnp.float32)
    spectral_norm = jnp.linalg.norm(array, ord=2)
    scale = jnp.sqrt(jnp.asarray(atom.fanin / atom.fanout, dtype=array.dtype))
    return float(scale * spectral_norm)


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


def compute_trunk_geometry_metrics(trunk, trunk_weights) -> Dict[str, float]:
    atoms = _iter_weighted_atoms(trunk)
    deviations: List[float] = []
    rms_deviations: List[float] = []
    metrics: Dict[str, float] = {}

    for layer_idx, (atom, weight) in enumerate(zip(atoms, trunk_weights)):
        deviation = stiefel_deviation(atom, weight)
        if deviation is None:
            continue
        deviations.append(deviation)
        metrics[f"trunk_stiefel_deviation_layer_{layer_idx}"] = float(deviation)

        rms_norm = _linear_rms_to_rms_norm(atom, weight)
        if rms_norm is not None:
            rms_deviation = abs(rms_norm - 1.0)
            rms_deviations.append(rms_deviation)
            metrics[f"trunk_rms_to_rms_norm_layer_{layer_idx}"] = float(rms_norm)
            metrics[f"trunk_rms_to_rms_deviation_layer_{layer_idx}"] = float(rms_deviation)

    if deviations:
        metrics["trunk_stiefel_deviation_mean"] = float(np.mean(deviations))
        metrics["trunk_stiefel_deviation_max"] = float(np.max(deviations))
    else:
        metrics["trunk_stiefel_deviation_mean"] = 0.0
        metrics["trunk_stiefel_deviation_max"] = 0.0

    if rms_deviations:
        metrics["trunk_rms_to_rms_deviation_mean"] = float(np.mean(rms_deviations))
        metrics["trunk_rms_to_rms_deviation_max"] = float(np.max(rms_deviations))
    else:
        metrics["trunk_rms_to_rms_deviation_mean"] = 0.0
        metrics["trunk_rms_to_rms_deviation_max"] = 0.0

    return metrics


def compute_trunk_stiefel_metrics(trunk, trunk_weights) -> Dict[str, float]:
    return compute_trunk_geometry_metrics(trunk, trunk_weights)


def compute_accuracy(predict_fn, trunk_weights, head_weights, inputs, labels, *, batch_size: int = 1024) -> float:
    total = inputs.shape[0]
    correct = 0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        logits = predict_fn(trunk_weights, head_weights, inputs[start:end])
        predictions = jnp.argmax(logits, axis=1)
        correct += int(jnp.sum(predictions == labels[start:end]))

    return 100.0 * correct / total


def compute_mean_loss(
    predict_fn,
    loss_fn,
    trunk_weights,
    head_weights,
    inputs,
    targets,
    *,
    batch_size: int = 1024,
) -> float:
    total = inputs.shape[0]
    weighted_loss = 0.0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_loss = loss_fn(predict_fn, trunk_weights, head_weights, inputs[start:end], targets[start:end])
        weighted_loss += float(batch_loss) * (end - start)

    return weighted_loss / total


def _mse_loss_fn(predict_fn, trunk_weights, head_weights, inputs, targets):
    logits = predict_fn(trunk_weights, head_weights, inputs)
    return jnp.mean((logits - targets) ** 2)


def _cross_entropy_loss_fn(predict_fn, trunk_weights, head_weights, inputs, targets):
    logits = predict_fn(trunk_weights, head_weights, inputs)
    return jnp.mean(optax.softmax_cross_entropy(logits, targets))


def _resolve_loss_fn(loss: str):
    if loss == "cross_entropy":
        return _cross_entropy_loss_fn
    if loss == "mse":
        return _mse_loss_fn
    raise ValueError(f"Unknown loss: {loss}")


def _scale_update_tree(updates, scale: float):
    return jax.tree_util.tree_map(
        lambda update: jnp.asarray(scale, dtype=update.dtype) * update,
        updates,
    )


def _decoupled_weight_decay_updates(params, *, learning_rate: float, weight_decay: float):
    return jax.tree_util.tree_map(
        lambda param: -jnp.asarray(learning_rate * weight_decay, dtype=param.dtype) * param,
        params,
    )


def _scaled_head_optimizer_update(
    method: str,
    head_optimizer,
    head_opt_state,
    head_grads,
    head_weights,
    *,
    learning_rate: float,
    adam_weight_decay: float,
    head_adam_update_scale: float,
):
    head_updates, head_opt_state = head_optimizer.update(head_grads, head_opt_state, params=head_weights)
    if method == "sgd":
        return head_updates, head_opt_state

    scaled_head_updates = _scale_update_tree(head_updates, head_adam_update_scale)
    if method != "adamw":
        return scaled_head_updates, head_opt_state

    decay_updates = _decoupled_weight_decay_updates(
        head_weights,
        learning_rate=learning_rate,
        weight_decay=adam_weight_decay,
    )
    return jax.tree_util.tree_map(lambda update, decay: update + decay, scaled_head_updates, decay_updates), head_opt_state


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
    dual_alpha: float,
    dual_beta: float,
    admm_steps: int,
    admm_rho: float,
    adam_weight_decay: float = 0.01,
    manifold_momentum: float = 0.9,
    manifold_weight_decay: float = 0.01,
    manifold_scaling: str = "fan_ratio",
    muon_scaling: str = "fan_ratio",
    muon_momentum: float = 0.9,
    muon_weight_decay: float = 0.01,
    sgd_momentum: float = 0.9,
    loss: str = "cross_entropy",
    logger=None,
    show_progress: bool = True,
    project_trunk_after_update: bool = False,
    head_adam_update_scale: float = 1.0,
    head_init_scale: float = 1.0,
):
    key = jax.random.PRNGKey(seed)
    key, trunk_key, head_key = jax.random.split(key, 3)

    trunk_weights = trunk.initialize(trunk_key)
    head_weights = _scale_update_tree(head.initialize(head_key), head_init_scale)
    predict_fn = make_predict_fn(trunk, head)
    loss_fn = _resolve_loss_fn(loss)
    loss_and_grad = jax.jit(jax.value_and_grad(lambda tw, hw, x, y: loss_fn(predict_fn, tw, hw, x, y), argnums=(0, 1)))

    if method == "sgd":
        head_optimizer = optax.sgd(learning_rate, momentum=sgd_momentum)
    else:
        head_optimizer = optax.adam(learning_rate)
    head_opt_state = head_optimizer.init(head_weights)

    trunk_optimizer = None
    trunk_opt_state = None
    if method == "adam":
        trunk_optimizer = optax.adam(learning_rate)
        trunk_opt_state = trunk_optimizer.init(trunk_weights)
    elif method == "adamw":
        trunk_optimizer = optax.adamw(learning_rate, weight_decay=adam_weight_decay)
        trunk_opt_state = trunk_optimizer.init(trunk_weights)
    elif method == "sgd":
        trunk_optimizer = optax.sgd(learning_rate, momentum=sgd_momentum)
        trunk_opt_state = trunk_optimizer.init(trunk_weights)

    dual_state = trunk.init_dual_state(trunk_weights) if method == "manifold_online" else None
    momentum_state = [jnp.zeros_like(weight) for weight in trunk_weights] if method in MANIFOLD_METHODS else None
    muon_momentum_state = [jnp.zeros_like(weight) for weight in trunk_weights] if method == "muon" else None
    progress = tqdm(range(steps), desc=f"{method} lr={learning_rate:.3g}", leave=False, disable=not show_progress)
    start_time = time.perf_counter()
    last_loss = 0.0

    for step in progress:
        key, batch_key = jax.random.split(key)
        batch_inputs, batch_targets = get_batch(batch_key, dataset.train_inputs, dataset.train_targets, batch_size)
        loss_value, (trunk_grads, head_grads) = loss_and_grad(trunk_weights, head_weights, batch_inputs, batch_targets)
        last_loss = float(loss_value)

        head_updates, head_opt_state = _scaled_head_optimizer_update(
            method,
            head_optimizer,
            head_opt_state,
            head_grads,
            head_weights,
            learning_rate=learning_rate,
            adam_weight_decay=adam_weight_decay,
            head_adam_update_scale=head_adam_update_scale,
        )
        head_weights = optax.apply_updates(head_weights, head_updates)

        if method in ("adam", "adamw", "sgd"):
            trunk_updates, trunk_opt_state = trunk_optimizer.update(trunk_grads, trunk_opt_state, params=trunk_weights)
            trunk_update_norm = tree_l2_norm(trunk_updates)
            trunk_weights = optax.apply_updates(trunk_weights, trunk_updates)
        elif method == "muon":
            muon_momentum_state = [
                muon_momentum * momentum + grad for momentum, grad in zip(muon_momentum_state, trunk_grads)
            ]
            trunk_directions = _muon_directions(
                trunk,
                muon_momentum_state,
                scaling=muon_scaling,
            )
            trunk_updates = [
                direction + muon_weight_decay * weight
                for weight, direction in zip(trunk_weights, trunk_directions)
            ]
            trunk_update_norm = tree_l2_norm(trunk_updates)
            trunk_weights = [
                weight - learning_rate * update for weight, update in zip(trunk_weights, trunk_updates)
            ]
        elif method in MANIFOLD_METHODS:
            momentum_state = [
                manifold_momentum * momentum + grad for momentum, grad in zip(momentum_state, trunk_grads)
            ]
            solver_grads = momentum_state

            if method == "manifold":
                tangents = trunk.dual_ascent(trunk_weights, solver_grads)
            elif method == "manifold_online":
                tangents, dual_state = trunk.online_dual_ascent(
                    dual_state,
                    trunk_weights,
                    solver_grads,
                    alpha=dual_alpha,
                    beta=dual_beta,
                )
            else:
                tangents = trunk.admm_dual_ascent(
                    trunk_weights,
                    solver_grads,
                    steps=admm_steps,
                    rho=admm_rho,
                )

            trunk_directions = _manifold_directions(
                trunk,
                tangents,
                scaling=manifold_scaling,
            )
            trunk_updates = [
                direction + manifold_weight_decay * weight
                for weight, direction in zip(trunk_weights, trunk_directions)
            ]
            trunk_update_norm = tree_l2_norm(trunk_updates)
            trunk_weights = [
                weight - learning_rate * update for weight, update in zip(trunk_weights, trunk_updates)
            ]
        else:
            raise ValueError(f"Unknown method: {method}")

        if method in MANIFOLD_METHODS or project_trunk_after_update:
            trunk_weights = trunk.retract(trunk_weights)

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
            epoch = (step + 1) * batch_size / max(int(dataset.train_inputs.shape[0]), 1)
            geometry_metrics = compute_trunk_geometry_metrics(trunk, trunk_weights)

            if logger is not None:
                logger.log(
                    {
                        "epoch": float(epoch),
                        "loss": float(loss_value),
                        "loss_name": loss,
                        "train_accuracy": float(train_acc),
                        "test_accuracy": float(test_acc),
                        "trunk_grad_norm": float(trunk_grad_norm),
                        "head_grad_norm": float(head_grad_norm),
                        "trunk_update_norm": float(trunk_update_norm),
                        "head_update_norm": float(head_update_norm),
                        "elapsed_time_seconds": float(elapsed_time_so_far),
                        "seconds_per_step_so_far": float(elapsed_time_so_far / max(step + 1, 1)),
                        **geometry_metrics,
                    }
                )

            progress.set_description(
                f"{method} lr={learning_rate:.3g} | epoch={epoch:.2f} | loss={float(loss_value):.4f} | train={train_acc:.2f}% | test={test_acc:.2f}%"
            )

    elapsed_time = time.perf_counter() - start_time
    final_epoch = steps * batch_size / max(int(dataset.train_inputs.shape[0]), 1)
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
    full_train_loss = compute_mean_loss(
        predict_fn,
        loss_fn,
        trunk_weights,
        head_weights,
        dataset.train_inputs,
        dataset.train_targets,
    )

    final_geometry_metrics = compute_trunk_geometry_metrics(trunk, trunk_weights)
    result = {
        "train_accuracy": float(final_train_accuracy),
        "test_accuracy": float(final_test_accuracy),
        "final_loss": float(last_loss),
        "full_train_loss": float(full_train_loss),
        "loss_name": loss,
        "steps": int(steps),
        "final_epoch": float(final_epoch),
        "training_time_seconds": float(elapsed_time),
        "seconds_per_step": float(elapsed_time / max(steps, 1)),
        **final_geometry_metrics,
    }

    if logger is not None:
        logger.log(result)

    return result


def make_run_config(
    args,
    *,
    dataset_name: str,
    num_classes: int,
    method: str,
    learning_rate: float,
    hidden_size: int | None = None,
) -> Dict[str, Any]:
    config = {
        "dataset": dataset_name,
        "num_classes": int(num_classes),
        "method": method,
        "learning_rate": float(learning_rate),
        "steps": int(args.steps),
        "epochs": float(args.epochs) if args.epochs is not None else None,
        "batch_size": int(args.batch_size),
        "cifar_normalization": str(args.cifar_normalization),
        "eval_every": int(args.eval_every),
        "loss": str(args.loss),
        "eval_train_samples": int(args.eval_train_samples),
        "seed": int(args.seed),
        "adam_weight_decay": float(args.adam_weight_decay),
        "sgd_momentum": float(args.sgd_momentum),
        "dual_alpha": float(args.dual_alpha),
        "dual_beta": float(args.dual_beta),
        "admm_steps": int(args.admm_steps),
        "admm_rho": float(args.admm_rho),
        "manifold_momentum": float(args.manifold_momentum),
        "manifold_weight_decay": float(args.manifold_weight_decay),
        "manifold_scaling": str(args.manifold_scaling),
        "muon_scaling": str(args.muon_scaling),
        "muon_momentum": float(args.muon_momentum),
        "muon_weight_decay": float(args.muon_weight_decay),
        "wandb_group": args.wandb_group,
        "synthetic_data": bool(args.synthetic_data),
        "smoke_test": bool(args.smoke_test),
    }
    if hidden_size is not None:
        config["hidden_size"] = int(hidden_size)
    if hasattr(args, "trunk"):
        config["trunk"] = str(args.trunk)
    if hasattr(args, "linear_normalizations"):
        config["linear_normalizations"] = list(args.linear_normalizations)
    if hasattr(args, "mup_base_width"):
        config["mup_base_width"] = int(args.mup_base_width)
    return config


def make_results_config(args, *, dataset_name: str, num_classes: int) -> Dict[str, Any]:
    config = {
        "dataset": dataset_name,
        "num_classes": int(num_classes),
        "learning_rates": [float(rate) for rate in args.learning_rates],
        "steps": int(args.steps),
        "epochs": float(args.epochs) if args.epochs is not None else None,
        "batch_size": int(args.batch_size),
        "cifar_normalization": str(args.cifar_normalization),
        "eval_every": int(args.eval_every),
        "loss": str(args.loss),
        "eval_train_samples": int(args.eval_train_samples),
        "seed": int(args.seed),
        "adam_weight_decay": float(args.adam_weight_decay),
        "sgd_momentum": float(args.sgd_momentum),
        "dual_alpha": float(args.dual_alpha),
        "dual_beta": float(args.dual_beta),
        "admm_steps": int(args.admm_steps),
        "admm_rho": float(args.admm_rho),
        "manifold_momentum": float(args.manifold_momentum),
        "manifold_weight_decay": float(args.manifold_weight_decay),
        "manifold_scaling": str(args.manifold_scaling),
        "muon_scaling": str(args.muon_scaling),
        "muon_momentum": float(args.muon_momentum),
        "muon_weight_decay": float(args.muon_weight_decay),
        "wandb_group": args.wandb_group,
        "methods": list(args.methods),
        "synthetic_data": bool(args.synthetic_data),
        "smoke_test": bool(args.smoke_test),
    }
    if hasattr(args, "hidden_sizes"):
        config["hidden_sizes"] = [int(hidden_size) for hidden_size in args.hidden_sizes]
    if hasattr(args, "trunk"):
        config["trunk"] = str(args.trunk)
    if hasattr(args, "linear_normalizations"):
        config["linear_normalizations"] = list(args.linear_normalizations)
    if hasattr(args, "mup_base_width"):
        config["mup_base_width"] = int(args.mup_base_width)
    return config


def summarize_lr_transfer(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[int, List[Mapping[str, Any]]]] = {}
    for run in runs:
        if "hidden_size" not in run or "linear_normalization" not in run or "learning_rate" not in run:
            continue
        variant = str(run["linear_normalization"])
        hidden_size = int(run["hidden_size"])
        grouped.setdefault(variant, {}).setdefault(hidden_size, []).append(run)

    summary: Dict[str, Any] = {}
    for variant, by_hidden in grouped.items():
        best_lr_by_hidden: Dict[str, float] = {}
        best_accuracy_by_hidden: Dict[str, float] = {}
        accuracy_by_hidden_lr: Dict[str, Dict[str, float]] = {}

        for hidden_size, hidden_runs in sorted(by_hidden.items()):
            best = max(hidden_runs, key=lambda run: float(run.get("test_accuracy", float("-inf"))))
            hidden_key = str(hidden_size)
            best_lr_by_hidden[hidden_key] = float(best["learning_rate"])
            best_accuracy_by_hidden[hidden_key] = float(best["test_accuracy"])
            accuracy_by_hidden_lr[hidden_key] = {
                f"{float(run['learning_rate']):.12g}": float(run["test_accuracy"])
                for run in sorted(hidden_runs, key=lambda run: float(run["learning_rate"]))
            }

        best_lrs = [rate for rate in best_lr_by_hidden.values() if rate > 0.0]
        if len(best_lrs) > 1:
            lr_logs = np.log10(np.asarray(best_lrs, dtype=np.float64))
            lr_spread = float(np.max(lr_logs) - np.min(lr_logs))
        else:
            lr_spread = 0.0

        summary[variant] = {
            "best_lr_by_hidden_size": best_lr_by_hidden,
            "best_test_accuracy_by_hidden_size": best_accuracy_by_hidden,
            "test_accuracy_by_hidden_size_and_lr": accuracy_by_hidden_lr,
            "best_lr_log10_spread": lr_spread,
        }

    return summary


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
        transfer_summary = summarize_lr_transfer(runs)
        if transfer_summary:
            payload["methods"][method]["lr_transfer"] = transfer_summary
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


def _plot_safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def plot_cifar_mlp_lr_transfer(results: Mapping[str, Sequence[Mapping[str, Any]]], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    for method, runs in results.items():
        grouped: Dict[str, Dict[int, List[Mapping[str, Any]]]] = {}
        for run in runs:
            if "hidden_size" not in run or "linear_normalization" not in run:
                continue
            grouped.setdefault(str(run["linear_normalization"]), {}).setdefault(int(run["hidden_size"]), []).append(run)

        for variant, by_hidden in grouped.items():
            if not by_hidden:
                continue

            fig, ax = plt.subplots(figsize=(7, 5))
            for hidden_size, hidden_runs in sorted(by_hidden.items()):
                sorted_runs = sorted(hidden_runs, key=lambda run: float(run["learning_rate"]))
                rates = [float(run["learning_rate"]) for run in sorted_runs]
                accuracies = [float(run["test_accuracy"]) for run in sorted_runs]
                ax.plot(rates, accuracies, marker="o", label=f"h={hidden_size}")

            ax.set_xscale("log")
            ax.set_xlabel("Learning rate")
            ax.set_ylabel("Test accuracy (%)")
            ax.set_title(f"{method} {variant}: accuracy vs LR")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                plots_dir / f"cifar10_mlp_{_plot_safe_name(method)}_{_plot_safe_name(variant)}_accuracy_vs_lr.png",
                dpi=300,
            )
            plt.close(fig)

        if grouped:
            fig, ax = plt.subplots(figsize=(7, 5))
            for variant, by_hidden in sorted(grouped.items()):
                hidden_sizes: List[int] = []
                best_lrs: List[float] = []
                for hidden_size, hidden_runs in sorted(by_hidden.items()):
                    best = max(hidden_runs, key=lambda run: float(run["test_accuracy"]))
                    hidden_sizes.append(hidden_size)
                    best_lrs.append(float(best["learning_rate"]))
                ax.plot(hidden_sizes, best_lrs, marker="o", label=variant)

            ax.set_yscale("log")
            ax.set_xlabel("Hidden size")
            ax.set_ylabel("Best learning rate")
            ax.set_title(f"{method}: best LR by hidden size")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots_dir / f"cifar10_mlp_{_plot_safe_name(method)}_best_lr_by_hidden_size.png", dpi=300)
            plt.close(fig)


def print_run_summary(method: str, run_result: Mapping[str, Any]) -> None:
    variant = f" {run_result['linear_normalization']}" if "linear_normalization" in run_result else ""
    print(
        f"[{method}{variant}] lr={run_result['learning_rate']:.3g}: "
        f"train acc={run_result['train_accuracy']:.2f}% | "
        f"test acc={run_result['test_accuracy']:.2f}% | "
        f"loss={run_result['final_loss']:.4f} | "
        f"full train loss={run_result['full_train_loss']:.4f} | "
        f"epoch={run_result['final_epoch']:.2f} | "
        f"time={run_result['training_time_seconds']:.2f}s"
    )
