import argparse
import json
import math
import pickle
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
import urllib.request

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import optax
from tqdm import tqdm

from benchmark.scaling import iter_weighted_atoms, manifold_directions, manifold_update_scale
from modula.atom import Linear

METHOD_CHOICES = ("sgd", "adam", "adamw", "manifold")


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


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_learning_rate: float,
    default_steps: int,
    default_batch_size: int,
    default_eval_every: int,
    default_output_path: Path,
) -> argparse.ArgumentParser:
    parser.add_argument("--learning-rate", type=float, default=default_learning_rate, help="Optimizer learning rate")
    parser.add_argument("--steps", type=int, default=default_steps, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=default_batch_size, help="Mini-batch size")
    parser.add_argument("--eval-every", type=int, default=default_eval_every, help="Metric logging interval")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    parser.add_argument("--output", type=Path, default=default_output_path, help="Path for summary metrics")
    parser.add_argument("--synthetic-data", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser


def validate_common_arguments(parser: argparse.ArgumentParser, args) -> None:
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.eval_every <= 0:
        parser.error("--eval-every must be positive")


def apply_smoke_test_overrides(args) -> None:
    if not args.smoke_test:
        return
    args.steps = min(int(args.steps), 2)
    args.batch_size = min(int(args.batch_size), 8)
    args.eval_every = 1


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
) -> DatasetBundle:
    train_size = 64 if smoke_test else 256
    test_size = 32 if smoke_test else 128
    rng = np.random.default_rng(seed)
    train_images, train_labels = _synthetic_split(train_size, num_classes, rng)
    test_images, test_labels = _synthetic_split(test_size, num_classes, rng)
    return DatasetBundle(
        name=name,
        num_classes=num_classes,
        train_inputs=normalize_cifar_images(train_images),
        train_targets=one_hot(jnp.asarray(train_labels, dtype=jnp.int32), num_classes),
        train_labels=jnp.asarray(train_labels, dtype=jnp.int32),
        test_inputs=normalize_cifar_images(test_images),
        test_targets=one_hot(jnp.asarray(test_labels, dtype=jnp.int32), num_classes),
        test_labels=jnp.asarray(test_labels, dtype=jnp.int32),
    )


def load_cifar10(normalize=True):
    data_dir = Path(__file__).resolve().parent / "cifar10_files"
    data_dir.mkdir(parents=True, exist_ok=True)

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filepath = data_dir / "cifar-10-python.tar.gz"
    extracted_dir = data_dir / "cifar-10-batches-py"

    if not extracted_dir.exists():
        if not filepath.is_file():
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, filepath)
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(data_dir)

    def load_batch(filename):
        with (extracted_dir / filename).open("rb") as handle:
            batch = pickle.load(handle, encoding="bytes")
        images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(batch[b"labels"])
        return images, labels

    train_images, train_labels = [], []
    for idx in range(1, 6):
        images, labels = load_batch(f"data_batch_{idx}")
        train_images.append(images)
        train_labels.append(labels)

    train_images = np.concatenate(train_images)
    train_labels = np.concatenate(train_labels)
    test_images, test_labels = load_batch("test_batch")

    if normalize:
        train_images = train_images.astype(np.float32) / 255.0
        test_images = test_images.astype(np.float32) / 255.0

    return train_images, train_labels, test_images, test_labels


def normalize_cifar_images(images):
    images = jnp.asarray(images, dtype=jnp.float32)
    return 2.0 * images - 1.0


def prepare_dataset(
    dataset_name: str,
    *,
    synthetic_data: bool = False,
    smoke_test: bool = False,
    seed: int = 0,
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
        )

    train_images, train_labels, test_images, test_labels = loader(normalize=True)
    train_labels = jnp.asarray(train_labels, dtype=jnp.int32)
    test_labels = jnp.asarray(test_labels, dtype=jnp.int32)

    return DatasetBundle(
        name=dataset_name,
        num_classes=num_classes,
        train_inputs=normalize_cifar_images(train_images),
        train_targets=one_hot(train_labels, num_classes),
        train_labels=train_labels,
        test_inputs=normalize_cifar_images(test_images),
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


def _manifold_update_scale(atom) -> float:
    return manifold_update_scale(atom)


def _manifold_directions(module, tangents, *, atoms: Sequence[Any] | None = None):
    return manifold_directions(module, tangents, atoms=atoms)


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


def _cross_entropy_loss_fn(predict_fn, trunk_weights, head_weights, inputs, targets):
    logits = predict_fn(trunk_weights, head_weights, inputs)
    return jnp.mean(optax.softmax_cross_entropy(logits, targets))


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
    seed: int,
    method: str,
    adam_weight_decay: float = 0.01,
    sgd_momentum: float = 0.9,
    manifold_momentum: float = 0.9,
    manifold_weight_decay: float = 0.01,
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
    loss_fn = _cross_entropy_loss_fn
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
    elif method != "manifold":
        raise ValueError(f"Unknown method: {method}")

    momentum_state = [jnp.zeros_like(weight) for weight in trunk_weights] if method == "manifold" else None
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
        else:
            momentum_state = [
                manifold_momentum * momentum + grad for momentum, grad in zip(momentum_state, trunk_grads)
            ]
            tangents = trunk.dual_ascent(trunk_weights, momentum_state)
            trunk_directions = _manifold_directions(trunk, tangents)
            trunk_updates = [
                direction + manifold_weight_decay * weight
                for weight, direction in zip(trunk_weights, trunk_directions)
            ]
            trunk_update_norm = tree_l2_norm(trunk_updates)
            trunk_weights = [
                weight - learning_rate * update for weight, update in zip(trunk_weights, trunk_updates)
            ]

        if method == "manifold" or project_trunk_after_update:
            trunk_weights = trunk.retract(trunk_weights)

        if step % eval_every == 0 or step == steps - 1:
            eval_train_inputs = dataset.train_inputs[: min(dataset.train_inputs.shape[0], 1000)]
            eval_train_labels = dataset.train_labels[: min(dataset.train_labels.shape[0], 1000)]
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

            epoch = (step + 1) * batch_size / max(int(dataset.train_inputs.shape[0]), 1)
            head_grad_norm = tree_l2_norm(head_grads)
            head_update_norm = tree_l2_norm(head_updates)
            trunk_grad_norm = tree_l2_norm(trunk_grads)
            progress.set_description(
                f"{method} lr={learning_rate:.3g} | epoch={epoch:.2f} | "
                f"loss={float(loss_value):.4f} | train={train_acc:.2f}% | test={test_acc:.2f}% | "
                f"grad={trunk_grad_norm + head_grad_norm:.2f} | update={trunk_update_norm + head_update_norm:.2f}"
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
    return {
        "train_accuracy": float(final_train_accuracy),
        "test_accuracy": float(final_test_accuracy),
        "final_loss": float(last_loss),
        "full_train_loss": float(full_train_loss),
        "loss_name": "cross_entropy",
        "steps": int(steps),
        "final_epoch": float(final_epoch),
        "training_time_seconds": float(elapsed_time),
        "seconds_per_step": float(elapsed_time / max(steps, 1)),
        **final_geometry_metrics,
    }


def make_run_config(
    args,
    *,
    dataset_name: str,
    num_classes: int,
    method: str,
    hidden_size: int,
    parameterization: str,
    mup_base_width: int | None = None,
) -> Dict[str, Any]:
    config = {
        "dataset": dataset_name,
        "num_classes": int(num_classes),
        "method": method,
        "learning_rate": float(args.learning_rate),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "eval_every": int(args.eval_every),
        "seed": int(args.seed),
        "hidden_size": int(hidden_size),
        "parameterization": parameterization,
        "synthetic_data": bool(args.synthetic_data),
        "smoke_test": bool(args.smoke_test),
    }
    if mup_base_width is not None:
        config["mup_base_width"] = int(mup_base_width)
    return config


def save_result(dataset_name: str, config: Mapping[str, Any], result: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": dataset_name, "config": dict(config), "result": dict(result)}
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)


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
