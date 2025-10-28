import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import trange

from data.cifar100 import load_cifar100
from modula.atom import Linear, matrix_sign
from modula.bond import ReLU

METHOD_ALIASES = {
    "descent": "descent",
    "gd": "descent",
    "dualize": "dualize",
    "manifold_online": "manifold_online",
    "online_manifold": "manifold_online",
}
METHOD_CHOICES = tuple(sorted(METHOD_ALIASES.keys()))


@dataclass
class TaskData:
    class_pair: Tuple[int, int]
    train_inputs: jnp.ndarray
    train_targets: jnp.ndarray
    train_binary_labels: jnp.ndarray
    test_inputs: jnp.ndarray
    test_binary_labels: jnp.ndarray


def prepare_data():
    train_images, train_labels, test_images, test_labels = load_cifar100()

    X_train = jnp.asarray(train_images, dtype=jnp.float32).reshape(train_images.shape[0], -1)
    X_test = jnp.asarray(test_images, dtype=jnp.float32).reshape(test_images.shape[0], -1)

    train_labels_int = jnp.asarray(train_labels, dtype=jnp.int32)
    test_labels_int = jnp.asarray(test_labels, dtype=jnp.int32)

    return X_train, train_labels_int, X_test, test_labels_int


def build_model(input_dim, hidden_width):
    output_dim = 2
    model = Linear(output_dim, hidden_width)
    model @= ReLU() @ Linear(hidden_width, hidden_width)
    model @= ReLU() @ Linear(hidden_width, hidden_width)
    model @= ReLU() @ Linear(hidden_width, hidden_width)

    model @= ReLU() @ Linear(hidden_width, input_dim)
    model.jit()
    return model


def sample_batch(key, batch_size, inputs, targets):
    if inputs.shape[0] == 0:
        raise ValueError("Cannot sample from an empty dataset")
    replace = inputs.shape[0] < batch_size
    idx = jax.random.choice(key, inputs.shape[0], shape=(batch_size,), replace=replace)
    return inputs[idx], targets[idx]


def mse_factory(model):
    def loss_fn(weights, batch_inputs, batch_targets):
        predictions = model(batch_inputs, weights)
        return jnp.mean((predictions - batch_targets) ** 2)

    return loss_fn


def compute_binary_accuracy(model, weights, inputs, binary_labels):
    logits = model(inputs, weights)
    predictions = jnp.argmax(logits, axis=1)
    return float(jnp.mean(predictions == binary_labels) * 100.0)


def sample_unique_class_pairs(key, num_tasks, num_classes=100):
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")

    max_disjoint_tasks = num_classes // 2
    if num_tasks <= max_disjoint_tasks:
        perm = jax.random.permutation(key, num_classes)
        selected = perm[: num_tasks * 2]
        pairs = [(int(selected[2 * i]), int(selected[2 * i + 1])) for i in range(num_tasks)]
        return pairs

    print(
        f"Warning: Requested {num_tasks} tasks, exceeding the disjoint class limit of {max_disjoint_tasks}."
        " Allowing class reuse while keeping class pairs unique."
    )

    pairs: List[Tuple[int, int]] = []
    seen_pairs = set()
    current_key = key

    while len(pairs) < num_tasks:
        current_key, pair_key = jax.random.split(current_key)
        classes = jax.random.choice(pair_key, num_classes, shape=(2,), replace=False)
        a, b = int(classes[0]), int(classes[1])
        if a == b:
            continue
        normalized = tuple(sorted((a, b)))
        if normalized in seen_pairs:
            continue
        seen_pairs.add(normalized)
        pairs.append((a, b))

    return pairs


def build_binary_task_datasets(
    X_train: jnp.ndarray,
    train_labels: jnp.ndarray,
    X_test: jnp.ndarray,
    test_labels: jnp.ndarray,
    class_pairs: Sequence[Tuple[int, int]],
) -> List[TaskData]:
    datasets: List[TaskData] = []
    for class_pair in class_pairs:
        a, b = class_pair
        train_mask = jnp.logical_or(train_labels == a, train_labels == b)
        test_mask = jnp.logical_or(test_labels == a, test_labels == b)

        train_inputs = X_train[train_mask]
        test_inputs = X_test[test_mask]

        train_selected_labels = train_labels[train_mask]
        test_selected_labels = test_labels[test_mask]

        train_binary = jnp.where(train_selected_labels == a, 0, 1).astype(jnp.int32)
        test_binary = jnp.where(test_selected_labels == a, 0, 1).astype(jnp.int32)

        train_targets = jax.nn.one_hot(train_binary, 2).astype(jnp.float32)

        datasets.append(
            TaskData(
                class_pair=class_pair,
                train_inputs=train_inputs,
                train_targets=train_targets,
                train_binary_labels=train_binary,
                test_inputs=test_inputs,
                test_binary_labels=test_binary,
            )
        )
    return datasets


def train_class_pair_sequence(
    model,
    base_key,
    method,
    learning_rate,
    steps_per_task,
    batch_size,
    target_norm,
    tasks: Sequence[TaskData],
):
    key_init, run_key = jax.random.split(base_key)
    weights = model.initialize(key_init)

    loss_fn = mse_factory(model)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    metrics: List[Dict[str, float]] = []

    for task_idx, task in enumerate(tasks):
        run_key, task_key = jax.random.split(run_key)
        dual_state = None
        if method == "manifold_online":
            dual_state = model.init_dual_state(weights)

        description = f"{method} task {task_idx + 1}/{len(tasks)} ({task.class_pair[0]} vs {task.class_pair[1]})"
        loss_value = jnp.nan
        for _ in trange(steps_per_task, leave=False, desc=description):
            task_key, batch_key = jax.random.split(task_key)
            batch_inputs, batch_targets = sample_batch(batch_key, batch_size, task.train_inputs, task.train_targets)
            loss_value, grad_weights = loss_and_grad(weights, batch_inputs, batch_targets)

            if method == "manifold_online":
                tangents, dual_state = model.online_dual_ascent(
                    dual_state,
                    weights,
                    grad_weights,
                    target_norm=target_norm,
                    alpha=2e-5,
                    beta=0.9,
                )
                weights = [w - learning_rate * t for w, t in zip(weights, tangents)]
                #weights = [matrix_sign(weight_matrix) for weight_matrix in weights]
                weights = model.retract(weights)

            elif method == "dualize":
                directions = model.dualize(grad_weights, target_norm=target_norm)
                weights = [w - learning_rate * direction for w, direction in zip(weights, directions)]

            elif method == "descent":
                weights = [w - learning_rate * grad for w, grad in zip(weights, grad_weights)]
            else:
                raise ValueError(f"Unknown training method: {method}")

        train_accuracy = compute_binary_accuracy(model, weights, task.train_inputs, task.train_binary_labels)
        test_accuracy = compute_binary_accuracy(model, weights, task.test_inputs, task.test_binary_labels)

        metrics.append(
            {
                "task_index": int(task_idx),
                "classes": [int(task.class_pair[0]), int(task.class_pair[1])],
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy),
                "final_loss": float(loss_value),
            }
        )

        print(
            f"[{method}] task {task_idx + 1}/{len(tasks)} | "
            f"classes=({task.class_pair[0]}, {task.class_pair[1]}) | "
            f"train acc={train_accuracy:.2f}% | test acc={test_accuracy:.2f}% | loss={float(loss_value):.4f}"
        )

    return weights, metrics


def plot_task_accuracies(results: Dict[str, List[Dict[str, float]]], plots_dir: Path) -> None:
    if not results:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_train, ax_test) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    color_map = plt.get_cmap("tab10")

    max_tasks = max((len(metrics) for metrics in results.values() if metrics), default=0)

    for idx, (method, metrics) in enumerate(results.items()):
        if not metrics:
            continue
        tasks = [metric["task_index"] + 1 for metric in metrics]
        train_accs = [metric["train_accuracy"] for metric in metrics]
        test_accs = [metric["test_accuracy"] for metric in metrics]
        color = color_map(idx % 10)
        ax_train.plot(tasks, train_accs, marker="o", color=color, label=method)
        ax_test.plot(tasks, test_accs, marker="o", color=color, label=method)

    if max_tasks:
        ax_test.set_xticks(range(1, max_tasks + 1))

    ax_test.set_xlabel("Task index")
    ax_train.set_ylabel("Train accuracy (%)")
    ax_test.set_ylabel("Test accuracy (%)")
    ax_train.set_title("Continual CIFAR-100 train accuracy per task")
    ax_test.set_title("Continual CIFAR-100 test accuracy per task")
    ax_train.grid(True, linestyle="--", alpha=0.3)
    ax_test.grid(True, linestyle="--", alpha=0.3)
    ax_train.legend()
    ax_test.legend()

    fig.tight_layout()
    fig.savefig(plots_dir / "continual_cifar100_accuracy.png", dpi=300)
    plt.close(fig)


def save_results(args, class_pairs, results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": {
            "num_tasks": int(args.num_tasks),
            "steps_per_task": int(args.steps_per_task),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "seed": int(args.seed),
            "pair_seed": int(args.pair_seed if args.pair_seed is not None else args.seed + 1),
            "target_norm": float(args.target_norm),
            "hidden_width": int(args.hidden_width),
            "methods": list(args.methods),
        },
        "class_pairs": [list(map(int, pair)) for pair in class_pairs],
        "methods": {},
    }

    for method, task_metrics in results.items():
        payload["methods"][method] = {"tasks": task_metrics}

    with output_path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Continual learning on binary CIFAR-100 tasks")
    parser.add_argument("--num-tasks", type=int, default=49, help="Number of sequential tasks (max 50)")
    parser.add_argument("--steps-per-task", type=int, default=1000, help="Training steps allocated to each task")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-2, help="Learning rate for all methods")
    parser.add_argument("--target-norm", type=float, default=1.0, help="Target norm for manifold-based updates")
    parser.add_argument("--hidden-width", type=int, default=128, help="Width of MLP hidden layers")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed for weight initialization and batches")
    parser.add_argument(
        "--pair-seed",
        type=int,
        default=None,
        help="Seed for sampling class pairs without replacement (defaults to seed+1)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["descent", "dualize", "manifold_online"],
        choices=METHOD_CHOICES,
        help="Training methods to evaluate",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("results/continual_cifar100_results.json"),
        help="Path to save summary metrics",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("plots") / "continual_cifar100",
        help="Directory to save accuracy plots",
    )
    return parser.parse_args()


def canonicalize_methods(methods: List[str]) -> List[str]:
    canonical = []
    for name in methods:
        key = name.lower()
        if key not in METHOD_ALIASES:
            raise ValueError(f"Unknown method: {name}")
        method = METHOD_ALIASES[key]
        if method not in canonical:
            canonical.append(method)
    return canonical


def main():
    args = parse_args()
    args.methods = canonicalize_methods(args.methods)

    X_train, train_labels, X_test, test_labels = prepare_data()

    input_dim = X_train.shape[1]
    model = build_model(input_dim, args.hidden_width)

    pair_seed = args.pair_seed if args.pair_seed is not None else args.seed + 1
    pair_key = jax.random.PRNGKey(pair_seed)
    class_pairs = sample_unique_class_pairs(pair_key, args.num_tasks, num_classes=100)

    task_datasets = build_binary_task_datasets(X_train, train_labels, X_test, test_labels, class_pairs)

    results: Dict[str, List[Dict[str, float]]] = {}

    base_key = jax.random.PRNGKey(args.seed)

    for method_idx, method in enumerate(args.methods):
        method_key = jax.random.fold_in(base_key, method_idx)
        _, metrics = train_class_pair_sequence(
            model,
            method_key,
            method,
            args.learning_rate,
            args.steps_per_task,
            args.batch_size,
            args.target_norm,
            task_datasets,
        )
        results[method] = metrics

    plot_task_accuracies(results, args.plots_dir)
    save_results(args, class_pairs, results, args.results_path)


if __name__ == "__main__":
    main()
