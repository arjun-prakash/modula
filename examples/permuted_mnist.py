import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from tqdm import trange

from data.mnist import load_mnist
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


def prepare_data():
    train_images, train_labels, test_images, test_labels = load_mnist()

    X_train = jnp.asarray(train_images, dtype=jnp.float32).reshape(train_images.shape[0], -1)
    X_test = jnp.asarray(test_images, dtype=jnp.float32).reshape(test_images.shape[0], -1)

    y_train_one_hot = jax.nn.one_hot(jnp.asarray(train_labels, dtype=jnp.int32), 10).astype(jnp.float32)
    train_labels_int = jnp.asarray(train_labels, dtype=jnp.int32)
    test_labels_int = jnp.asarray(test_labels, dtype=jnp.int32)

    return X_train, y_train_one_hot, train_labels_int, X_test, test_labels_int


def build_model(input_dim, output_dim, width):
    model = Linear(output_dim, width)
    model @= ReLU() @ Linear(width, width)
    model @= ReLU() @ Linear(width, input_dim)
    model.jit()
    return model


def sample_batch(key, batch_size, inputs, targets):
    idx = jax.random.choice(key, inputs.shape[0], shape=(batch_size,), replace=False)
    return inputs[idx], targets[idx]


def mse_factory(model):
    def loss_fn(weights, batch_inputs, batch_targets):
        predictions = model(batch_inputs, weights)
        return jnp.mean((predictions - batch_targets) ** 2)

    return loss_fn


def compute_accuracy(model, weights, inputs, labels):
    logits = model(inputs, weights)
    predictions = jnp.argmax(logits, axis=1)
    return float(jnp.mean(predictions == labels) * 100.0)


def generate_task_permutations(key, num_tasks, input_dim, include_identity=True):
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")

    permutations: List[jnp.ndarray] = []
    remaining = num_tasks

    if include_identity:
        permutations.append(jnp.arange(input_dim))
        remaining -= 1

    current_key = key
    for _ in range(remaining):
        current_key, perm_key = jax.random.split(current_key)
        permutations.append(jax.random.permutation(perm_key, input_dim))

    return permutations


def train_permuted_sequence(
    model,
    base_key,
    method,
    learning_rate,
    steps_per_task,
    batch_size,
    target_norm,
    permutations,
    X_train,
    y_train_one_hot,
    train_labels_int,
    X_test,
    test_labels_int,
):
    key_init, run_key = jax.random.split(base_key)
    weights = model.initialize(key_init)

    loss_fn = mse_factory(model)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    metrics: List[Dict[str, float]] = []

    for task_idx, permutation in enumerate(permutations):
        run_key, task_key = jax.random.split(run_key)

        perm_train_inputs = X_train[:, permutation]
        perm_test_inputs = X_test[:, permutation]

        dual_state = None
        if method == "manifold_online":
            dual_state = model.init_dual_state(weights)

        description = f"{method} task {task_idx + 1}/{len(permutations)}"
        loss_value = jnp.nan
        for _ in trange(steps_per_task, leave=False, desc=description):
            task_key, batch_key = jax.random.split(task_key)
            batch_inputs, batch_targets = sample_batch(batch_key, batch_size, perm_train_inputs, y_train_one_hot)
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

        train_accuracy = compute_accuracy(model, weights, perm_train_inputs, train_labels_int)
        test_accuracy = compute_accuracy(model, weights, perm_test_inputs, test_labels_int)

        metrics.append(
            {
                "task_index": int(task_idx),
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy),
                "final_loss": float(loss_value),
            }
        )

        print(
            f"[{method}] task {task_idx + 1}/{len(permutations)} | "
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
    ax_train.set_title("Permuted MNIST train accuracy per task")
    ax_test.set_title("Permuted MNIST test accuracy per task")
    ax_train.grid(True, linestyle="--", alpha=0.3)
    ax_test.grid(True, linestyle="--", alpha=0.3)
    ax_train.legend()
    ax_test.legend()

    fig.tight_layout()
    fig.savefig(plots_dir / "permuted_mnist_accuracy.png", dpi=300)
    plt.close(fig)


def save_results(args, permutations, results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": {
            "num_tasks": int(args.num_tasks),
            "steps_per_task": int(args.steps_per_task),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "seed": int(args.seed),
            "permutation_seed": int(args.permutation_seed if args.permutation_seed is not None else args.seed + 1),
            "target_norm": float(args.target_norm),
            "hidden_width": int(args.hidden_width),
            "methods": list(args.methods),
            "include_identity": bool(args.include_identity),
        },
        "permutations": [list(map(int, permutation.tolist())) for permutation in permutations],
        "methods": {},
    }

    for method, task_metrics in results.items():
        payload["methods"][method] = {"tasks": task_metrics}

    with output_path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Continual learning on permuted MNIST")
    parser.add_argument("--num-tasks", type=int, default=100, help="Number of sequential tasks (including identity if enabled)")
    parser.add_argument("--steps-per-task", type=int, default=1000, help="Training steps allocated to each task")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-1, help="Learning rate for all methods")
    parser.add_argument("--target-norm", type=float, default=1.0, help="Target norm for manifold-based updates")
    parser.add_argument("--hidden-width", type=int, default=32, help="Width of MLP hidden layers")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed for weight initialization and batches")
    parser.add_argument(
        "--permutation-seed",
        type=int,
        default=None,
        help="Seed for sampling pixel permutations (defaults to seed+1)",
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
        "--include-identity",
        dest="include_identity",
        action="store_true",
        help="Include the original MNIST task before permuted tasks (default)",
    )
    parser.add_argument(
        "--no-identity",
        dest="include_identity",
        action="store_false",
        help="Skip the original MNIST task",
    )
    parser.set_defaults(include_identity=True)
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("results/permuted_mnist_results.json"),
        help="Path to save summary metrics",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("plots") / "permuted_mnist",
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

    X_train, y_train_one_hot, train_labels_int, X_test, test_labels_int = prepare_data()

    input_dim = X_train.shape[1]
    output_dim = y_train_one_hot.shape[1]

    model = build_model(input_dim, output_dim, args.hidden_width)

    permutation_seed = args.permutation_seed if args.permutation_seed is not None else args.seed + 1
    permutation_key = jax.random.PRNGKey(permutation_seed)
    permutations = generate_task_permutations(permutation_key, args.num_tasks, input_dim, include_identity=args.include_identity)

    results: Dict[str, List[Dict[str, float]]] = {}

    base_key = jax.random.PRNGKey(args.seed)

    for method_idx, method in enumerate(args.methods):
        method_key = jax.random.fold_in(base_key, method_idx)
        _, metrics = train_permuted_sequence(
            model,
            method_key,
            method,
            args.learning_rate,
            args.steps_per_task,
            args.batch_size,
            args.target_norm,
            permutations,
            X_train,
            y_train_one_hot,
            train_labels_int,
            X_test,
            test_labels_int,
        )
        results[method] = metrics

    plot_task_accuracies(results, args.plots_dir)
    save_results(args, permutations, results, args.results_path)


if __name__ == "__main__":
    main()
