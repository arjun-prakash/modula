import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import tqdm

from data.cifar10 import load_cifar10
from modula.atom import Conv2D, Linear
from modula.bond import AvgPool2D, Flatten, ReLU, MaxPool2D

METHOD_CHOICES = ("manifold_online", "dualize", "descent", "adam")


def one_hot(labels, num_classes, dtype=jnp.float32):
    """Create one-hot encoding of labels."""
    return jnp.array(labels[:, None] == jnp.arange(num_classes), dtype)


def prepare_cifar10(normalize=True):
    """Load and preprocess CIFAR-10 data."""
    print("Loading CIFAR-10...")
    train_images, train_labels, test_images, test_labels = load_cifar10(normalize=normalize)

    X_train = jnp.asarray(train_images, dtype=jnp.float32)
    y_train = one_hot(train_labels, 10)
    X_test = jnp.asarray(test_images, dtype=jnp.float32)
    y_test = one_hot(test_labels, 10)

    print(f"CIFAR-10 prepared - X_train: {X_train.shape}, y_train: {y_train.shape}")
    return X_train, y_train, X_test, y_test


def get_batch(key, X, y, batch_size):
    """Sample a batch with proper handling for small datasets."""
    if X.shape[0] == 0:
        raise ValueError("Cannot sample from an empty dataset")
    replace = X.shape[0] < batch_size
    idx = jax.random.choice(key, X.shape[0], shape=(batch_size,), replace=replace)
    return X[idx], y[idx]


# def build_avgpool_cnn(num_classes=10):
#     """CNN with average pooling instead of max pooling."""
#     cnn = Linear(num_classes, 4 * 4 * 128)
#     cnn @= Flatten()
#     cnn @= AvgPool2D(pool_size=2)
#     cnn @= ReLU() @ Conv2D(64, 128, kernel_size=3)
#     cnn @= AvgPool2D(pool_size=2)
#     cnn @= ReLU() @ Conv2D(32, 64, kernel_size=3)
#     cnn @= AvgPool2D(pool_size=2)
#     cnn @= ReLU() @ Conv2D(3, 32, kernel_size=3)
#     cnn.jit()
#     return cnn

def build_maxpool_cnn(num_classes=10):
    """CNN with max pooling."""
    cnn = Linear(num_classes, 4 * 4 * 128)
    cnn @= Flatten()
    cnn @= MaxPool2D(pool_size=2)
    cnn @= ReLU() @ Conv2D(64, 128, kernel_size=3)
    cnn @= MaxPool2D(pool_size=2)
    cnn @= ReLU() @ Conv2D(32, 64, kernel_size=3)
    cnn @= MaxPool2D(pool_size=2)
    cnn @= ReLU() @ Conv2D(3, 32, kernel_size=3)
    cnn.jit()
    return cnn


def mse_loss(weights, model, inputs, targets):
    """Mean squared error loss."""
    outputs = model(inputs, weights)
    return jnp.mean((outputs - targets) ** 2)


def compute_accuracy(model, weights, inputs, targets):
    """Compute classification accuracy using one-hot targets."""
    logits = model(inputs, weights)
    predictions = jnp.argmax(logits, axis=1)
    labels = jnp.argmax(targets, axis=1)
    return float(jnp.mean(predictions == labels) * 100.0)


def _unwrap(weight):
    return weight[0] if isinstance(weight, (list, tuple)) else weight


def _svdvals(matrix):
    return jnp.linalg.svd(matrix.astype(jnp.float32), compute_uv=False, full_matrices=False)


def _conv_svdvals(weights):
    kernel = jnp.asarray(weights)
    matrix = kernel.reshape(-1, kernel.shape[-1])
    return _svdvals(matrix)


def singular_values_per_layer(weights):
    """Return singular values per parameter tensor."""
    values = []
    for weight in weights:
        tensor = _unwrap(weight)
        if not hasattr(tensor, "ndim"):
            continue
        if tensor.ndim == 2:
            values.append(_svdvals(tensor))
        elif tensor.ndim >= 4:
            values.append(_conv_svdvals(tensor))
        else:
            matrix = tensor.reshape(-1, tensor.shape[-1])
            values.append(_svdvals(matrix))
    return values


def singular_values_combined(weights):
    layer_values = singular_values_per_layer(weights)
    if not layer_values:
        return np.array([])
    stacked = jnp.concatenate(layer_values, axis=0)
    return np.asarray(jnp.sort(stacked)[::-1])


def plot_accuracy(results, plots_dir):
    """Plot train/test accuracy versus learning rate for each method."""
    if not results:
        return

    plots_path = Path(plots_dir)
    fig, (ax_train, ax_test) = plt.subplots(2, 1, sharex=True, figsize=(7, 8))
    color_map = plt.get_cmap("tab10")

    for idx, (method, runs) in enumerate(results.items()):
        if not runs:
            continue
        runs_sorted = sorted(runs, key=lambda entry: entry["learning_rate"])
        color = color_map(idx % 10)
        learning_rates = [entry["learning_rate"] for entry in runs_sorted]
        train_accs = [entry["train_accuracy"] for entry in runs_sorted]
        test_accs = [entry["test_accuracy"] for entry in runs_sorted]

        ax_train.plot(learning_rates, train_accs, marker="o", color=color, label=method)
        ax_test.plot(learning_rates, test_accs, marker="o", color=color, label=method)

    ax_test.set_xscale("log")
    ax_test.set_xlabel("Learning rate")

    ax_train.set_ylabel("Train accuracy (%)")
    ax_test.set_ylabel("Test accuracy (%)")

    ax_train.set_title("CIFAR-10 manifold GD sweep: train accuracy")
    ax_test.set_title("CIFAR-10 manifold GD sweep: test accuracy")

    ax_train.grid(True, linestyle="--", alpha=0.3)
    ax_test.grid(True, linestyle="--", alpha=0.3)

    ax_train.legend()
    ax_test.legend()

    fig.tight_layout()
    plots_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_path / "cifar10_accuracy_vs_lr.png", dpi=300)
    plt.close(fig)


def plot_singular_values(best_runs, plots_dir):
    """Plot singular value spectra for the best run of each method."""
    if not best_runs:
        return

    plots_path = Path(plots_dir)
    fig, ax = plt.subplots(figsize=(7, 5))
    color_map = plt.get_cmap("tab10")

    for idx, (method, run) in enumerate(best_runs.items()):
        combined = singular_values_combined(run["weights"])
        if combined.size == 0:
            continue
        indices = np.arange(1, combined.size + 1)
        label = f"{method} lr={run['learning_rate']:.3g} (test {run['test_accuracy']:.2f}%)"
        ax.plot(indices, combined, marker="o", color=color_map(idx % 10), label=label)

    ax.set_yscale("log")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Magnitude")
    ax.set_title("Best-run singular spectra by method")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    plots_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_path / "cifar10_best_lr_singular_values.png", dpi=300)
    plt.close(fig)


def train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    batch_size=128,
    steps=2000,
    learning_rate=0.05,
    eval_every=100,
    eval_subset_size=1000,
    seed=0,
    method="dualize",
    target_norm=1.0,
    dual_alpha=2e-8,
    dual_beta=0.95,
):
    """Train a model and track metrics."""
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    weights = model.initialize(init_key)
    dual_state = model.init_dual_state(weights) if method == "manifold_online" else None
    optimizer = None
    opt_state = None
    if method == "adam":
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(weights)

    mse_and_grad = jax.jit(jax.value_and_grad(lambda w, x, y: mse_loss(w, model, x, y)))

    train_losses = []
    train_accs = []
    test_accs = []
    step_indices = []
    start_time = time.time()

    progress_prefix = f"{method} lr={learning_rate:.3g}"
    progress_bar = tqdm(range(steps), desc=f"{progress_prefix} | loss={0:.4f}", leave=False)

    for step in progress_bar:
        key, batch_key = jax.random.split(key)
        batch_inputs, batch_targets = get_batch(batch_key, X_train, y_train, batch_size)

        loss, grad_weights = mse_and_grad(weights, batch_inputs, batch_targets)

        if method == "dualize":
            directions = model.dualize(grad_weights, target_norm=target_norm)
            weights = [w - learning_rate * direction for w, direction in zip(weights, directions)]
        elif method == "descent":
            weights = [w - learning_rate * grad for w, grad in zip(weights, grad_weights)]
        elif method == "manifold_online":
            tangents, dual_state = model.online_dual_ascent(
                dual_state,
                weights,
                grad_weights,
                target_norm=target_norm,
                alpha=dual_alpha,
                beta=dual_beta,
            )
            weights = [w - learning_rate * tangent for w, tangent in zip(weights, tangents)]
            weights = model.retract(weights)
        elif method == "adam":
            updates, opt_state = optimizer.update(grad_weights, opt_state, params=weights)
            weights = optax.apply_updates(weights, updates)
        else:
            raise ValueError(f"Unknown method: {method}. Use {METHOD_CHOICES}.")

        if step % eval_every == 0 or step == steps - 1:
            if eval_subset_size and X_train.shape[0] > eval_subset_size:
                eval_inputs = X_train[:eval_subset_size]
                eval_targets = y_train[:eval_subset_size]
            else:
                eval_inputs = X_train
                eval_targets = y_train

            train_acc = compute_accuracy(model, weights, eval_inputs, eval_targets)
            test_acc = compute_accuracy(model, weights, X_test, y_test)

            train_losses.append(float(loss))
            train_accs.append(float(train_acc))
            test_accs.append(float(test_acc))
            step_indices.append(step)

            progress_bar.set_description(
                f"{progress_prefix} | loss={loss:.4f} | train={train_acc:.2f}% | test={test_acc:.2f}%"
            )

    elapsed_time = time.time() - start_time
    final_test_acc = compute_accuracy(model, weights, X_test, y_test)

    return {
        "model": model,
        "weights": weights,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_accs": test_accs,
        "step_indices": step_indices,
        "final_test_acc": float(final_test_acc),
        "final_loss": float(train_losses[-1]) if train_losses else float(loss),
        "training_time": elapsed_time,
        "method": method,
    }


def save_results(results, best_runs, args):
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "learning_rates": [float(lr) for lr in args.learning_rates],
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "target_norm": float(args.target_norm),
            "dual_alpha": float(args.dual_alpha),
            "dual_beta": float(args.dual_beta),
            "eval_every": int(args.eval_every),
            "eval_train_samples": int(args.eval_train_samples),
            "methods": list(args.methods),
        },
        "methods": {},
    }

    for method, runs in results.items():
        payload["methods"][method] = {
            "runs": [
                {
                    "learning_rate": float(entry["learning_rate"]),
                    "train_accuracy": float(entry["train_accuracy"]),
                    "test_accuracy": float(entry["test_accuracy"]),
                    "final_loss": float(entry["final_loss"]),
                }
                for entry in runs
            ]
        }
        best = best_runs.get(method)
        if best is not None:
            payload["methods"][method]["best"] = {
                "learning_rate": float(best["learning_rate"]),
                "train_accuracy": float(best["train_accuracy"]),
                "test_accuracy": float(best["test_accuracy"]),
                "final_loss": float(best["final_loss"]),
            }

    with args.results_path.open("w") as fp:
        json.dump(payload, fp, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 manifold gradient descent sweep")
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[10, 1, 1e-1,5e-2, 1e-2,5e-3, 1e-3, 5e-4, 1e-4],
        help="Learning rates to sweep",
    )
    parser.add_argument("--steps", type=int, default=2000, help="Training steps per learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--eval-every", type=int, default=100, help="Evaluation interval (steps)")
    parser.add_argument("--eval-train-samples", type=int, default=1000, help="Number of train samples used for periodic eval (0 for full set)")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    parser.add_argument("--target-norm", type=float, default=1.0, help="Target norm for tangent updates")
    parser.add_argument("--dual-alpha", type=float, default=2e-8, help="Dual ascent alpha for manifold_online")
    parser.add_argument("--dual-beta", type=float, default=0.95, help="Dual ascent beta for manifold_online")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=list(METHOD_CHOICES),
        choices=METHOD_CHOICES,
        help="Training methods to compare",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("results/cifar10_manifold_results.json"),
        help="Path to save sweep metrics",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("plots"),
        help="Directory for plot outputs",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    X_train, y_train, X_test, y_test = prepare_cifar10()
    model = build_maxpool_cnn(num_classes=10)

    results = {method: [] for method in args.methods}
    best_runs = {}

    base_key = jax.random.PRNGKey(args.seed)

    for method_idx, method in enumerate(args.methods):
        method_key = jax.random.fold_in(base_key, method_idx)
        for lr_idx, learning_rate in enumerate(args.learning_rates):
            run_key = jax.random.fold_in(method_key, lr_idx)
            run_seed = int(jax.random.randint(run_key, (), 0, np.iinfo(np.int32).max))

            run_result = train_model(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                batch_size=args.batch_size,
                steps=args.steps,
                learning_rate=learning_rate,
                eval_every=args.eval_every,
                eval_subset_size=args.eval_train_samples,
                seed=run_seed,
                method=method,
                target_norm=args.target_norm,
                dual_alpha=args.dual_alpha,
                dual_beta=args.dual_beta,
            )

            train_accuracy = compute_accuracy(model, run_result["weights"], X_train, y_train)
            test_accuracy = compute_accuracy(model, run_result["weights"], X_test, y_test)

            entry = {
                "learning_rate": learning_rate,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "final_loss": run_result["final_loss"],
            }
            results[method].append(entry)

            best = best_runs.get(method)
            if best is None or test_accuracy > best["test_accuracy"]:
                best_runs[method] = {
                    **entry,
                    "weights": run_result["weights"],
                }

            print(
                f"[{method}] lr={learning_rate:.3g}: train acc={train_accuracy:.2f}% | "
                f"test acc={test_accuracy:.2f}% | loss={run_result['final_loss']:.4f}"
            )

    plot_accuracy(results, args.plots_dir)
    plot_singular_values(best_runs, args.plots_dir)
    save_results(results, best_runs, args)


if __name__ == "__main__":
    main()
