import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from data.mnist import load_mnist
from modula.atom import Linear, matrix_sign
from modula.bond import ReLU

METHOD_CHOICES = ("manifold", "manifold_admm", "manifold_online", "dualize", "descent")


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


def train_single_run(model, base_key, method, learning_rate, steps, batch_size, target_norm, inputs, targets):
    key_init, key_loop = jax.random.split(base_key)
    weights = model.initialize(key_init)

    loss_fn = mse_factory(model)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    dual_alpha = 2e-5
    dual_beta = 0.9
    dual_state = model.init_dual_state(weights) if method == "manifold_online" else None


    description = f"{method} lr={learning_rate:.3g}"
    for step in trange(steps, leave=False, desc=description):
        key_loop, batch_key = jax.random.split(key_loop)
        batch_inputs, batch_targets = sample_batch(batch_key, batch_size, inputs, targets)

        loss_value, grad_weights = loss_and_grad(weights, batch_inputs, batch_targets)

        #step_lr = learning_rate * (1.0 - step / steps)

        if method == "manifold":
            tangents = model.dual_ascent(weights, grad_weights, target_norm=target_norm)
            weights = [w - learning_rate * t for w, t in zip(weights, tangents)]
            weights = [matrix_sign(weight_matrix) for weight_matrix in weights]

        elif method == 'manifold_online':
            tangents, dual_state = model.online_dual_ascent(
                    dual_state,
                    weights,
                    grad_weights,
                    target_norm=1,
                    alpha=dual_alpha,
                    beta=dual_beta,
                )
            weights = [w - learning_rate * t for w, t in zip(weights, tangents)]
            weights = [matrix_sign(weight_matrix) for weight_matrix in weights]  # retraction

        elif method == "manifold_admm":
            tangents = model.admm_dual_ascent(weights, grad_weights, target_norm=target_norm)
            weights = [w - learning_rate * t for w, t in zip(weights, tangents)]
            weights = [matrix_sign(weight_matrix) for weight_matrix in weights]

        elif method == "dualize":
            directions = model.dualize(grad_weights, target_norm=target_norm)
            weights = [w - learning_rate * direction for w, direction in zip(weights, directions)]
        elif method == "descent":
            weights = [w - learning_rate * grad for w, grad in zip(weights, grad_weights)]
            #weights = model.project(weights)
        else:
            raise ValueError(f"Unknown training method: {method}")

    return weights, float(loss_value)


def singular_values_per_layer(weights):
    values = []
    for layer_weights in weights:
        singular_vals = jnp.linalg.svd(layer_weights, compute_uv=False)
        values.append(np.asarray(singular_vals))
    return values


def singular_values_combined(weights):
    layer_values = singular_values_per_layer(weights)
    if not layer_values:
        return np.array([])
    stacked = np.concatenate(layer_values)
    return np.sort(stacked)[::-1]


def plot_accuracy(results, plots_dir):
    if not results:
        return

    fig, (ax_train, ax_test) = plt.subplots(2, 1, sharex=True, figsize=(7, 8))
    color_map = plt.get_cmap("tab10")

    for idx, (method, runs) in enumerate(results.items()):
        if not runs:
            continue
        color = color_map(idx % 10)
        learning_rates = [entry["learning_rate"] for entry in runs]
        train_accs = [entry["train_accuracy"] for entry in runs]
        test_accs = [entry["test_accuracy"] for entry in runs]

        ax_train.plot(learning_rates, train_accs, marker="o", color=color, label=method)
        ax_test.plot(learning_rates, test_accs, marker="o", color=color, label=method)

    ax_test.set_xscale("log")
    ax_test.set_xlabel("Learning rate")

    ax_train.set_ylabel("Train accuracy (%)")
    ax_test.set_ylabel("Test accuracy (%)")

    ax_train.set_title("MNIST manifold GD sweep: train accuracy")
    ax_test.set_title("MNIST manifold GD sweep: test accuracy")

    ax_train.grid(True, linestyle="--", alpha=0.3)
    ax_test.grid(True, linestyle="--", alpha=0.3)

    ax_train.legend()
    ax_test.legend()

    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "mnist_accuracy_vs_lr.png", dpi=300)
    plt.close(fig)


def plot_singular_values(best_runs, plots_dir):
    if not best_runs:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    color_map = plt.get_cmap("tab10")

    for idx, (method, run) in enumerate(best_runs.items()):
        combined = singular_values_combined(run["weights"])
        if combined.size == 0:
            continue
        indices = np.arange(1, combined.size + 1)
        ax.plot(
            indices,
            combined,
            marker="o",
            color=color_map(idx % 10),
            label=f"{method} lr={run['learning_rate']:.3g} (test {run['test_accuracy']:.2f}%)",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Magnitude")
    ax.set_title("Best-run singular spectra by method")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "mnist_best_lr_singular_values.png", dpi=300)
    plt.close(fig)



def save_results(results, best_runs, args, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": {
            "learning_rates": [float(lr) for lr in args.learning_rates],
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "target_norm": float(args.target_norm),
            "hidden_width": int(args.hidden_width),
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

    with output_path.open("w") as fp:
        json.dump(payload, fp, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST manifold gradient descent sweep")
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e-1, 1e-2, 1e-3],
        help="Learning rates to sweep",
    )
    parser.add_argument("--steps", type=int, default=1000, help="Training steps per learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    parser.add_argument("--target-norm", type=float, default=1.0, help="Target norm for tangent updates")
    parser.add_argument("--hidden-width", type=int, default=32, help="Hidden layer width")
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
        default=Path("results/mnist_manifold_results.json"),
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

    X_train, y_train_one_hot, train_labels_int, X_test, test_labels_int = prepare_data()

    input_dim = X_train.shape[1]
    output_dim = y_train_one_hot.shape[1]

    model = build_model(input_dim, output_dim, args.hidden_width)

    results = {method: [] for method in args.methods}
    best_runs = {}

    base_key = jax.random.PRNGKey(args.seed)

    for method_idx, method in enumerate(args.methods):
        method_key = jax.random.fold_in(base_key, method_idx)

        for lr_idx, learning_rate in enumerate(args.learning_rates):
            run_key = jax.random.fold_in(method_key, lr_idx)
            weights, final_loss = train_single_run(
                model,
                run_key,
                method,
                learning_rate,
                args.steps,
                args.batch_size,
                args.target_norm,
                X_train,
                y_train_one_hot,
            )

            train_accuracy = compute_accuracy(model, weights, X_train, train_labels_int)
            test_accuracy = compute_accuracy(model, weights, X_test, test_labels_int)

            entry = {
                "learning_rate": learning_rate,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "final_loss": final_loss,
            }
            results[method].append(entry)

            best = best_runs.get(method)
            if best is None or test_accuracy > best["test_accuracy"]:
                best_runs[method] = {
                    "learning_rate": learning_rate,
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                    "final_loss": final_loss,
                    "weights": weights,
                }

            print(
                f"[{method}] lr={learning_rate:.3g}: train acc={train_accuracy:.2f}% | "
                f"test acc={test_accuracy:.2f}% | loss={final_loss:.4f}"
            )

    plot_accuracy(results, args.plots_dir)
    plot_singular_values(best_runs, args.plots_dir)

    save_results(results, best_runs, args, args.results_path)


if __name__ == "__main__":
    main()
