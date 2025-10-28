import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from data.cifar10 import load_cifar10
from modula.atom import Conv2D, Linear
from modula.bond import ReLU, Flatten, AvgPool2D

# Load CIFAR-10
print("Loading CIFAR-10...")
train_images_10, train_labels_10, test_images_10, test_labels_10 = load_cifar10(normalize=True)

def one_hot(labels, num_classes, dtype=jnp.float32):
    """Create one-hot encoding of labels."""
    return jnp.array(labels[:, None] == jnp.arange(num_classes), dtype)

# Prepare CIFAR-10 data
X_train_10 = jnp.asarray(train_images_10, dtype=jnp.float32)
y_train_10 = one_hot(train_labels_10, 10)
X_test_10 = jnp.asarray(test_images_10, dtype=jnp.float32)
y_test_10 = one_hot(test_labels_10, 10)


print(f"CIFAR-10 prepared - X_train: {X_train_10.shape}, y_train: {y_train_10.shape}")

# Batch sampler with proper replace handling
def get_batch(key, X, y, batch_size):
    """Sample a batch with proper handling for small datasets."""
    if X.shape[0] == 0:
        raise ValueError("Cannot sample from an empty dataset")
    # Use replace=True if dataset is smaller than batch size
    replace = X.shape[0] < batch_size
    idx = jax.random.choice(key, X.shape[0], shape=(batch_size,), replace=replace)
    return X[idx], y[idx]

def build_avgpool_cnn(num_classes=10):
    """CNN with average pooling instead of max pooling"""
    # Same structure as medium CNN but with AvgPool2D
    
    cnn = Linear(num_classes, 4*4*128)
    cnn @= Flatten()
    cnn @= AvgPool2D(pool_size=2)
    cnn @= ReLU() @ Conv2D(64, 128, kernel_size=3)
    cnn @= AvgPool2D(pool_size=2)
    cnn @= ReLU() @ Conv2D(32, 64, kernel_size=3)
    cnn @= AvgPool2D(pool_size=2)
    cnn @= ReLU() @ Conv2D(3, 32, kernel_size=3)
    
    cnn.jit()
    return cnn

test_cnn = build_avgpool_cnn(10)
print("Avgpool CNN:")
print(test_cnn)


def mse_loss(w, model, inputs, targets):
    """Mean squared error loss."""
    outputs = model(inputs, w)
    return jnp.mean((outputs - targets) ** 2)

def compute_accuracy(model, w, X, y_true):
    """Compute classification accuracy."""
    predictions = model(X, w)
    pred_labels = jnp.argmax(predictions, axis=1)
    true_labels = jnp.argmax(y_true, axis=1)
    return jnp.mean(pred_labels == true_labels) * 100


def _weight_to_matrix(weight_array):
    """Convert a weight tensor to a 2D matrix for SVD."""
    array = jnp.asarray(weight_array)
    if array.ndim < 2:
        return None
    if array.ndim == 2:
        return array
    return array.reshape(-1, array.shape[-1])


import jax
import jax.numpy as jnp
import numpy as np

def _unwrap(weight):
    return weight[0] if isinstance(weight, (list, tuple)) else weight

def _svdvals(M):
    # stable, no U/V
    return jnp.linalg.svd(M.astype(jnp.float32), compute_uv=False, full_matrices=False)

def _conv_svdvals_per_slice(W):
    # W: [k, k, d_in, d_out]; return all σ across slices flattened
    k, _, d_in, d_out = W.shape
    sv_fn = jax.vmap(jax.vmap(lambda w: _svdvals(w), in_axes=0), in_axes=0)  # over k x k
    sv = sv_fn(W)  # [k, k, min(d_in, d_out)]
    return sv.reshape(-1)

def singular_values_per_layer(weights):
    """Return singular values per parameter tensor (Linear: matrix SVD;
       Conv2D: per-slice SVD on [d_in × d_out] slices)."""
    values = []
    for w in weights:
        W = _unwrap(w)
        if not hasattr(W, "ndim"):  # skip non-arrays
            continue
        if W.ndim == 2:
            # Linear weight
            sv = _svdvals(W)
            values.append(sv)
        elif W.ndim == 4:
            # Conv kernel [k,k,d_in,d_out]
            sv = _conv_svdvals_per_slice(W)
            values.append(sv)
        elif W.ndim == 5 and W.shape[0] == 1:
            # Some wrappers store conv as [1,k,k,d_in,d_out]; squeeze then handle
            sv = _conv_svdvals_per_slice(jnp.squeeze(W, axis=0))
            values.append(sv)
        else:
            # Fallback: flatten last dim as cols (your old behavior)
            mat = W.reshape(-1, W.shape[-1])
            sv = _svdvals(mat)
            values.append(sv)
    return values

def singular_values_combined(weights):
    layer_values = singular_values_per_layer(weights)
    if not layer_values:
        return np.array([])
    stacked = jnp.concatenate(layer_values, axis=0)
    # convert to numpy only for plotting
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

def train_model(model, X_train, y_train, X_test, y_test,
                batch_size=128, steps=2000, learning_rate=0.05,
                eval_every=100, seed=0, method='dualize', target_norm=1.0,
                dual_alpha=2e-8, dual_beta=0.95):
    """Train a model and track metrics.
    
    Args:
        model: The neural network model to train
        X_train, y_train: Training data and labels
        X_test, y_test: Test data and labels
        batch_size: Batch size for training
        steps: Number of training steps
        learning_rate: Learning rate for weight updates
        eval_every: Evaluate metrics every N steps
        seed: Random seed
        method: Weight update method - 'descent', 'dualize', or 'manifold_online'
        target_norm: Target norm for dualize/manifold transforms (default: 1.0)
        dual_alpha: Alpha coefficient for manifold_online dual ascent
        dual_beta: Beta coefficient for manifold_online dual ascent
    """
    
    # Initialize weights
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    w = model.initialize(init_key)
    dual_state = model.init_dual_state(w) if method == 'manifold_online' else None
    
    # JIT compile loss and gradient
    mse_and_grad = jax.jit(jax.value_and_grad(lambda w, x, y: mse_loss(w, model, x, y)))
    
    # Training metrics
    train_losses = []
    train_accs = []
    test_accs = []
    step_indices = []
    
    start_time = time.time()
    
    progress_bar = tqdm(range(steps), desc=f"Loss: {0:.4f}, Train Acc: {0:.2f}%")
    
    for step in progress_bar:
        # Sample batch
        key, batch_key = jax.random.split(key)
        inputs, targets = get_batch(batch_key, X_train, y_train, batch_size)
        
        # Compute loss and gradients
        loss, grad_w = mse_and_grad(w, inputs, targets)
        
        # Update weights based on method
        if method == 'dualize':
            d_w = model.dualize(grad_w, target_norm=target_norm)
            w = [weight - learning_rate * d_weight for weight, d_weight in zip(w, d_w)]
        elif method == 'descent':
            w = [weight - learning_rate * grad for weight, grad in zip(w, grad_w)]
        elif method == 'manifold_online':
            tangents, dual_state = model.online_dual_ascent(
                dual_state,
                w,
                grad_w,
                target_norm=target_norm,
                alpha=dual_alpha,
                beta=dual_beta,
            )
            w = [weight - learning_rate * tangent for weight, tangent in zip(w, tangents)]
            w = model.retract(w)  # retract to maintain orthogonality constraints
        else:
            raise ValueError(f"Unknown method: {method}. Use 'descent', 'dualize', or 'manifold_online'")
        
        # Evaluate periodically
        if step % eval_every == 0 or step == steps - 1:
            train_acc = compute_accuracy(model, w, X_train[:1000], y_train[:1000])
            test_acc = compute_accuracy(model, w, X_test, y_test)
            
            train_losses.append(float(loss))
            train_accs.append(float(train_acc))
            test_accs.append(float(test_acc))
            step_indices.append(step)
            
            progress_bar.set_description(
                f"Loss: {loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%"
            )
    
    elapsed_time = time.time() - start_time
    final_test_acc = compute_accuracy(model, w, X_test, y_test)
    
    print(f"Training completed in {elapsed_time:.2f}s")
    print(f"Final test accuracy: {final_test_acc:.2f}%")
    
    return {
        'model': model,
        'weights': w,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'step_indices': step_indices,
        'final_test_acc': float(final_test_acc),
        'training_time': elapsed_time,
        'method': method
    }


print("=" * 60)
print("Training AvgPool CNN on CIFAR-10")
print("=" * 60)

avgpool_cnn_10 = build_avgpool_cnn(num_classes=10)

batch_size = 128
steps = 2000
learning_rate = 0.005
eval_every = 100
seed = 42
methods = ["manifold_online", "descent", "dualize"]

plots_dir = Path("plots")
results = {method: [] for method in methods}
best_runs = {}

for method in methods:
    run_result = train_model(
        avgpool_cnn_10,
        X_train_10,
        y_train_10,
        X_test_10,
        y_test_10,
        batch_size=batch_size,
        steps=steps,
        learning_rate=learning_rate,
        eval_every=eval_every,
        seed=seed,
        method=method,
    )

    train_accuracy = float(run_result["train_accs"][-1]) if run_result["train_accs"] else float("nan")
    test_accuracy = float(run_result["final_test_acc"])
    final_loss = float(run_result["train_losses"][-1]) if run_result["train_losses"] else float("nan")

    entry = {
        "learning_rate": learning_rate,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "final_loss": final_loss,
    }
    results[method].append(entry)

    best = best_runs.get(method)
    if best is None or test_accuracy > best["test_accuracy"]:
        best_runs[method] = {**entry, "weights": run_result["weights"]}

    # Preserve legacy variable for the last run, matching prior behavior.
    results_avgpool_10 = run_result


plot_accuracy(results, plots_dir)
plot_singular_values(best_runs, plots_dir)
