from data.mnist import load_mnist

# Load the MNIST dataset
train_images, train_labels, test_images, test_labels = load_mnist()

# Print shapes to verify loading
print(f"Training images shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")

import jax
import jax.numpy as jnp

def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

# Reshape images and convert labels
X_train = train_images.reshape(train_images.shape[0], -1)
y_train = one_hot(train_labels, 10)

# Get a batch
def get_batch(key, batch_size):
    idx = jax.random.choice(key, X_train.shape[1], shape=(batch_size,))
    return X_train[idx, :], y_train[idx, :]


from modula.atom import Linear, matrix_sign
from modula.bond import ReLU


input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
width = 256

mlp = Linear(output_dim, width)
mlp @= ReLU() @ Linear(width, width)
mlp @= ReLU() @ Linear(width, input_dim)

print(mlp)

mlp.jit()


from tqdm import tqdm

@jax.jit
def spec_norm(M, num_steps=20):
    v = jax.random.normal(jax.random.PRNGKey(0), (M.shape[1],))
    
    for _ in range(num_steps):
        u = M @ v
        u = u / jnp.linalg.norm(u)
        v = M.T @ u 
        v = v / jnp.linalg.norm(v)
    
    return jnp.linalg.norm(M @ v)

def mse(w, inputs, targets):
    outputs = mlp(inputs, w)
    loss = ((outputs-targets) ** 2).mean()
    return loss

mse_and_grad = jax.jit(jax.value_and_grad(mse))

batch_size = 128
steps = 1000
learning_rate = 0.1


methods = ['descent']
results = {}

for method in methods:
    print(f"\n=== Running method: {method} ===")
    results[method] = {}

    for i in [1.0]:

        key = jax.random.PRNGKey(0)
        w = mlp.initialize(key)

        progress_bar = tqdm(range(steps), desc=f"{method} Loss: {0:.4f}")
        for step in progress_bar:
            key = jax.random.PRNGKey(step)
            inputs, targets = get_batch(key, batch_size)

            loss, grad_w = mse_and_grad(w, inputs, targets)

            if method == "dualize":
                d_w = mlp.dualize(grad_w, target_norm=i)
                w = [weight - learning_rate * d_weight for weight, d_weight in zip(w, d_w)]
            elif method == "descent":
                #d_w = [g / spec_norm(g) / 3 * jnp.sqrt(g.shape[0]/g.shape[1]) for g in grad_w]
                w = [weight - learning_rate * grad for weight, grad in zip(w, grad_w)]
                w = mlp.project(w)
            elif method == "manifold":
                tangents = mlp.dual_ascent(w, grad_w, target_norm=i)
                w = [weight - learning_rate * dt for weight, dt in zip(w, tangents)]
                w = [matrix_sign(weight) for weight in w]  # retraction

            else:
                raise ValueError(f"Unknown training method: {method}")

            progress_bar.set_description(f"{method} Loss: {loss:.4f}")

        # Evaluate on the test set
        X_test = test_images.reshape(test_images.shape[0], -1)
        test_outputs = mlp(X_test, w)
        predicted_labels = jnp.argmax(test_outputs, axis=1)

        total_correct = (predicted_labels == test_labels).sum()
        total_samples = len(test_labels)
        accuracy = 100 * total_correct / total_samples
        results[method][i] = float(accuracy)
        print(f"[{method}] target_norm={i}: accuracy {accuracy:.2f}%")

print("\nSummary of accuracies:")
for method, scores in results.items():
    for target_norm, accuracy in scores.items():
        print(f"  {method} (target_norm={target_norm}): {accuracy:.2f}%")