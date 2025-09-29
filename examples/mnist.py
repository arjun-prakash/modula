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


from modula.atom import Linear
from modula.bond import ReLU

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
width = 256

mlp = Linear(output_dim, width)
mlp @= ReLU() @ Linear(width, width)
mlp @= ReLU() @ Linear(width, width)
mlp @= ReLU() @ Linear(width, width)

mlp @= ReLU() @ Linear(width, input_dim)

print(mlp)

mlp.jit()


from tqdm import tqdm

def mse(w, inputs, targets):
    outputs = mlp(inputs, w)
    loss = ((outputs-targets) ** 2).mean()
    return loss

mse_and_grad = jax.jit(jax.value_and_grad(mse))

batch_size = 128
steps = 1000
learning_rate = 0.001


for i in [1.0, 1.25]:

    key = jax.random.PRNGKey(0)
    w = mlp.initialize(key)

    progress_bar = tqdm(range(steps), desc=f"Loss: {0:.4f}")
    for step in progress_bar:
        key = jax.random.PRNGKey(step)
        inputs, targets = get_batch(key, batch_size)

        loss, grad_w = mse_and_grad(w, inputs, targets)
        d_w = mlp.dualize(grad_w, target_norm=i)
        w = [weight - learning_rate * d_weight for weight, d_weight in zip(w, d_w)]
        progress_bar.set_description(f"Loss: {loss:.4f}")


    # Get predictions for test images
    X_test = test_images.reshape(test_images.shape[0], -1)
    test_outputs = mlp(X_test, w)
    predicted_labels = jnp.argmax(test_outputs, axis=1)

    # Calculate and print overall test accuracy
    total_correct = (predicted_labels == test_labels).sum()
    total_samples = len(test_labels)
    print(f"Overall test accuracy: {100 * total_correct/total_samples:.2f}%")