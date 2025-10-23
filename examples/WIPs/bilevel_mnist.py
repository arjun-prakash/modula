import jax
import jax.numpy as jnp

from data.mnist import load_mnist
from modula.atom import Linear
from modula.bond import ReLU


def prepare_data(num_validation: int = 10_000):
    train_images, train_labels, test_images, test_labels = load_mnist()

    train_images = train_images.reshape(train_images.shape[0], -1).astype(jnp.float32)
    test_images = test_images.reshape(test_images.shape[0], -1).astype(jnp.float32)

    train_labels = train_labels.astype(jnp.int32)
    test_labels = test_labels.astype(jnp.int32)

    x_train = jnp.array(train_images[num_validation:])
    y_train = jnp.array(train_labels[num_validation:])

    x_val = jnp.array(train_images[:num_validation])
    y_val = jnp.array(train_labels[:num_validation])

    x_test = jnp.array(test_images)
    y_test = jnp.array(test_labels)

    y_train_one_hot = jax.nn.one_hot(y_train, 10)
    y_val_one_hot = jax.nn.one_hot(y_val, 10)

    return (x_train, y_train, y_train_one_hot,
            x_val, y_val, y_val_one_hot,
            x_test, y_test)


def build_model(input_dim: int, width: int = 128):
    mlp = Linear(10, width)
    mlp @= ReLU() @ Linear(width, width)
    mlp @= ReLU() @ Linear(width, width)
    mlp @= ReLU() @ Linear(width, width)
    mlp @= ReLU() @ Linear(width, input_dim)
    mlp.jit()
    return mlp


def cross_entropy_loss(params, inputs, targets_one_hot, mlp):
    logits = mlp(inputs, params)
    log_probs = jax.nn.log_softmax(logits, axis=1)
    return -jnp.mean(jnp.sum(targets_one_hot * log_probs, axis=1))


def accuracy_from_logits(logits, labels):
    predictions = jnp.argmax(logits, axis=1)
    return jnp.mean(predictions == labels)


def main():
    (x_train, y_train, y_train_one_hot,
     x_val, y_val, y_val_one_hot,
     x_test, y_test) = prepare_data()

    input_dim = x_train.shape[1]
    train_size = x_train.shape[0]

    batch_size = 128
    inner_steps = 1000/50
    inner_lr = 0.001

    outer_steps = 50
    outer_lr = 0.5
    radius_bounds = (1e-3, 5.0)
    radius = jnp.array(1.0)

    mlp = build_model(input_dim)

    loss_and_grad = jax.jit(jax.value_and_grad(
        lambda params, inputs, targets: cross_entropy_loss(params, inputs, targets, mlp)))

    @jax.jit
    def train_inner(target_radius, init_params, key):
        def body(carry, _):
            params, rng = carry
            rng, sample_key = jax.random.split(rng)
            idx = jax.random.randint(sample_key, (batch_size,), 0, train_size)
            inputs = x_train[idx]
            targets = y_train_one_hot[idx]

            loss, grad_params = loss_and_grad(params, inputs, targets)
            scaled_grad = mlp.dualize(grad_params, target_norm=target_radius)
            params = jax.tree_util.tree_map(
                lambda w, dw: w - inner_lr * dw, params, scaled_grad)
            return (params, rng), loss

        (final_params, _), losses = jax.lax.scan(
            body, (init_params, key), xs=None, length=inner_steps)
        return final_params, losses[-1]

    def outer_objective(target_radius, init_params, key):
        final_params, train_loss = train_inner(target_radius, init_params, key)
        val_logits = mlp(x_val, final_params)
        val_log_probs = jax.nn.log_softmax(val_logits, axis=1)
        val_loss = -jnp.mean(jnp.sum(y_val_one_hot * val_log_probs, axis=1))
        val_acc = accuracy_from_logits(val_logits, y_val)
        aux = {
            "final_params": final_params,
            "train_loss": train_loss,
            "val_acc": val_acc,
        }
        return val_loss, aux

    outer_value_and_grad = jax.jit(jax.value_and_grad(outer_objective, argnums=0, has_aux=True))

    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    params = mlp.initialize(init_key)

    for outer_step in range(outer_steps):
        key, train_key = jax.random.split(key)

        (val_loss, aux), grad_radius = outer_value_and_grad(radius, params, train_key)
        radius = radius - outer_lr * grad_radius
        radius = jnp.clip(radius, *radius_bounds)

        params = jax.tree_util.tree_map(jax.lax.stop_gradient, aux["final_params"])
        train_loss = aux["train_loss"]
        val_acc = aux["val_acc"]

        print(
            f"Outer step {outer_step:02d}: "
            f"radius={float(radius):.4f}, "
            f"grad={float(grad_radius):.4f}, "
            f"train_loss={float(train_loss):.4f}, "
            f"val_loss={float(val_loss):.4f}, "
            f"val_acc={float(val_acc) * 100:.2f}%"
        )

    test_logits = mlp(x_test, params)
    test_acc = accuracy_from_logits(test_logits, y_test)
    print(f"Test accuracy after bilevel training: {float(test_acc) * 100:.2f}%")


if __name__ == "__main__":
    main()
