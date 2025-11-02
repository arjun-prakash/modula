import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from data.cifar100 import load_cifar100
from modula.abstract import Bond
from modula.atom import Linear, Conv2D, dampen_dual_state
from modula.bond import ReLU, Flatten, MaxPool2D

METHOD_CHOICES = ("manifold", "manifold_online", "dualize", "descent")


def format_method_label(generator_method: str, discriminator_method: str) -> str:
    if generator_method == discriminator_method:
        return generator_method
    return f"G:{generator_method}|D:{discriminator_method}"


class Reshape(Bond):
    """Bond to reshape flat vectors back into image grids."""

    def __init__(self, target_shape: Tuple[int, ...]):
        super().__init__()
        self.target_shape = target_shape
        self.smooth = True
        self.sensitivity = 1

    def forward(self, x, w):
        batch = x.shape[0]
        return jnp.reshape(x, (batch, *self.target_shape))


class Tanh(Bond):
    """Elementwise tanh to keep generator outputs in [-1, 1]."""

    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1

    def forward(self, x, w):
        return jnp.tanh(x)


def prepare_data() -> jnp.ndarray:
    train_images, _, _, _ = load_cifar100(normalize=True)
    images = jnp.asarray(train_images, dtype=jnp.float32)
    images = images * 2.0 - 1.0  # scale to [-1, 1]
    return images


def build_generator(
    latent_dim: int,
    image_shape: Tuple[int, int, int],
    hidden_dim: int,
    conv_channels: int = 64,
):
    height, width, channels = image_shape
    base_channels = conv_channels
    upscale_factor = 2
    expanded_height = height * upscale_factor
    expanded_width = width * upscale_factor
    flatten_dim = expanded_height * expanded_width * base_channels

    generator = Tanh()
    generator @= Conv2D(base_channels, channels, kernel_size=3, retract_enabled=False)
    generator @= ReLU()
    generator @= MaxPool2D(pool_size=upscale_factor)
    generator @= Conv2D(base_channels, base_channels, kernel_size=3, retract_enabled=False)
    generator @= ReLU()
    generator @= Conv2D(base_channels, base_channels, kernel_size=3, retract_enabled=False)
    generator @= ReLU()
    generator @= Reshape((expanded_height, expanded_width, base_channels))
    generator @= Linear(flatten_dim, hidden_dim)
    generator @= ReLU()
    generator @= Linear(hidden_dim, latent_dim)
    generator.jit()
    return generator


def build_discriminator(
    image_shape: Tuple[int, int, int],
    hidden_dim: int,
    conv_channels: int = 64,
):
    height, width, channels = image_shape
    conv1_channels = conv_channels
    conv2_channels = max(conv_channels * 2, conv_channels)
    pool_size = 2

    def pooled_dim(size: int) -> int:
        return ((size - pool_size) // pool_size) + 1

    height_after_pool1 = pooled_dim(height)
    width_after_pool1 = pooled_dim(width)
    height_after_pool2 = pooled_dim(height_after_pool1)
    width_after_pool2 = pooled_dim(width_after_pool1)
    flatten_dim = height_after_pool2 * width_after_pool2 * conv2_channels

    discriminator = Linear(1, hidden_dim)
    discriminator @= ReLU()
    discriminator @= Linear(hidden_dim, flatten_dim)
    discriminator @= Flatten()
    discriminator @= MaxPool2D(pool_size=pool_size)
    discriminator @= ReLU()
    discriminator @= Conv2D(conv1_channels, conv2_channels, kernel_size=3)
    discriminator @= MaxPool2D(pool_size=pool_size)
    discriminator @= ReLU()
    discriminator @= Conv2D(channels, conv1_channels, kernel_size=3)
    discriminator.jit()
    return discriminator


def sample_real_batch(key: jax.Array, batch_size: int, dataset: jnp.ndarray) -> jnp.ndarray:
    idx = jax.random.choice(key, dataset.shape[0], shape=(batch_size,), replace=False)
    return dataset[idx]


def sample_latent(key: jax.Array, batch_size: int, latent_dim: int) -> jnp.ndarray:
    return jax.random.normal(key, shape=(batch_size, latent_dim), dtype=jnp.float32)


def make_discriminator_loss(discriminator, generator):
    def loss_fn(disc_w, gen_w, real_batch, noise):
        fake_images = generator(noise, gen_w)
        real_logits = discriminator(real_batch, disc_w)
        fake_logits = discriminator(fake_images, disc_w)
        real_loss = jnp.mean(jax.nn.softplus(-real_logits))
        fake_loss = jnp.mean(jax.nn.softplus(fake_logits))
        return real_loss + fake_loss

    return loss_fn


def make_generator_loss(discriminator, generator):
    def loss_fn(gen_w, disc_w, noise):
        fake_images = generator(noise, gen_w)
        fake_logits = discriminator(fake_images, disc_w)
        return jnp.mean(jax.nn.softplus(-fake_logits))

    return loss_fn


def train_single_run(
    generator,
    discriminator,
    base_key: jax.Array,
    generator_method: str,
    discriminator_method: str,
    learning_rate: float,
    steps: int,
    batch_size: int,
    target_norm: float,
    dataset: jnp.ndarray,
    latent_dim: int,
    dual_alpha: float,
    dual_beta: float,
) -> Dict[str, object]:
    key_gen_init, key_disc_init, key_loop = jax.random.split(base_key, 3)
    gen_weights = generator.initialize(key_gen_init)
    disc_weights = discriminator.initialize(key_disc_init)

    gen_dual_state = generator.init_dual_state(gen_weights) if generator_method == "manifold_online" else None
    disc_dual_state = discriminator.init_dual_state(disc_weights) if discriminator_method == "manifold_online" else None

    disc_loss_fn = make_discriminator_loss(discriminator, generator)
    gen_loss_fn = make_generator_loss(discriminator, generator)
    disc_loss_and_grad = jax.jit(jax.value_and_grad(disc_loss_fn))
    gen_loss_and_grad = jax.jit(jax.value_and_grad(gen_loss_fn))

    generator_losses: List[float] = []
    discriminator_losses: List[float] = []
    disc_loss_value = 0.0
    gen_loss_value = 0.0

    description = f"G:{generator_method} D:{discriminator_method} lr={learning_rate:.3g}"
    for _ in trange(steps, leave=False, desc=description):
        key_loop, key_real, key_noise_d, key_noise_g = jax.random.split(key_loop, 4)
        real_batch = sample_real_batch(key_real, batch_size, dataset)
        noise_for_disc = sample_latent(key_noise_d, batch_size, latent_dim)
        noise_for_gen = sample_latent(key_noise_g, batch_size, latent_dim)

        disc_loss_value, disc_grads = disc_loss_and_grad(disc_weights, gen_weights, real_batch, noise_for_disc)

        if discriminator_method == "manifold":
            tangents = discriminator.dual_ascent(disc_weights, disc_grads, target_norm=target_norm)
            disc_weights = [w - learning_rate * t for w, t in zip(disc_weights, tangents)]
            disc_weights = discriminator.retract(disc_weights)
        elif discriminator_method == "manifold_online":
            tangents, disc_dual_state = discriminator.online_dual_ascent(
                disc_dual_state,
                disc_weights,
                disc_grads,
                target_norm=target_norm,
                alpha=dual_alpha,
                beta=dual_beta,
            )
            disc_weights = [w - learning_rate * t for w, t in zip(disc_weights, tangents)]
            disc_weights = discriminator.retract(disc_weights)
            disc_dual_state = dampen_dual_state(disc_dual_state, factor=0.25, zero_velocity=True)
        elif discriminator_method == "dualize":
            directions = discriminator.dualize(disc_grads, target_norm=target_norm)
            disc_weights = [w - learning_rate * direction for w, direction in zip(disc_weights, directions)]
        elif discriminator_method == "descent":
            disc_weights = [w - learning_rate * grad for w, grad in zip(disc_weights, disc_grads)]
        else:
            raise ValueError(f"Unknown discriminator method: {discriminator_method}")

        gen_loss_value, gen_grads = gen_loss_and_grad(gen_weights, disc_weights, noise_for_gen)

        if generator_method == "manifold":
            tangents = generator.dual_ascent(gen_weights, gen_grads, target_norm=target_norm)
            gen_weights = [w - learning_rate * t for w, t in zip(gen_weights, tangents)]
            gen_weights = generator.retract(gen_weights)
        elif generator_method == "manifold_online":
            tangents, gen_dual_state = generator.online_dual_ascent(
                gen_dual_state,
                gen_weights,
                gen_grads,
                target_norm=target_norm,
                alpha=dual_alpha,
                beta=dual_beta,
            )
            gen_weights = [w - learning_rate * t for w, t in zip(gen_weights, tangents)]
            gen_weights = generator.retract(gen_weights)
            gen_dual_state = dampen_dual_state(gen_dual_state, factor=0.25, zero_velocity=True)
        elif generator_method == "dualize":
            directions = generator.dualize(gen_grads, target_norm=target_norm)
            gen_weights = [w - learning_rate * direction for w, direction in zip(gen_weights, directions)]
        elif generator_method == "descent":
            gen_weights = [w - learning_rate * grad for w, grad in zip(gen_weights, gen_grads)]
        else:
            raise ValueError(f"Unknown generator method: {generator_method}")

        discriminator_losses.append(float(disc_loss_value))
        generator_losses.append(float(gen_loss_value))

    key_loop, key_eval_noise, key_eval_real = jax.random.split(key_loop, 3)
    eval_noise = sample_latent(key_eval_noise, batch_size, latent_dim)
    eval_real = sample_real_batch(key_eval_real, batch_size, dataset)

    fake_images = generator(eval_noise, gen_weights)
    fake_logits = discriminator(fake_images, disc_weights)
    real_logits = discriminator(eval_real, disc_weights)
    mean_fake_score = float(jnp.mean(jax.nn.sigmoid(fake_logits)))
    mean_real_score = float(jnp.mean(jax.nn.sigmoid(real_logits)))

    return {
        "generator_weights": gen_weights,
        "discriminator_weights": disc_weights,
        "generator_losses": generator_losses,
        "discriminator_losses": discriminator_losses,
        "final_generator_loss": float(gen_loss_value),
        "final_discriminator_loss": float(disc_loss_value),
        "mean_real_score": mean_real_score,
        "mean_fake_score": mean_fake_score,
    }


def plot_losses(best_runs: Dict[str, Dict[str, object]], plots_dir: Path) -> None:
    if not best_runs:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax_gen, ax_disc) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    cmap = plt.get_cmap("tab10")

    for idx, (method, run) in enumerate(best_runs.items()):
        steps = np.arange(len(run["generator_losses"]))
        color = cmap(idx % 10)
        ax_gen.plot(steps, run["generator_losses"], label=method, color=color)
        ax_disc.plot(steps, run["discriminator_losses"], label=method, color=color)

    ax_gen.set_title("Generator loss")
    ax_disc.set_title("Discriminator loss")
    ax_gen.set_xlabel("Step")
    ax_disc.set_xlabel("Step")
    ax_gen.set_ylabel("Loss")
    ax_disc.set_ylabel("Loss")
    ax_gen.grid(True, linestyle="--", alpha=0.3)
    ax_disc.grid(True, linestyle="--", alpha=0.3)
    ax_gen.legend()
    ax_disc.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "cifar100_gan_loss_curves.png", dpi=300)
    plt.close(fig)


def save_samples(
    generator,
    best_runs: Dict[str, Dict[str, object]],
    latent_dim: int,
    plots_dir: Path,
    grid_size: int,
    seed: int,
) -> List[Tuple[str, Path]]:
    if not best_runs:
        return []

    plots_dir.mkdir(parents=True, exist_ok=True)
    sample_records: List[Tuple[str, Path]] = []

    for idx, (method, run) in enumerate(best_runs.items()):
        key = jax.random.PRNGKey(seed + idx)
        noise = sample_latent(key, grid_size * grid_size, latent_dim)
        generated = generator(noise, run["generator_weights"])
        generated = np.asarray(generated)
        generated = np.clip((generated + 1.0) / 2.0, 0.0, 1.0)

        if generated.ndim == 4:
            if generated.shape[-1] == 1:
                display_images = generated[..., 0]
            else:
                display_images = generated
        elif generated.ndim == 3:
            display_images = generated
        else:
            flat = generated.reshape(generated.shape[0], -1)
            side = int(np.sqrt(flat.shape[-1]))
            if side * side == flat.shape[-1]:
                display_images = flat.reshape(generated.shape[0], side, side)
            else:
                display_images = flat

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
        for image, axis in zip(display_images, axes.flatten()):
            if image.ndim == 2:
                axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            elif image.ndim == 3:
                axis.imshow(image, vmin=0.0, vmax=1.0)
            else:
                axis.plot(image)
            axis.axis("off")

        fig.tight_layout()
        output_path = plots_dir / f"cifar100_gan_samples_{method}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        sample_records.append((method, output_path))

    return sample_records


def save_results(
    results: Dict[str, List[Dict[str, object]]],
    best_runs: Dict[str, Dict[str, object]],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": {
            "learning_rates": [float(lr) for lr in args.learning_rates],
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "target_norm": float(args.target_norm),
            "hidden_width": int(args.hidden_width),
            "latent_dim": int(args.latent_dim),
            "methods": list(args.methods),
            "generator_methods": list(args.generator_methods) if args.generator_methods is not None else None,
            "discriminator_methods": list(args.discriminator_methods) if args.discriminator_methods is not None else None,
        },
        "methods": {},
    }

    for method, runs in results.items():
        payload["methods"][method] = {
            "runs": [
                {
                    "learning_rate": float(entry["learning_rate"]),
                    "final_generator_loss": float(entry["final_generator_loss"]),
                    "final_discriminator_loss": float(entry["final_discriminator_loss"]),
                    "mean_real_score": float(entry["mean_real_score"]),
                    "mean_fake_score": float(entry["mean_fake_score"]),
                    "generator_losses": [float(val) for val in entry["generator_losses"]],
                    "discriminator_losses": [float(val) for val in entry["discriminator_losses"]],
                    "generator_method": entry["generator_method"],
                    "discriminator_method": entry["discriminator_method"],
                }
                for entry in runs
            ]
        }
        best = best_runs.get(method)
        if best is not None:
            payload["methods"][method]["best"] = {
                "learning_rate": float(best["learning_rate"]),
                "final_generator_loss": float(best["final_generator_loss"]),
                "final_discriminator_loss": float(best["final_discriminator_loss"]),
                "mean_real_score": float(best["mean_real_score"]),
                "mean_fake_score": float(best["mean_fake_score"]),
                "generator_method": best["generator_method"],
                "discriminator_method": best["discriminator_method"],
            }

    with output_path.open("w") as fp:
        json.dump(payload, fp, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-100 GAN experiment with manifold optimization")
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e-3],
        help="Learning rates to sweep",
    )
    parser.add_argument("--steps", type=int, default=1000, help="Training steps per learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    parser.add_argument("--target-norm", type=float, default=1.0, help="Target norm for tangent updates")
    parser.add_argument("--hidden-width", type=int, default=512, help="Hidden layer width")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension for generator input")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["descent", "dualize", "manifold_online"],
        choices=METHOD_CHOICES,
        help="Training methods to compare",
    )
    parser.add_argument(
        "--generator-methods",
        type=str,
        nargs="+",
        choices=METHOD_CHOICES,
        help="Generator training methods to compare (defaults to --methods when unset)",
    )
    parser.add_argument(
        "--discriminator-methods",
        type=str,
        nargs="+",
        choices=METHOD_CHOICES,
        help="Discriminator training methods to compare (defaults to --methods when unset)",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("results/cifar100_gan_manifold_results.json"),
        help="Path to save sweep metrics",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("plots/cifar100"),
        help="Directory for plot outputs",
    )
    parser.add_argument(
        "--sample-grid",
        type=int,
        default=4,
        help="Grid size for generated sample visualization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = prepare_data()
    image_shape = tuple(dataset.shape[1:])

    generator = build_generator(args.latent_dim, image_shape, args.hidden_width)
    discriminator = build_discriminator(image_shape, args.hidden_width)

    base_key = jax.random.PRNGKey(args.seed)

    dual_alpha = 2e-5
    dual_beta = 0.90

    if args.generator_methods is None and args.discriminator_methods is None:
        method_pairs = [(method, method) for method in args.methods]
    else:
        generator_methods = args.generator_methods or args.methods
        discriminator_methods = args.discriminator_methods or args.methods
        method_pairs = []
        for gen_method in generator_methods:
            for disc_method in discriminator_methods:
                method_pairs.append((gen_method, disc_method))

    method_labels = [format_method_label(gen_method, disc_method) for gen_method, disc_method in method_pairs]
    results: Dict[str, List[Dict[str, object]]] = {label: [] for label in method_labels}
    best_runs: Dict[str, Dict[str, object]] = {}

    for pair_idx, (generator_method, discriminator_method) in enumerate(method_pairs):
        label = format_method_label(generator_method, discriminator_method)
        method_key = jax.random.fold_in(base_key, pair_idx)

        for lr_idx, learning_rate in enumerate(args.learning_rates):
            run_key = jax.random.fold_in(method_key, lr_idx)
            run = train_single_run(
                generator,
                discriminator,
                run_key,
                generator_method,
                discriminator_method,
                learning_rate,
                args.steps,
                args.batch_size,
                args.target_norm,
                dataset,
                args.latent_dim,
                dual_alpha,
                dual_beta,
            )

            entry = {
                "learning_rate": learning_rate,
                "final_generator_loss": run["final_generator_loss"],
                "final_discriminator_loss": run["final_discriminator_loss"],
                "mean_real_score": run["mean_real_score"],
                "mean_fake_score": run["mean_fake_score"],
                "generator_losses": run["generator_losses"],
                "discriminator_losses": run["discriminator_losses"],
                "generator_method": generator_method,
                "discriminator_method": discriminator_method,
            }
            results[label].append(entry)

            best = best_runs.get(label)
            if best is None or run["final_generator_loss"] < best["final_generator_loss"]:
                best_runs[label] = {
                    "learning_rate": learning_rate,
                    "final_generator_loss": run["final_generator_loss"],
                    "final_discriminator_loss": run["final_discriminator_loss"],
                    "mean_real_score": run["mean_real_score"],
                    "mean_fake_score": run["mean_fake_score"],
                    "generator_losses": run["generator_losses"],
                    "discriminator_losses": run["discriminator_losses"],
                    "generator_weights": run["generator_weights"],
                    "discriminator_weights": run["discriminator_weights"],
                    "generator_method": generator_method,
                    "discriminator_method": discriminator_method,
                }

            print(
                f"[G:{generator_method} | D:{discriminator_method}] lr={learning_rate:.3g}: "
                f"G loss={run['final_generator_loss']:.4f} | "
                f"D loss={run['final_discriminator_loss']:.4f} | "
                f"real={run['mean_real_score']:.3f} fake={run['mean_fake_score']:.3f}"
            )

    plot_losses(best_runs, args.plots_dir)
    sample_paths = save_samples(generator, best_runs, args.latent_dim, args.plots_dir, args.sample_grid, args.seed)

    save_results(results, best_runs, args, args.results_path)

    if sample_paths:
        for method, path in sample_paths:
            print(f"Saved samples for {method} to {path}")


if __name__ == "__main__":
    main()
