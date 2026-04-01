import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import wandb
from scipy import linalg
from scipy.stats import entropy
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights

from data.cifar10 import load_cifar10
from modula.abstract import Bond, Identity, Add
from modula.atom import Linear, Conv2D, Conv2DTranspose, BatchNorm2D, dampen_dual_state
from modula.bond import ReLU, LeakyReLU, Flatten, MaxPool2D

METHOD_CHOICES = ("manifold", "manifold_online", "dualize", "descent", "adam")


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
    train_images, _, _, _ = load_cifar10(normalize=True)
    images = jnp.asarray(train_images, dtype=jnp.float32)
    images = images * 2.0 - 1.0  # scale to [-1, 1]
    return images


def build_inception_v3_features():
    """Load pretrained InceptionV3 model for FID and IS calculation.
    Returns the model in evaluation mode with only the feature extraction layers.
    """
    # Load pretrained InceptionV3
    inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    inception_model.eval()
    
    # Set to evaluation mode and disable gradients
    for param in inception_model.parameters():
        param.requires_grad = False
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception_model = inception_model.to(device)
    
    return inception_model, device


def preprocess_images_for_inception(images: np.ndarray, device: torch.device) -> torch.Tensor:
    """Preprocess images for InceptionV3.
    
    Args:
        images: Images in range [-1, 1] with shape (N, H, W, C)
        device: torch device to put tensors on
        
    Returns:
        Preprocessed images ready for InceptionV3
    """
    # Convert to [0, 1]
    images = (images + 1.0) / 2.0
    images = np.clip(images, 0.0, 1.0)
    
    # InceptionV3 expects (N, C, H, W) and images of size at least 299x299
    # CIFAR is 32x32, so we need to resize
    from torch.nn import functional as F
    
    # Convert from (N, H, W, C) to (N, C, H, W)
    images_nchw = np.transpose(images, (0, 3, 1, 2))
    images_tensor = torch.from_numpy(images_nchw).float().to(device)
    
    # Resize to 299x299
    images_resized = F.interpolate(images_tensor, size=(299, 299), mode='bilinear', align_corners=False)
    
    # Normalize for InceptionV3 (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    images_normalized = (images_resized - mean) / std
    
    return images_normalized


def get_inception_features(images: np.ndarray, inception_model, device: torch.device, batch_size: int = 50) -> np.ndarray:
    """Extract features from images using InceptionV3.
    
    Args:
        images: Images in range [-1, 1]
        inception_model: Pretrained InceptionV3 model
        device: torch device
        batch_size: Batch size for processing
        
    Returns:
        Features from the pool3 layer (2048-dimensional)
    """
    num_images = images.shape[0]
    features_list = []
    
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            batch = images[i:i+batch_size]
            batch_preprocessed = preprocess_images_for_inception(batch, device)
            
            # Get features from the final pooling layer before classifier
            # InceptionV3 forward pass
            x = inception_model.Conv2d_1a_3x3(batch_preprocessed)
            x = inception_model.Conv2d_2a_3x3(x)
            x = inception_model.Conv2d_2b_3x3(x)
            x = inception_model.maxpool1(x)
            x = inception_model.Conv2d_3b_1x1(x)
            x = inception_model.Conv2d_4a_3x3(x)
            x = inception_model.maxpool2(x)
            x = inception_model.Mixed_5b(x)
            x = inception_model.Mixed_5c(x)
            x = inception_model.Mixed_5d(x)
            x = inception_model.Mixed_6a(x)
            x = inception_model.Mixed_6b(x)
            x = inception_model.Mixed_6c(x)
            x = inception_model.Mixed_6d(x)
            x = inception_model.Mixed_6e(x)
            x = inception_model.Mixed_7a(x)
            x = inception_model.Mixed_7b(x)
            x = inception_model.Mixed_7c(x)
            # Global average pooling
            x = inception_model.avgpool(x)
            x = torch.flatten(x, 1)
            
            features_list.append(x.cpu().numpy())
    
    features = np.concatenate(features_list, axis=0)
    return features


def get_inception_predictions(images: np.ndarray, inception_model, device: torch.device, batch_size: int = 50) -> np.ndarray:
    """Get InceptionV3 predictions (probabilities) for images.
    
    Args:
        images: Images in range [-1, 1]
        inception_model: Pretrained InceptionV3 model
        device: torch device
        batch_size: Batch size for processing
        
    Returns:
        Softmax probabilities from InceptionV3
    """
    num_images = images.shape[0]
    predictions_list = []
    
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            batch = images[i:i+batch_size]
            batch_preprocessed = preprocess_images_for_inception(batch, device)
            
            # Full forward pass
            logits = inception_model(batch_preprocessed)
            probs = torch.nn.functional.softmax(logits, dim=1)
            predictions_list.append(probs.cpu().numpy())
    
    predictions = np.concatenate(predictions_list, axis=0)
    return predictions


def calculate_inception_score(images: jnp.ndarray, inception_model, device: torch.device, splits=10, eps=1e-16):
    """Calculate Inception Score for generated images using pretrained InceptionV3.
    
    Args:
        images: Generated images in range [-1, 1]
        inception_model: Pretrained InceptionV3 model
        device: torch device
        splits: Number of splits for IS calculation
        eps: Small value for numerical stability
    """
    images_np = np.asarray(images)
    
    # Get predictions
    preds = get_inception_predictions(images_np, inception_model, device)
    
    # Calculate IS
    split_scores = []
    split_size = preds.shape[0] // splits
    
    for i in range(splits):
        part = preds[i * split_size:(i + 1) * split_size]
        # KL divergence between p(y|x) and p(y)
        py = np.mean(part, axis=0)
        scores = []
        for j in range(part.shape[0]):
            pyx = part[j]
            scores.append(np.sum(pyx * np.log(pyx / (py + eps) + eps)))
        split_scores.append(np.exp(np.mean(scores)))
    
    return float(np.mean(split_scores)), float(np.std(split_scores))


def calculate_fid(real_images: jnp.ndarray, fake_images: jnp.ndarray, 
                  inception_model, device: torch.device, eps=1e-6):
    """Calculate Fréchet Inception Distance using pretrained InceptionV3.
    
    Args:
        real_images: Real images in range [-1, 1]
        fake_images: Generated images in range [-1, 1]
        inception_model: Pretrained InceptionV3 model
        device: torch device
        eps: Small value for numerical stability
    """
    real_images_np = np.asarray(real_images)
    fake_images_np = np.asarray(fake_images)
    
    # Extract features
    real_features = get_inception_features(real_images_np, inception_model, device)
    fake_features = get_inception_features(fake_images_np, inception_model, device)
    
    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    
    # Handle numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return float(fid)


class GlobalAvgPool(Bond):
    """Global average pooling."""
    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1

    def forward(self, x, w):
        return jnp.mean(x, axis=(1, 2))


def build_generator(
    latent_dim: int,
    image_shape: Tuple[int, int, int],
    hidden_dim: int,
    conv_channels: int = 64,
):
    height, width, channels = image_shape
    # Starting from 4×4 spatial size with 512 channels
    start_size = 4
    start_channels = 512
    
    # Build in reverse (bottom-up)
    generator = Tanh()
    generator @= Conv2DTranspose(64, channels, kernel_size=4, stride=1, retract_enabled=False, use_weight_norm=True)
    generator @= ReLU()
    generator @= BatchNorm2D(64)
    generator @= Conv2DTranspose(128, 64, kernel_size=4, stride=2)
    generator @= ReLU()
    generator @= BatchNorm2D(128)
    generator @= Conv2DTranspose(256, 128, kernel_size=4, stride=2)
    generator @= ReLU()
    generator @= BatchNorm2D(256)
    generator @= Conv2DTranspose(start_channels, 256, kernel_size=4, stride=2)
    generator @= ReLU()
    generator @= BatchNorm2D(start_channels)
    generator @= Reshape((start_size, start_size, start_channels))
    generator @= Linear(start_size * start_size * start_channels, latent_dim)
    generator.jit()
    return generator


def build_discriminator(
    image_shape: Tuple[int, int, int],
    hidden_dim: int,
    conv_channels: int = 64,
):
    height, width, channels = image_shape
    # After stride=2 three times: 32 -> 16 -> 8 -> 4
    final_spatial_size = 4
    final_channels = 512
    flatten_dim = final_spatial_size * final_spatial_size * final_channels
    
    # Build in reverse (top-down)
    discriminator = Linear(1, flatten_dim)
    discriminator @= Flatten()
    discriminator @= LeakyReLU(negative_slope=0.2)
    discriminator @= Conv2D(512, 512, kernel_size=3, stride=1)
    discriminator @= LeakyReLU(negative_slope=0.2)
    discriminator @= Conv2D(256, 512, kernel_size=3, stride=2)
    discriminator @= LeakyReLU(negative_slope=0.2)
    discriminator @= Conv2D(256, 256, kernel_size=3, stride=1)
    discriminator @= LeakyReLU(negative_slope=0.2)
    discriminator @= Conv2D(128, 256, kernel_size=3, stride=2)
    discriminator @= LeakyReLU(negative_slope=0.2)
    discriminator @= Conv2D(128, 128, kernel_size=3, stride=1)
    discriminator @= LeakyReLU(negative_slope=0.2)
    discriminator @= Conv2D(64, 128, kernel_size=3, stride=2)
    discriminator @= LeakyReLU(negative_slope=0.2)
    discriminator @= Conv2D(channels, 64, kernel_size=3, stride=1)
    discriminator.jit()
    return discriminator


def sample_real_batch(key: jax.Array, batch_size: int, dataset: jnp.ndarray) -> jnp.ndarray:
    idx = jax.random.choice(key, dataset.shape[0], shape=(batch_size,), replace=False)
    return dataset[idx]


def sample_latent(key: jax.Array, batch_size: int, latent_dim: int) -> jnp.ndarray:
    return jax.random.normal(key, shape=(batch_size, latent_dim), dtype=jnp.float32)


def make_discriminator_loss(discriminator, generator, lambda_gp=10.0, use_gradient_penalty=True):
    def loss_fn(disc_w, gen_w, real_batch, noise, gp_key):
        fake_images = generator(noise, gen_w)
        real_logits = discriminator(real_batch, disc_w)
        fake_logits = discriminator(fake_images, disc_w)
        
        # Wasserstein loss: we want to maximize (real_logits - fake_logits),
        # so we minimize (fake_logits - real_logits)
        wasserstein_loss = jnp.mean(fake_logits) - jnp.mean(real_logits)
        
        if use_gradient_penalty:
            # Gradient penalty with key for random interpolation
            batch_size = real_batch.shape[0]
            alpha = jax.random.uniform(gp_key, shape=(batch_size, 1, 1, 1))
            interpolated = alpha * real_batch + (1 - alpha) * fake_images
            
            def disc_fn(x):
                return jnp.sum(discriminator(x, disc_w))
            
            grads = jax.grad(disc_fn)(interpolated)
            grads_flat = grads.reshape(batch_size, -1)
            grad_norms = jnp.sqrt(jnp.sum(grads_flat ** 2, axis=1) + 1e-12)
            gradient_penalty = lambda_gp * jnp.mean((grad_norms - 1.0) ** 2)
        else:
            gradient_penalty = 0.0
            grad_norms = jnp.array(0.0)
        
        return wasserstein_loss + gradient_penalty, (wasserstein_loss, gradient_penalty, jnp.mean(grad_norms) if use_gradient_penalty else 0.0)

    return loss_fn


def make_generator_loss(discriminator, generator):
    def loss_fn(gen_w, disc_w, noise):
        fake_images = generator(noise, gen_w)
        fake_logits = discriminator(fake_images, disc_w)
        # Generator wants to maximize discriminator output on fake images (minimize negative)
        return -jnp.mean(fake_logits)

    return loss_fn


def train_single_run(
    generator,
    discriminator,
    base_key: jax.Array,
    generator_method: str,
    discriminator_method: str,
    generator_lr: float,
    discriminator_lr: float,
    steps: int,
    batch_size: int,
    target_norm: float,
    dataset: jnp.ndarray,
    latent_dim: int,
    dual_alpha: float,
    dual_beta: float,
    use_wandb: bool = False,
    lambda_gp: float = 1.0,
    inception_model = None,
    device = None,
    eval_interval: int = 1000,
    discriminator_train_steps: int = 1,
    use_gradient_penalty: bool = True,
) -> Dict[str, object]:
    key_gen_init, key_disc_init, key_pretrain, key_loop = jax.random.split(base_key, 4)
    gen_weights = generator.initialize(key_gen_init)
    disc_weights = discriminator.initialize(key_disc_init)

    disc_dual_state = discriminator.init_dual_state(disc_weights) if discriminator_method == "manifold_online" else None
    
    # Initialize Adam optimizer states if needed
    gen_adam_m = None
    gen_adam_v = None
    disc_adam_m = None
    disc_adam_v = None
    adam_beta1 = 0.0
    adam_beta2 = 0.9
    adam_eps = 1e-8
    gen_adam_step = 0
    disc_adam_step = 0
    
    if generator_method == "adam":
        gen_adam_m = [jnp.zeros_like(w) for w in gen_weights]
        gen_adam_v = [jnp.zeros_like(w) for w in gen_weights]
    if discriminator_method == "adam":
        disc_adam_m = [jnp.zeros_like(w) for w in disc_weights]
        disc_adam_v = [jnp.zeros_like(w) for w in disc_weights]

    disc_loss_fn = make_discriminator_loss(discriminator, generator, lambda_gp, use_gradient_penalty)
    gen_loss_fn = make_generator_loss(discriminator, generator)
    disc_loss_and_grad = jax.jit(jax.value_and_grad(disc_loss_fn, has_aux=True))
    gen_loss_and_grad = jax.jit(jax.value_and_grad(gen_loss_fn))

    generator_losses: List[float] = []
    discriminator_losses: List[float] = []
    disc_loss_value = 0.0
    gen_loss_value = 0.0
    wasserstein_dist = 0.0
    gp_value = 0.0
    gp_grad_norm = 0.0

    description = f"G:{generator_method}(lr={generator_lr:.3g}) D:{discriminator_method}(lr={discriminator_lr:.3g})"
    for step in trange(steps, leave=False, desc=description):
        key_loop, key_real, key_noise_d, key_noise_g = jax.random.split(key_loop, 4)
        
        # Train discriminator multiple times per generator update
        for disc_iter in range(discriminator_train_steps):
            key_loop, key_real_inner, key_noise_inner, key_gp = jax.random.split(key_loop, 4)
            real_batch = sample_real_batch(key_real_inner, batch_size, dataset)
            noise_for_disc = sample_latent(key_noise_inner, batch_size, latent_dim)
            
            (disc_loss_value, (wasserstein_dist, gp_value, gp_grad_norm)), disc_grads = disc_loss_and_grad(disc_weights, gen_weights, real_batch, noise_for_disc, key_gp)
            disc_grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in disc_grads))

            if discriminator_method == "manifold_online":
                tangents, disc_dual_state = discriminator.online_dual_ascent(
                    disc_dual_state,
                    disc_weights,
                    disc_grads,
                    target_norm=target_norm,
                    alpha=dual_alpha,
                    beta=dual_beta,
                )
                disc_tangent_norm = jnp.sqrt(sum(jnp.sum(t ** 2) for t in tangents))
                disc_weights = [w - discriminator_lr * t for w, t in zip(disc_weights, tangents)]
                disc_weights = discriminator.retract(disc_weights)
                disc_dual_state = dampen_dual_state(disc_dual_state, factor=0.25, zero_velocity=True)
            elif discriminator_method == "adam":
                disc_adam_step += 1
                disc_adam_m = [adam_beta1 * m + (1 - adam_beta1) * g for m, g in zip(disc_adam_m, disc_grads)]
                disc_adam_v = [adam_beta2 * v + (1 - adam_beta2) * (g ** 2) for v, g in zip(disc_adam_v, disc_grads)]
                m_hat = [m / (1 - adam_beta1 ** disc_adam_step) for m in disc_adam_m]
                v_hat = [v / (1 - adam_beta2 ** disc_adam_step) for v in disc_adam_v]
                disc_weights = [w - discriminator_lr * mh / (jnp.sqrt(vh) + adam_eps) for w, mh, vh in zip(disc_weights, m_hat, v_hat)]
                disc_tangent_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in disc_grads))
            else:
                raise ValueError(f"Unknown discriminator method: {discriminator_method}")

        # Train generator once
        real_batch = sample_real_batch(key_real, batch_size, dataset)
        noise_for_gen = sample_latent(key_noise_g, batch_size, latent_dim)
        
        gen_loss_value, gen_grads = gen_loss_and_grad(gen_weights, disc_weights, noise_for_gen)
        gen_grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in gen_grads))

        if generator_method == "dualize":
            directions = generator.dualize(gen_grads, target_norm=target_norm)
            gen_tangent_norm = jnp.sqrt(sum(jnp.sum(d ** 2) for d in directions))
            gen_weights = [w - generator_lr * direction for w, direction in zip(gen_weights, directions)]
        elif generator_method == "descent":
            gen_tangent_norm = gen_grad_norm
            gen_weights = [w - generator_lr * g for w, g in zip(gen_weights, gen_grads)]
        elif generator_method == "adam":
            gen_adam_step += 1
            gen_adam_m = [adam_beta1 * m + (1 - adam_beta1) * g for m, g in zip(gen_adam_m, gen_grads)]
            gen_adam_v = [adam_beta2 * v + (1 - adam_beta2) * (g ** 2) for v, g in zip(gen_adam_v, gen_grads)]
            m_hat = [m / (1 - adam_beta1 ** gen_adam_step) for m in gen_adam_m]
            v_hat = [v / (1 - adam_beta2 ** gen_adam_step) for v in gen_adam_v]
            gen_weights = [w - generator_lr * mh / (jnp.sqrt(vh) + adam_eps) for w, mh, vh in zip(gen_weights, m_hat, v_hat)]
            gen_tangent_norm = gen_grad_norm
        else:
            raise ValueError(f"Unknown generator method: {generator_method}")

        discriminator_losses.append(float(disc_loss_value))
        generator_losses.append(float(gen_loss_value))
        
        if use_wandb:
            # Calculate per-layer gradient norms
            disc_grad_norms_per_layer = [float(jnp.linalg.norm(g)) for g in disc_grads]
            gen_grad_norms_per_layer = [float(jnp.linalg.norm(g)) for g in gen_grads]
            
            log_dict = {
                "step": step,
                "generator_loss": float(gen_loss_value),
                "discriminator_loss": float(disc_loss_value),
                "wasserstein_distance": float(wasserstein_dist),
                "gradient_penalty": float(gp_value),
                "gp_grad_norm": float(gp_grad_norm),
                "generator_grad_norm": float(gen_grad_norm),
                "discriminator_grad_norm": float(disc_grad_norm),
                "generator_tangent_norm": float(gen_tangent_norm),
                "discriminator_tangent_norm": float(disc_tangent_norm),
            }
            
            # Log per-layer gradient norms
            for i, norm in enumerate(disc_grad_norms_per_layer):
                log_dict[f"discriminator_grad_norm_layer_{i}"] = norm
            for i, norm in enumerate(gen_grad_norms_per_layer):
                log_dict[f"generator_grad_norm_layer_{i}"] = norm
            
            # Log sample images and metrics every eval_interval steps
            if step % eval_interval == 0 and step > 0:
                key_loop, key_sample, key_fid_real, key_fid_fake = jax.random.split(key_loop, 4)
                sample_noise = sample_latent(key_sample, 16, latent_dim)
                sample_images = generator(sample_noise, gen_weights)
                sample_images_np = np.asarray(sample_images)
                sample_images_display = np.clip((sample_images_np + 1.0) / 2.0, 0.0, 1.0)
                log_dict["generated_samples"] = [wandb.Image(img) for img in sample_images_display[:16]]
                
                # Calculate IS and FID if inception model is available
                if inception_model is not None and device is not None:
                    # Generate more samples for better statistics
                    eval_noise = sample_latent(key_fid_fake, min(1000, dataset.shape[0]), latent_dim)
                    eval_fake = generator(eval_noise, gen_weights)
                    eval_real = sample_real_batch(key_fid_real, min(1000, dataset.shape[0]), dataset)
                    
                    try:
                        is_mean, is_std = calculate_inception_score(eval_fake, inception_model, device)
                        log_dict["inception_score_mean"] = is_mean
                        log_dict["inception_score_std"] = is_std
                    except Exception as e:
                        print(f"Warning: Could not calculate IS: {e}")
                    
                    try:
                        fid_score = calculate_fid(eval_real, eval_fake, inception_model, device)
                        log_dict["fid_score"] = fid_score
                    except Exception as e:
                        print(f"Warning: Could not calculate FID: {e}")
            
            wandb.log(log_dict)

    key_loop, key_eval_noise, key_eval_real = jax.random.split(key_loop, 3)
    eval_noise = sample_latent(key_eval_noise, batch_size, latent_dim)
    eval_real = sample_real_batch(key_eval_real, batch_size, dataset)

    fake_images = generator(eval_noise, gen_weights)
    fake_logits = discriminator(fake_images, disc_weights)
    real_logits = discriminator(eval_real, disc_weights)
    mean_fake_score = float(jnp.mean(fake_logits))
    mean_real_score = float(jnp.mean(real_logits))
    
    # Calculate final metrics
    final_is_mean, final_is_std, final_fid = None, None, None
    if inception_model is not None and device is not None:
        try:
            # Generate samples for final evaluation
            key_loop, key_final_fake, key_final_real = jax.random.split(key_loop, 3)
            final_eval_noise = sample_latent(key_final_fake, min(2000, dataset.shape[0]), latent_dim)
            final_eval_fake = generator(final_eval_noise, gen_weights)
            final_eval_real = sample_real_batch(key_final_real, min(2000, dataset.shape[0]), dataset)
            
            final_is_mean, final_is_std = calculate_inception_score(final_eval_fake, inception_model, device)
            final_fid = calculate_fid(final_eval_real, final_eval_fake, inception_model, device)
        except Exception as e:
            print(f"Warning: Could not calculate final metrics: {e}")
    
    if use_wandb:
        final_log = {
            "final_generator_loss": float(gen_loss_value),
            "final_discriminator_loss": float(disc_loss_value),
            "mean_real_score": mean_real_score,
            "mean_fake_score": mean_fake_score,
        }
        if final_is_mean is not None:
            final_log["final_inception_score_mean"] = final_is_mean
            final_log["final_inception_score_std"] = final_is_std
        if final_fid is not None:
            final_log["final_fid_score"] = final_fid
        
        wandb.log(final_log)
        # Log sample images
        sample_noise = sample_latent(key_eval_noise, 16, latent_dim)
        sample_images = generator(sample_noise, gen_weights)
        sample_images = np.asarray(sample_images)
        sample_images = np.clip((sample_images + 1.0) / 2.0, 0.0, 1.0)
        wandb.log({"generated_samples": [wandb.Image(img) for img in sample_images[:16]]})

    result = {
        "generator_weights": gen_weights,
        "discriminator_weights": disc_weights,
        "generator_losses": generator_losses,
        "discriminator_losses": discriminator_losses,
        "final_generator_loss": float(gen_loss_value),
        "final_discriminator_loss": float(disc_loss_value),
        "mean_real_score": mean_real_score,
        "mean_fake_score": mean_fake_score,
    }
    
    if final_is_mean is not None:
        result["final_inception_score_mean"] = final_is_mean
        result["final_inception_score_std"] = final_is_std
    if final_fid is not None:
        result["final_fid_score"] = final_fid
    
    return result


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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(plots_dir / f"lipschitz_gan_plot_{timestamp}.png", dpi=300)
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
        # Include learning rates and method in filename to avoid conflicts
        gen_lr = run["generator_lr"]
        disc_lr = run["discriminator_lr"]
        safe_method = method.replace(":", "_").replace("|", "_")
        output_path = plots_dir / f"lipschitz_gan_samples_{safe_method}_glr{gen_lr}_dlr{disc_lr}_{timestamp}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        sample_records.append((method, output_path))

    return sample_records


def save_model(
    generator,
    discriminator,
    gen_weights,
    disc_weights,
    method_label: str,
    gen_lr: float,
    disc_lr: float,
    output_dir: Path,
) -> Path:
    """Save trained model weights to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_method = method_label.replace(":", "_").replace("|", "_")
    model_path = output_dir / f"model_{safe_method}_glr{gen_lr}_dlr{disc_lr}_{timestamp}.pkl"
    
    model_data = {
        "generator_weights": gen_weights,
        "discriminator_weights": disc_weights,
        "method_label": method_label,
        "generator_lr": gen_lr,
        "discriminator_lr": disc_lr,
        "timestamp": timestamp,
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    
    return model_path


def save_results(
    results: Dict[str, List[Dict[str, object]]],
    best_runs: Dict[str, Dict[str, object]],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": {
            "generator_learning_rates": [float(lr) for lr in args.generator_lrs],
            "discriminator_learning_rates": [float(lr) for lr in args.discriminator_lrs],
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
                    "generator_lr": float(entry["generator_lr"]),
                    "discriminator_lr": float(entry["discriminator_lr"]),
                    "final_generator_loss": float(entry["final_generator_loss"]),
                    "final_discriminator_loss": float(entry["final_discriminator_loss"]),
                    "mean_real_score": float(entry["mean_real_score"]),
                    "mean_fake_score": float(entry["mean_fake_score"]),
                    "generator_losses": [float(val) for val in entry["generator_losses"]],
                    "discriminator_losses": [float(val) for val in entry["discriminator_losses"]],
                    "generator_method": entry["generator_method"],
                    "discriminator_method": entry["discriminator_method"],
                    "final_inception_score_mean": float(entry.get("final_inception_score_mean", -1)),
                    "final_inception_score_std": float(entry.get("final_inception_score_std", -1)),
                    "final_fid_score": float(entry.get("final_fid_score", -1)),
                }
                for entry in runs
            ]
        }
        best = best_runs.get(method)
        if best is not None:
            payload["methods"][method]["best"] = {
                "generator_lr": float(best["generator_lr"]),
                "discriminator_lr": float(best["discriminator_lr"]),
                "final_generator_loss": float(best["final_generator_loss"]),
                "final_discriminator_loss": float(best["final_discriminator_loss"]),
                "mean_real_score": float(best["mean_real_score"]),
                "mean_fake_score": float(best["mean_fake_score"]),
                "generator_method": best["generator_method"],
                "discriminator_method": best["discriminator_method"],
                "final_inception_score_mean": float(best.get("final_inception_score_mean", -1)),
                "final_inception_score_std": float(best.get("final_inception_score_std", -1)),
                "final_fid_score": float(best.get("final_fid_score", -1)),
            }

    with output_path.open("w") as fp:
        json.dump(payload, fp, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-100 GAN experiment with manifold optimization")
    parser.add_argument(
        "--generator-lrs",
        type=float,
        nargs="+",
        default=[1e-3],
        help="Generator learning rates to sweep",
    )
    parser.add_argument(
        "--discriminator-lrs",
        type=float,
        nargs="+",
        default=[1e-3],
        help="Discriminator learning rates to sweep",
    )
    parser.add_argument("--steps", type=int, default=1000, help="Training steps per learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
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
        default=Path("results/lipschitz_gan_results.json"),
        help="Path to save sweep metrics",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("plots/lipschitz_gan"),
        help="Directory for plot outputs",
    )
    parser.add_argument(
        "--sample-grid",
        type=int,
        default=4,
        help="Grid size for generated sample visualization",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="lipschitz-gan",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity (username or team)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the best trained model weights to disk",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/lipschitz_gan"),
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1000,
        help="Interval (in steps) for evaluating IS and FID metrics",
    )
    parser.add_argument(
        "--disable-metrics",
        action="store_true",
        help="Disable IS and FID metric calculation (faster training)",
    )
    parser.add_argument(
        "--discriminator-train-steps",
        type=int,
        default=1,
        help="Number of discriminator training steps per generator update",
    )
    parser.add_argument(
        "--use-gradient-penalty",
        action="store_true",
        default=True,
        help="Use gradient penalty in discriminator loss (default: True)",
    )
    parser.add_argument(
        "--no-gradient-penalty",
        action="store_false",
        dest="use_gradient_penalty",
        help="Disable gradient penalty in discriminator loss",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = prepare_data()
    image_shape = tuple(dataset.shape[1:])

    generator = build_generator(args.latent_dim, image_shape, args.hidden_width)
    discriminator = build_discriminator(image_shape, args.hidden_width)

    # Initialize InceptionV3 for IS and FID if metrics are enabled
    inception_model = None
    device = None
    if not args.disable_metrics:
        try:
            print("Loading pretrained InceptionV3 model...")
            inception_model, device = build_inception_v3_features()
            print(f"InceptionV3 loaded successfully on device: {device}")
        except Exception as e:
            print(f"Warning: Could not load InceptionV3: {e}")
            print("Continuing without IS/FID metrics...")

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

        lr_idx = 0
        for gen_lr in args.generator_lrs:
            for disc_lr in args.discriminator_lrs:
                run_key = jax.random.fold_in(method_key, lr_idx)
                lr_idx += 1
                
                if args.use_wandb:
                    wandb.init(
                        project=args.wandb_project,
                        entity=args.wandb_entity,
                        config={
                            "generator_method": generator_method,
                            "discriminator_method": discriminator_method,
                            "generator_lr": gen_lr,
                            "discriminator_lr": disc_lr,
                            "steps": args.steps,
                            "batch_size": args.batch_size,
                            "target_norm": args.target_norm,
                            "hidden_width": args.hidden_width,
                            "latent_dim": args.latent_dim,
                            "dual_alpha": dual_alpha,
                            "dual_beta": dual_beta,
                        },
                        name=f"{label}_glr{gen_lr}_dlr{disc_lr}",
                        reinit=True,
                    )
                
                run = train_single_run(
                    generator,
                    discriminator,
                    run_key,
                    generator_method,
                    discriminator_method,
                    gen_lr,
                    disc_lr,
                    args.steps,
                    args.batch_size,
                    args.target_norm,
                    dataset,
                    args.latent_dim,
                    dual_alpha,
                    dual_beta,
                    args.use_wandb,
                    inception_model=inception_model,
                    device=device,
                    eval_interval=args.eval_interval,
                    discriminator_train_steps=args.discriminator_train_steps,
                    use_gradient_penalty=args.use_gradient_penalty,
                )
                
                if args.use_wandb:
                    wandb.finish()

                entry = {
                    "generator_lr": gen_lr,
                    "discriminator_lr": disc_lr,
                    "final_generator_loss": run["final_generator_loss"],
                    "final_discriminator_loss": run["final_discriminator_loss"],
                    "mean_real_score": run["mean_real_score"],
                    "mean_fake_score": run["mean_fake_score"],
                    "generator_losses": run["generator_losses"],
                    "discriminator_losses": run["discriminator_losses"],
                    "generator_method": generator_method,
                    "discriminator_method": discriminator_method,
                }
                
                # Add metrics if available
                if "final_inception_score_mean" in run:
                    entry["final_inception_score_mean"] = run["final_inception_score_mean"]
                    entry["final_inception_score_std"] = run["final_inception_score_std"]
                if "final_fid_score" in run:
                    entry["final_fid_score"] = run["final_fid_score"]
                
                results[label].append(entry)

                best = best_runs.get(label)
                if best is None or run["final_generator_loss"] < best["final_generator_loss"]:
                    best_runs[label] = {
                        "generator_lr": gen_lr,
                        "discriminator_lr": disc_lr,
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
                    # Add metrics if available
                    if "final_inception_score_mean" in run:
                        best_runs[label]["final_inception_score_mean"] = run["final_inception_score_mean"]
                        best_runs[label]["final_inception_score_std"] = run["final_inception_score_std"]
                    if "final_fid_score" in run:
                        best_runs[label]["final_fid_score"] = run["final_fid_score"]

                metrics_str = ""
                if "final_inception_score_mean" in run:
                    metrics_str += f" | IS={run['final_inception_score_mean']:.2f}±{run['final_inception_score_std']:.2f}"
                if "final_fid_score" in run:
                    metrics_str += f" | FID={run['final_fid_score']:.2f}"
                
                print(
                    f"[G:{generator_method} | D:{discriminator_method}] G_lr={gen_lr:.3g} D_lr={disc_lr:.3g}: "
                    f"G loss={run['final_generator_loss']:.4f} | "
                    f"D loss={run['final_discriminator_loss']:.4f} | "
                    f"real={run['mean_real_score']:.3f} fake={run['mean_fake_score']:.3f}"
                    f"{metrics_str}"
                )

    # Save best models if requested
    if args.save_model:
        print("\nSaving best models...")
        for label, run in best_runs.items():
            model_path = save_model(
                generator,
                discriminator,
                run["generator_weights"],
                run["discriminator_weights"],
                label,
                run["generator_lr"],
                run["discriminator_lr"],
                args.model_dir,
            )
            print(f"Saved model for {label} to {model_path}")

    plot_losses(best_runs, args.plots_dir)
    sample_paths = save_samples(generator, best_runs, args.latent_dim, args.plots_dir, args.sample_grid, args.seed)

    save_results(results, best_runs, args, args.results_path)

    if sample_paths:
        for method, path in sample_paths:
            print(f"Saved samples for {method} to {path}")


if __name__ == "__main__":
    main()
