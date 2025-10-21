#!/usr/bin/env python3
"""
training.py - Modula training loops for the physics-informed DeepONet.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

from controllers import encode_gust_features, encode_trunk_vec
from deeponet import DeepONet, DeepONetConfig
from physics_models import GustField, QuadcopterDynamics

METHOD_CHOICES = ("descent", "dualize", "manifold_online")

PathLike = Union[str, Path]
BatchDict = Dict[str, jnp.ndarray]


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 600
    batch_size: int = 64
    learning_rate: float = 5e-3
    lambda_dyn: float = 0.3
    target_norm: float = 1.0
    dual_alpha: float = 2e-5
    dual_beta: float = 0.9
    position_scale: float = 4.0
    velocity_scale: float = 1.5
    control_scale: float = 6.0
    storm_probability: float = 0.5
    max_bursts: int = 4
    max_encoded_bursts: int = 2
    time_horizon: float = 10.0
    seed: int = 0
    log_interval: int = 50


@dataclass
class TrainingResult:
    method: str
    learning_rate: float
    config: TrainingConfig
    deeponet_config: DeepONetConfig
    loss_history: List[float]
    mse_history: List[float]
    dyn_history: List[float]
    weights: List[jnp.ndarray]

    @property
    def final_loss(self) -> float:
        return self.loss_history[-1] if self.loss_history else float("nan")


def _spawn_burst(generator: np.random.Generator, gust_field: GustField, drone_pos: np.ndarray) -> None:
    angle_theta = float(generator.uniform(0.0, 2.0 * np.pi))
    angle_phi = float(generator.uniform(0.0, np.pi))
    distance = float(generator.uniform(2.0, gust_field.spawn_radius))

    direction = np.array([
        np.sin(angle_phi) * np.cos(angle_theta),
        np.sin(angle_phi) * np.sin(angle_theta),
        np.cos(angle_phi),
    ], dtype=np.float32)
    position = drone_pos + distance * direction

    if gust_field.storm_mode:
        intensity = float(generator.uniform(30.0, 70.0))
        lifetime = float(generator.uniform(1.5, 4.0))
    else:
        intensity = float(generator.uniform(15.0, 40.0))
        lifetime = float(generator.uniform(0.5, 2.0))

    pulse_freq = float(generator.uniform(2.0, 8.0))
    gust_field.bursts.append({
        "position": position.astype(np.float32),
        "intensity": intensity,
        "lifetime": lifetime,
        "pulse_freq": pulse_freq,
    })


def _sample_physics_batch(
    generator: np.random.Generator,
    quad: QuadcopterDynamics,
    config: TrainingConfig,
) -> BatchDict:
    gust_features: List[np.ndarray] = []
    trunk_features: List[np.ndarray] = []
    disturbances: List[np.ndarray] = []
    accelerations: List[np.ndarray] = []
    controls: List[np.ndarray] = []
    velocities: List[np.ndarray] = []

    for _ in range(config.batch_size):
        drone_pos = generator.normal(size=3).astype(np.float32) * config.position_scale
        drone_vel = generator.normal(size=3).astype(np.float32) * config.velocity_scale
        state = np.concatenate([drone_pos, drone_vel], axis=0).astype(np.float32)

        storm_mode = bool(generator.random() < config.storm_probability)
        gust_field = GustField(max_bursts=config.max_bursts, storm_mode=storm_mode)
        max_burst_count = max(1, gust_field.max_bursts)
        n_bursts = int(generator.integers(0, max_burst_count))
        for _ in range(n_bursts):
            _spawn_burst(generator, gust_field, drone_pos)

        t = float(generator.uniform(0.0, config.time_horizon))
        disturbance = gust_field.get_force(drone_pos, t).astype(np.float32)
        control = generator.normal(size=3).astype(np.float32) * config.control_scale

        gravity = np.array([0.0, 0.0, -quad.mass * quad.g], dtype=np.float32)
        drag = -quad.drag * drone_vel
        acceleration = (control + disturbance + drag + gravity) / quad.mass

        gust_vec = encode_gust_features(drone_pos, gust_field, t, max_bursts=config.max_encoded_bursts)
        trunk_vec = encode_trunk_vec(state, gust_field, t)

        gust_features.append(gust_vec.astype(np.float32))
        trunk_features.append(trunk_vec.astype(np.float32))
        disturbances.append(disturbance)
        accelerations.append(acceleration.astype(np.float32))
        controls.append(control.astype(np.float32))
        velocities.append(drone_vel.astype(np.float32))

    def _stack(values: List[np.ndarray]) -> jnp.ndarray:
        return jnp.asarray(np.stack(values, axis=0), dtype=jnp.float32)

    return {
        "gust_features": _stack(gust_features),
        "trunk_features": _stack(trunk_features),
        "disturbance": _stack(disturbances),
        "acceleration": _stack(accelerations),
        "control": _stack(controls),
        "velocity": _stack(velocities),
    }


def _make_loss_fn(model, lambda_dyn: float, quad: QuadcopterDynamics):
    mass = jnp.asarray(quad.mass, dtype=jnp.float32)
    drag = jnp.asarray(quad.drag, dtype=jnp.float32)
    gravity_vec = jnp.array([0.0, 0.0, mass * quad.g], dtype=jnp.float32)
    lambda_dyn = jnp.asarray(lambda_dyn, dtype=jnp.float32)

    def loss_fn(weights: Sequence[jnp.ndarray], batch: BatchDict):
        preds = model((batch["gust_features"], batch["trunk_features"]), weights)
        err = preds - batch["disturbance"]
        mse = jnp.mean(jnp.square(err))

        physics_residual = (
            mass * batch["acceleration"]
            - (batch["control"] - gravity_vec - drag * batch["velocity"])
        )
        dyn_loss = jnp.mean(jnp.square(preds - physics_residual))
        total = mse + lambda_dyn * dyn_loss
        return total, (mse, dyn_loss)

    return loss_fn


def train_single_run(
    method: str,
    *,
    config: TrainingConfig,
    deeponet_config: Optional[DeepONetConfig] = None,
    learning_rate: Optional[float] = None,
    rng_key: Optional[jax.Array] = None,
    save_path: Optional[PathLike] = None,
) -> TrainingResult:
    if method not in METHOD_CHOICES:
        raise ValueError(f"Unknown training method: {method}")

    deeponet_config = deeponet_config or DeepONetConfig()
    learning_rate = float(learning_rate if learning_rate is not None else config.learning_rate)

    if rng_key is None:
        rng_key = jax.random.PRNGKey(config.seed)
    key_init, key_data = jax.random.split(rng_key)
    numpy_seed = int(jax.random.randint(key_data, shape=(), minval=0, maxval=2**31 - 1))
    generator = np.random.default_rng(numpy_seed)

    quad = QuadcopterDynamics(mass=1.0, drag_coeff=0.1)

    deeponet = DeepONet(deeponet_config)
    weights = deeponet.initialize(key_init)
    model = deeponet.model

    loss_fn = _make_loss_fn(model, config.lambda_dyn, quad)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    dual_state = model.init_dual_state(weights) if method == "manifold_online" else None

    loss_history: List[float] = []
    mse_history: List[float] = []
    dyn_history: List[float] = []

    progress = trange(config.steps, leave=False, desc=f"{method} lr={learning_rate:.3g}")
    for step in progress:
        batch = _sample_physics_batch(generator, quad, config)
        (loss_value, (mse_value, dyn_value)), grad_weights = loss_and_grad(weights, batch)

        loss_scalar = float(loss_value)
        mse_scalar = float(mse_value)
        dyn_scalar = float(dyn_value)

        loss_history.append(loss_scalar)
        mse_history.append(mse_scalar)
        dyn_history.append(dyn_scalar)

        grad_list = [jnp.asarray(g) for g in grad_weights]

        if method == "descent":
            weights = [w - learning_rate * g for w, g in zip(weights, grad_list)]
        elif method == "dualize":
            directions = model.dualize(grad_list, target_norm=config.target_norm)
            weights = [w - learning_rate * direction for w, direction in zip(weights, directions)]
        elif method == "manifold_online":
            tangents, dual_state = model.online_dual_ascent(
                dual_state,
                weights,
                grad_list,
                target_norm=config.target_norm,
                alpha=config.dual_alpha,
                beta=config.dual_beta,
            )
            weights = [w - learning_rate * tangent for w, tangent in zip(weights, tangents)]
            weights = model.retract(weights)
        else:  # pragma: no cover - guarded above
            raise ValueError(f"Unhandled method: {method}")

        if config.log_interval and (step % config.log_interval == 0 or step == config.steps - 1):
            progress.set_postfix({"loss": f"{loss_scalar:.4e}", "mse": f"{mse_scalar:.4e}", "dyn": f"{dyn_scalar:.4e}"})

    deeponet.set_weights(weights)
    if save_path is not None:
        deeponet.save_weights(save_path, weights)

    return TrainingResult(
        method=method,
        learning_rate=learning_rate,
        config=config,
        deeponet_config=deeponet_config,
        loss_history=loss_history,
        mse_history=mse_history,
        dyn_history=dyn_history,
        weights=weights,
    )


def run_method_sweep(
    methods: Sequence[str],
    learning_rates: Sequence[float],
    *,
    config: TrainingConfig,
    deeponet_config: Optional[DeepONetConfig] = None,
    base_seed: Optional[int] = None,
    save_directory: Optional[PathLike] = None,
) -> Dict[str, List[TrainingResult]]:
    deeponet_config = deeponet_config or DeepONetConfig()
    base_seed = config.seed if base_seed is None else base_seed

    results: Dict[str, List[TrainingResult]] = {method: [] for method in methods}
    base_key = jax.random.PRNGKey(base_seed)

    for method_idx, method in enumerate(methods):
        method_key = jax.random.fold_in(base_key, method_idx)
        for lr_idx, lr in enumerate(learning_rates):
            run_key = jax.random.fold_in(method_key, lr_idx)
            save_path = None
            if save_directory is not None:
                save_dir = Path(save_directory)
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"deeponet_{method}_lr{lr:.3g}.npz"

            result = train_single_run(
                method,
                config=config,
                deeponet_config=deeponet_config,
                learning_rate=lr,
                rng_key=run_key,
                save_path=save_path,
            )
            results[method].append(result)

    return results
