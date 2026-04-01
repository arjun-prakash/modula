import functools
from typing import Any, Tuple

import jax
import jax.numpy as jnp

try:
    from manifold_muon.msign import ABC_LIST_STABLE as _ABC_LIST
except ModuleNotFoundError:
    _ABC_LIST = [
        (8.28721201814563, -23.595886519098837, 17.300387312530933),
        (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
        (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
        (3.3184196573706015, -2.488488024314874, 0.51004894012372),
        (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
        (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
        (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
        (1.875, -1.25, 0.375),
    ]

    ABC_LIST_STABLE = [
        (
            float(a) / 1.01,
            float(b) / 1.01**3,
            float(c) / 1.01**5,
        )
        if idx < len(_ABC_LIST) - 1
        else (float(a), float(b), float(c))
        for idx, (a, b, c) in enumerate(_ABC_LIST)
    ]
else:
    ABC_LIST_STABLE = [tuple(float(x) for x in coeff) for coeff in _ABC_LIST]

Array = jnp.ndarray


def orient_tall(matrix: Array) -> Tuple[Array, bool]:
    """Return a matrix with rows >= cols and whether a transpose was applied."""
    transposed = matrix.shape[-2] < matrix.shape[-1]
    if transposed:
        matrix = jnp.swapaxes(matrix, -1, -2)
    return matrix, transposed


def restore_orientation(matrix: Array, transposed: bool) -> Array:
    if transposed:
        matrix = jnp.swapaxes(matrix, -1, -2)
    return matrix


def tangent_constraint_residual(weight: Array, tangent: Array) -> Array:
    residual = weight.T @ tangent + tangent.T @ weight
    denom = jnp.sqrt(jnp.asarray(residual.size, dtype=weight.dtype))
    return jnp.linalg.norm(residual) / denom


@functools.partial(jax.jit, static_argnames=("steps",))
def matrix_sign(matrix: Array, *, steps: int = 10) -> Array:
    """Return the matrix sign using the Polar Express polynomial iteration."""
    transposed = matrix.shape[-2] > matrix.shape[-1]
    matrix_tall = jnp.swapaxes(matrix, -1, -2) if transposed else matrix

    norm = jnp.linalg.norm(matrix_tall, ord="fro")
    norm = jnp.where(norm == 0, 1.0, norm)
    x = matrix_tall / (norm * 1.01)
    eye = jnp.eye(x.shape[-2], dtype=x.dtype)

    for step in range(steps):
        idx = min(step, len(ABC_LIST_STABLE) - 1)
        a, b, c = ABC_LIST_STABLE[idx]
        a = jnp.asarray(a, dtype=x.dtype)
        b = jnp.asarray(b, dtype=x.dtype)
        c = jnp.asarray(c, dtype=x.dtype)

        s = x @ jnp.swapaxes(x, -1, -2)
        y = c * s + b * eye
        y = y @ s
        y = y + a * eye
        x = y @ x

    x = jnp.swapaxes(x, -1, -2) if transposed else x
    return jnp.nan_to_num(x.astype(matrix.dtype))


@functools.partial(jax.jit, static_argnames=("steps", "tol"))
def _dual_ascent_tall(
    weight: Array,
    grad: Array,
    lambda_init: Array,
    alpha: float,
    steps: int,
    tol: float,
) -> Array:
    total_steps = max(int(steps), 0)
    denom = float(max(total_steps, 1))

    def body_fn(step, state):
        lam, converged = state

        tangent = matrix_sign(grad + 2.0 * weight @ lam)
        residual = weight.T @ tangent + tangent.T @ weight
        residual_norm = tangent_constraint_residual(weight, tangent)

        step_float = jnp.asarray(step, dtype=weight.dtype)
        step_scale = alpha * (1.0 - step_float / denom)
        zero = jnp.zeros([], dtype=weight.dtype)
        step_scale = jnp.where(total_steps == 0, zero, step_scale)

        this_converged = residual_norm < tol
        should_update = jnp.logical_not(converged | this_converged)
        lam_candidate = lam - step_scale * residual
        lam_next = jnp.where(should_update, lam_candidate, lam)
        converged_next = converged | this_converged
        return lam_next, converged_next

    lam_final, _ = jax.lax.fori_loop(0, total_steps, body_fn, (lambda_init, jnp.asarray(False)))
    return matrix_sign(grad + 2.0 * weight @ lam_final)


@functools.partial(jax.jit, static_argnames=("steps",))
def _admm_dual_ascent_tall(
    weight: Array,
    grad: Array,
    *,
    steps: int,
    rho: float,
) -> Array:
    total_steps = max(int(steps), 0)
    rho = jnp.asarray(rho, dtype=weight.dtype)
    inv_rho = 1.0 / rho
    inv_rho_sq = inv_rho * inv_rho

    lambda_init = -0.25 * (weight.T @ grad + grad.T @ weight)
    x_init = grad + 2.0 * weight @ lambda_init
    omega_init = jnp.zeros_like(x_init)

    def body_fn(_, state):
        lam, x, omega = state

        tmp = inv_rho * omega + x - grad
        p = weight.T @ tmp
        lam_upd = 0.25 * (p + p.T)

        b = grad + 2.0 * weight @ lam_upd - inv_rho * omega
        eye = jnp.eye(b.shape[-1], dtype=b.dtype)
        p_pos = 0.5 * (eye + matrix_sign(b.T @ b - inv_rho_sq * eye))

        x_upd = (b - inv_rho * matrix_sign(b)) @ p_pos
        omega_upd = omega + rho * (x_upd - 2.0 * weight @ lam_upd - grad)
        return lam_upd, x_upd, omega_upd

    lam_final, _, _ = jax.lax.fori_loop(0, total_steps, body_fn, (lambda_init, x_init, omega_init))
    return matrix_sign(grad + 2.0 * weight @ lam_final)


@jax.jit
def _online_dual_ascent_step_tall(
    weight: Array,
    grad: Array,
    lam: Array,
    vel: Array,
    *,
    alpha: Array,
    beta: Array,
) -> Tuple[Array, Array, Array]:
    lam = 0.5 * (lam + lam.T)
    vel = 0.5 * (vel + vel.T)

    lam_tilde = 0.5 * (lam + beta * vel + (lam + beta * vel).T)
    tangent = matrix_sign(grad + 2.0 * weight @ lam_tilde)

    residual = weight.T @ tangent + tangent.T @ weight
    residual = 0.5 * (residual + residual.T)
    vel_next = beta * vel - alpha * residual
    lam_next = lam + vel_next

    lam_next = 0.5 * (lam_next + lam_next.T)
    vel_next = 0.5 * (vel_next + vel_next.T)
    return tangent, lam_next, vel_next


def dual_ascent_tangent(
    weight: Array,
    grad: Array,
    *,
    alpha: float = 0.01,
    steps: int = 100,
    tol: float = 1e-6,
) -> Array:
    weight_tall, transposed = orient_tall(weight)
    grad_tall = restore_orientation(grad, transposed)
    lambda_init = -0.25 * (weight_tall.T @ grad_tall + grad_tall.T @ weight_tall)
    tangent_tall = _dual_ascent_tall(
        weight_tall,
        grad_tall,
        lambda_init,
        alpha=alpha,
        steps=steps,
        tol=tol,
    )
    return restore_orientation(tangent_tall, transposed)


def admm_dual_ascent_tangent(
    weight: Array,
    grad: Array,
    *,
    steps: int = 10,
    rho: float = 4.0,
) -> Array:
    weight_tall, transposed = orient_tall(weight)
    grad_tall = restore_orientation(grad, transposed)
    tangent_tall = _admm_dual_ascent_tall(
        weight_tall,
        grad_tall,
        steps=steps,
        rho=rho,
    )
    return restore_orientation(tangent_tall, transposed)


def online_dual_ascent_step(
    weight: Array,
    grad: Array,
    lam: Array,
    vel: Array,
    *,
    alpha: Array,
    beta: Array,
) -> Tuple[Array, Array, Array]:
    weight_tall, transposed = orient_tall(weight)
    grad_tall = restore_orientation(grad, transposed)
    tangent_tall, lam_next, vel_next = _online_dual_ascent_step_tall(
        weight_tall,
        grad_tall,
        lam,
        vel,
        alpha=alpha,
        beta=beta,
    )
    return restore_orientation(tangent_tall, transposed), lam_next, vel_next


def orthogonalize(matrix: Array) -> Array:
    """Six-step Newton-Schulz orthogonalization."""
    abc_list = [
        (3955 / 1024, -8306 / 1024, 5008 / 1024),
        (3735 / 1024, -6681 / 1024, 3463 / 1024),
        (3799 / 1024, -6499 / 1024, 3211 / 1024),
        (4019 / 1024, -6385 / 1024, 2906 / 1024),
        (2677 / 1024, -3029 / 1024, 1162 / 1024),
        (2172 / 1024, -1833 / 1024, 682 / 1024),
    ]

    matrix_tall, transposed = orient_tall(matrix)
    matrix_tall = matrix_tall / jnp.linalg.norm(matrix_tall)

    for a, b, c in abc_list:
        gram = matrix_tall.T @ matrix_tall
        eye = jnp.eye(gram.shape[0], dtype=matrix_tall.dtype)
        matrix_tall = matrix_tall @ (a * eye + b * gram + c * gram @ gram)

    return restore_orientation(matrix_tall, transposed)


def dampen_dual_state(state: Any, *, factor: float = 0.25, zero_velocity: bool = True) -> Any:
    """Recursively dampen all `(Lambda, velocity)` pairs in a dual-state pytree."""
    if state is None:
        return None

    def _rec(node):
        if isinstance(node, tuple) and len(node) == 2 and all(hasattr(x, "shape") for x in node):
            lam, vel = node
            lam = factor * lam
            vel = jnp.zeros_like(vel) if zero_velocity else factor * vel
            return (lam, vel)
        if isinstance(node, (list, tuple)):
            return type(node)(_rec(child) for child in node)
        return node

    return _rec(state)


__all__ = [
    "ABC_LIST_STABLE",
    "Array",
    "admm_dual_ascent_tangent",
    "dampen_dual_state",
    "dual_ascent_tangent",
    "matrix_sign",
    "online_dual_ascent_step",
    "orient_tall",
    "orthogonalize",
    "restore_orientation",
    "tangent_constraint_residual",
]
