import functools
from typing import Tuple, Union

import jax
import jax.numpy as jnp

from modula.abstract import Atom

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
        (float(a) / 1.01, float(b) / 1.01**3, float(c) / 1.01**5) if idx < len(_ABC_LIST) - 1 else (float(a), float(b), float(c))
        for idx, (a, b, c) in enumerate(_ABC_LIST)
    ]
else:
    ABC_LIST_STABLE = [tuple(float(x) for x in coeff) for coeff in _ABC_LIST]

Array = jnp.ndarray



@functools.partial(jax.jit, static_argnames=('steps',))
def matrix_sign(matrix: Array, *, steps: int = 10) -> Array:
    """Return the matrix sign using the Polar Express polynomial iteration."""
    transpose = matrix.shape[-2] > matrix.shape[-1]
    if transpose:
        matrix = jnp.swapaxes(matrix, -1, -2)

    norm = jnp.linalg.norm(matrix, ord='fro')
    norm = jnp.where(norm == 0, 1.0, norm)
    x = (matrix / (norm * 1.01)).astype(jnp.bfloat16)
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

    if transpose:
        x = jnp.swapaxes(x, -1, -2)
    x = jnp.nan_to_num(x.astype(matrix.dtype))
    return x


@functools.partial(jax.jit, static_argnames=('steps', 'tol'))
def _dual_ascent(
    w: Array,
    g: Array,
    lambda_init: Array,
    alpha: float,
    steps: int,
    tol: float,
) -> Array:
    # (PyTorch line: "Initialize the dual variable")
    # In this JAX helper, the initial dual variable Λ₀ is passed in as `lambda_init`.
    total_steps = max(int(steps), 0)
    denom = float(max(total_steps, 1))

    # (PyTorch line: "Ascend on the dual problem to find the update direction A")
    def body_fn(step, state):
        lam, converged = state

        # (PyTorch line: "Update the candidate direction A")
        # A = msign(G + 2 * W @ Lambda)
        a = matrix_sign(g + 2.0 * w @ lam)

        # (PyTorch line: "Measure deviation of A from the tangent space:")
        # H = W.T @ A + A.T @ W
        h = w.T @ a + a.T @ w

        # (PyTorch line: "Check the stopping criterion")
        # if torch.norm(H) / math.sqrt(H.numel()) < tol: break
        # JAX loops are fixed-iteration; we emulate 'break' by freezing updates once converged.
        denom_norm = jnp.sqrt(jnp.asarray(h.size, dtype=w.dtype))
        mean_norm = jnp.linalg.norm(h) / denom_norm

        # (Implements the same linear schedule α * (1 - step / steps))
        step_float = jnp.asarray(step, dtype=w.dtype)
        step_scale = alpha * (1.0 - step_float / denom)
        zero = jnp.zeros([], dtype=w.dtype)
        step_scale = jnp.where(total_steps == 0, zero, step_scale)

        # (PyTorch line: "Update the dual variable")
        # Lambda -= alpha * (1 - step / steps) * H
        lam_next = lam - step_scale * h

        # Freeze Λ after convergence (emulates the 'break' in PyTorch)
        lam_next = jnp.where(converged, lam, lam_next)
        converged_next = converged | (mean_norm < tol)

        return lam_next, converged_next

    lam_final, _ = jax.lax.fori_loop(0, total_steps, body_fn, (lambda_init, False))

    # (PyTorch flow next would be:)
    # "Descend on the primal problem" and "Retract to the manifold"
    # This helper only returns A(Λ_final) so the caller can perform:
    # new_W = W - eta * A; new_W = msign(new_W)
    return matrix_sign(g + 2.0 * w @ lam_final)



@jax.jit
def _online_dual_ascent_step(
    weight: Array,
    grad: Array,
    lam: Array,
    vel: Array,
    *,
    target_norm: Array,
    alpha: Array,
    beta: Array,
) -> Tuple[Array, Array, Array]:
    lam = 0.5 * (lam + lam.T)
    vel = 0.5 * (vel + vel.T)

    lam_tilde = lam + beta * vel
    argument = grad + 2.0 * weight @ lam_tilde
    tangent = matrix_sign(argument)
    tangent = target_norm * tangent

    h = weight.T @ tangent + tangent.T @ weight
    h = 0.5 * (h + h.T)
    vel_next = beta * vel - alpha * h
    lam_next = lam + vel_next

    lam_next = 0.5 * (lam_next + lam_next.T)
    vel_next = 0.5 * (vel_next + vel_next.T)

    return tangent, lam_next, vel_next


def orthogonalize(M):
    # six step Newton-Schulz by @YouJiacheng
    # coefficients from: https://twitter.com/YouJiacheng/status/1893704552689303901
    # found by optimization: https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b/5bff1f7781cf7d062a155eecd2f13075756482ae
    # the idea of stability loss was from @leloykun

    abc_list = [
        (3955/1024, -8306/1024, 5008/1024),
        (3735/1024, -6681/1024, 3463/1024),
        (3799/1024, -6499/1024, 3211/1024),
        (4019/1024, -6385/1024, 2906/1024),
        (2677/1024, -3029/1024, 1162/1024),
        (2172/1024, -1833/1024,  682/1024)
    ]

    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    M = M / jnp.linalg.norm(M)
    for a, b, c in abc_list:
        A = M.T @ M
        I = jnp.eye(A.shape[0])
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T
    return M


class Linear(Atom):
    def __init__(self, fanout, fanin):
        super().__init__()
        self.fanin  = fanin
        self.fanout = fanout
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

    def forward(self, x, w):
        # x shape is [..., fanin]
        weights = w[0]  # shape is [fanout, fanin]
        return jnp.einsum("...ij,...j->...i", weights, x)

    def initialize(self, key):
        weight = jax.random.normal(key, shape=(self.fanout, self.fanin))
        weight = orthogonalize(weight) * jnp.sqrt(self.fanout / self.fanin)
        return [weight]

    def project(self, w):
        weight = w[0]
        weight = orthogonalize(weight) * jnp.sqrt(self.fanout / self.fanin)
        return [weight]

    def retract(self, w):
        weight = w[0]
        weight = matrix_sign(weight)
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        d_weight = orthogonalize(grad) * jnp.sqrt(self.fanout / self.fanin) * target_norm
        return [d_weight]

    def dual_ascent(self, w, grad_w, target_norm=1.0):
        weight = w[0]
        grad = grad_w[0]

        alpha =  0.01
        steps =  100
        tol = 1e-6

        transpose = weight.shape[0] < weight.shape[1]
        if transpose:
            weight_t = weight.T
            grad_t = grad.T
        else:
            weight_t = weight
            grad_t = grad

        lambda_init = -0.25 * (weight_t.T @ grad_t + grad_t.T @ weight_t)
        tangent = _dual_ascent(
            weight_t,
            grad_t,
            lambda_init,
            alpha=alpha,
            steps=steps,
            tol=tol,
        )

        if transpose:
            tangent = tangent.T

        return [tangent]

    def init_dual_state(self, w):
        weight = w[0]
        transpose = weight.shape[0] < weight.shape[1]
        weight_t = weight.T if transpose else weight
        dim = weight_t.shape[1]
        lam0 = jnp.zeros((dim, dim), dtype=weight.dtype)
        vel0 = jnp.zeros_like(lam0)
        return [(lam0, vel0)]

    def online_dual_ascent(self, state, w, grad_w, *, target_norm=1.0, alpha=1e-2, beta=0.9):
        weight = w[0]
        grad = grad_w[0]
        transpose = weight.shape[0] < weight.shape[1]
        weight_t = weight.T if transpose else weight
        grad_t = grad.T if transpose else grad

        if not state:
            lam, vel = self.init_dual_state(w)[0]
        else:
            lam, vel = state[0]

        alpha = jnp.asarray(alpha, dtype=weight.dtype)
        beta = jnp.asarray(beta, dtype=weight.dtype)
        target_norm = jnp.asarray(target_norm, dtype=weight.dtype)

        tangent_t, lam_next, vel_next = _online_dual_ascent_step(
            weight_t,
            grad_t,
            lam,
            vel,
            target_norm=target_norm,
            alpha=alpha,
            beta=beta,
        )

        tangent = tangent_t.T if transpose else tangent_t

        return [tangent], [(lam_next, vel_next)]


class ProbDist(Linear):
    def retract(self, w):
        weight = w[0]
        weight = jax.nn.softmax(weight, axis=-1)
        return [weight]


class Embed(Atom):
    def __init__(self, d_embed, num_embed):
        super().__init__()
        self.num_embed = num_embed
        self.d_embed = d_embed
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

    def forward(self, x, w):
        weights = w[0]  # shape [num_embed, d_embed]
        return weights[x]

    def initialize(self, key):
        weight = jax.random.normal(key, shape=(self.num_embed, self.d_embed))
        weight = weight / jnp.linalg.norm(weight, axis=1, keepdims=True) * jnp.sqrt(self.d_embed)
        return [weight]

    def project(self, w):
        weight = w[0]
        weight = weight / jnp.linalg.norm(weight, axis=1, keepdims=True) * jnp.sqrt(self.d_embed)
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        d_weight = grad / jnp.linalg.norm(grad, axis=1, keepdims=True) * jnp.sqrt(self.d_embed) * target_norm
        d_weight = jnp.nan_to_num(d_weight)
        return [d_weight]


if __name__ == "__main__":

    key = jax.random.PRNGKey(0)

    # sample a random d0xd1 matrix
    d0, d1 = 50, 100
    M = jax.random.normal(key, shape=(d0, d1))
    O = orthogonalize(M)

    # compute SVD of M and O
    U, S, Vh = jnp.linalg.svd(M, full_matrices=False)
    s = jnp.linalg.svd(O, compute_uv=False)

    # print singular values
    print(f"min singular value of O: {jnp.min(s)}")
    print(f"max singular value of O: {jnp.max(s)}")

    print(f"min singular value of M: {jnp.min(S)}")
    print(f"max singular value of M: {jnp.max(S)}")

    # check that M is close to its SVD
    error_M = jnp.linalg.norm(M - U @ jnp.diag(S) @ Vh) / jnp.linalg.norm(M)
    error_O = jnp.linalg.norm(O - U @ Vh) / jnp.linalg.norm(U @ Vh)
    print(f"relative error in M's SVD: {error_M}")
    print(f"relative error in O: {error_O}")
