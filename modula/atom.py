import jax
import jax.numpy as jnp

from modula.abstract import Atom
from modula.manifold import (
    admm_dual_ascent_tangent,
    dual_ascent_tangent,
    matrix_sign,
    online_dual_ascent_step,
    orthogonalize,
)


class Linear(Atom):
    def __init__(self, fanout, fanin):
        super().__init__()
        self.fanin  = fanin
        self.fanout = fanout
        self.stiefel_radius = 1.0
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

    def forward(self, x, w):
        # x shape is [..., fanin]
        weights = w[0]  # shape is [fanout, fanin]
        return jnp.einsum("...ij,...j->...i", weights, x)

    def initialize(self, key):
        weight = jax.random.normal(key, shape=(self.fanout, self.fanin))
        weight = orthogonalize(weight)
        return [weight]

    def project(self, w):
        weight = w[0]
        weight = orthogonalize(weight)
        return [weight]

    def retract(self, w):
        weight = w[0]
        weight = matrix_sign(weight)
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        d_weight = orthogonalize(grad) * target_norm
        return [d_weight]

    def dual_ascent(self, w, grad_w, target_norm=1.0):
        weight = w[0]
        grad = grad_w[0]
        tangent = dual_ascent_tangent(weight, grad, alpha=0.01, steps=100, tol=1e-6)
        return [tangent]

    def admm_dual_ascent(self, w, grad_w, *, target_norm=1.0, steps=10, rho=4.0):
        weight = w[0]
        grad = grad_w[0]
        tangent = admm_dual_ascent_tangent(weight, grad, steps=steps, rho=rho)
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
        tangent_t, lam_next, vel_next = online_dual_ascent_step(weight_t, grad_t, lam, vel, alpha=alpha, beta=beta)
        tangent = tangent_t.T if transpose else tangent_t
        return [tangent], [(lam_next, vel_next)]


class StandardParamLinear(Linear):
    """Linear layer with standard parametrization initialization."""

    def initialize(self, key):
        scale = jnp.asarray(self.fanin, dtype=jnp.float32) ** -0.5
        weight = scale * jax.random.normal(key, shape=(self.fanout, self.fanin))
        return [weight]

    def project(self, w):
        return [w[0]]

    def retract(self, w):
        return [w[0]]


class RMSRadiusLinear(Linear):
    """Linear layer whose stored Stiefel radius gives unit RMS-to-RMS norm."""

    def __init__(self, fanout, fanin):
        super().__init__(fanout, fanin)
        self.stiefel_radius = float((fanout / fanin) ** 0.5)

    def _radius(self, dtype):
        return jnp.asarray(self.stiefel_radius, dtype=dtype)

    def initialize(self, key):
        weight = jax.random.normal(key, shape=(self.fanout, self.fanin))
        weight = orthogonalize(weight)
        return [self._radius(weight.dtype) * weight]

    def project(self, w):
        weight = w[0]
        weight = orthogonalize(weight)
        return [self._radius(weight.dtype) * weight]

    def retract(self, w):
        weight = w[0]
        weight = matrix_sign(weight)
        return [self._radius(weight.dtype) * weight]

    def dual_ascent(self, w, grad_w, target_norm=1.0):
        weight = w[0]
        grad = grad_w[0]
        unit_weight = weight / self._radius(weight.dtype)
        tangent = dual_ascent_tangent(unit_weight, grad, alpha=0.01, steps=100, tol=1e-6)
        return [tangent]

    def admm_dual_ascent(self, w, grad_w, *, target_norm=1.0, steps=10, rho=4.0):
        weight = w[0]
        grad = grad_w[0]
        unit_weight = weight / self._radius(weight.dtype)
        tangent = admm_dual_ascent_tangent(unit_weight, grad, steps=steps, rho=rho)
        return [tangent]

    def online_dual_ascent(self, state, w, grad_w, *, target_norm=1.0, alpha=1e-2, beta=0.9):
        weight = w[0]
        grad = grad_w[0]
        radius = self._radius(weight.dtype)
        unit_weight = weight / radius
        transpose = unit_weight.shape[0] < unit_weight.shape[1]
        weight_t = unit_weight.T if transpose else unit_weight
        grad_t = grad.T if transpose else grad

        if not state:
            lam, vel = self.init_dual_state(w)[0]
        else:
            lam, vel = state[0]

        alpha = jnp.asarray(alpha, dtype=weight.dtype)
        beta = jnp.asarray(beta, dtype=weight.dtype)
        tangent_t, lam_next, vel_next = online_dual_ascent_step(weight_t, grad_t, lam, vel, alpha=alpha, beta=beta)
        tangent = tangent_t.T if transpose else tangent_t
        return [tangent], [(lam_next, vel_next)]


class Bias(Atom):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

    def forward(self, x, w):
        bias = w[0]
        return x + bias

    def initialize(self, key):
        bias = jnp.zeros((self.dim,), dtype=jnp.float32)
        return [bias]

    def project(self, w):
        return [w[0]]

    def retract(self, w):
        return [w[0]]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        norm = jnp.linalg.norm(grad)
        scale = target_norm / (norm + 1e-12)
        return [grad * scale]

    def dual_ascent(self, w, grad_w, target_norm=1.0):
        return self.dualize(grad_w, target_norm)

    def init_dual_state(self, w):
        return [jnp.zeros_like(w[0])]

    def online_dual_ascent(self, state, w, grad_w, *, target_norm=1.0, alpha=1e-2, beta=0.9):
        tangent = self.dualize(grad_w, target_norm)
        return tangent, (state if state else self.init_dual_state(w))


class ProbDist(Linear):
    def retract(self, w):
        weight = w[0]
        weight = jax.nn.softmax(weight, axis=-1)
        return [weight]


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
