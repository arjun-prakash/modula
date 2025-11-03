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
    x = (matrix / (norm * 1.01)) #.astype(jnp.bfloat16)
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



@functools.partial(jax.jit, static_argnames=('steps',))
def _admm_dual_ascent(
    w: Array,
    g: Array,
    *,
    steps: int,
    rho: float,
) -> Array:
    rho = jnp.asarray(rho, dtype=w.dtype)
    inv_rho = 1.0 / rho
    inv_rho_sq = inv_rho * inv_rho

    lambda_init = -0.25 * (w.T @ g + g.T @ w)
    x_init = g + 2.0 * w @ lambda_init
    omega_init = jnp.zeros_like(x_init)

    def body_fn(_, state):
        lam, x, omega = state

        tmp = inv_rho * omega + x - g
        p = w.T @ tmp
        lam_upd = 0.25 * (p + p.T)

        b = g + 2.0 * w @ lam_upd - inv_rho * omega
        cols = b.shape[-1]
        eye = jnp.eye(cols, dtype=b.dtype)
        p_pos = 0.5 * (eye + matrix_sign(b.T @ b - inv_rho_sq * eye))

        x_upd = (b - inv_rho * matrix_sign(b)) @ p_pos
        omega_upd = omega + rho * (x_upd - 2.0 * w @ lam_upd - g)

        return lam_upd, x_upd, omega_upd

    lam_final, _, _ = jax.lax.fori_loop(0, steps, body_fn, (lambda_init, x_init, omega_init))
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
        weight = matrix_sign(weight) #should this be scaled? * jnp.sqrt(self.fanout / self.fanin)
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

    def admm_dual_ascent(self, w, grad_w, *, target_norm=1.0, steps=10, rho=4.0):
        weight = w[0]
        grad = grad_w[0]

        transpose = weight.shape[0] < weight.shape[1]
        if transpose:
            weight_t = weight.T
            grad_t = grad.T
        else:
            weight_t = weight
            grad_t = grad

        tangent_t = _admm_dual_ascent(
            weight_t,
            grad_t,
            steps=steps,
            rho=rho,
        )

        if transpose:
            tangent_t = tangent_t.T

        target_norm = jnp.asarray(target_norm, dtype=weight.dtype)
        tangent = target_norm * tangent_t
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


from typing import Any
def dampen_dual_state(state: Any, *, factor: float = 0.25, zero_velocity: bool = True) -> Any:
        """
        Recursively damp the dual state after a retraction.

        - For (Λ, V) tuples (Linear: [ (Λ, V) ], Conv2D: [ (Λ, V) ]):
            Λ ← factor · Λ
            V ← 0            if zero_velocity else factor · V
        - Leaves non-(Λ,V) leaves (e.g., Bias state vectors) unchanged.
        - Handles arbitrarily nested [list]/(tuple) structures returned by your model.
        - Safe on None: returns None.

        Args:
            state: the dual_state structure returned by model.online_dual_ascent(...)
            factor: multiplicative damping for Λ (e.g., 0.25)
            zero_velocity: if True, reset V to zeros; otherwise scale V by `factor`.

        Returns:
            A new dual_state with damping applied.
        """
        if state is None:
            return None

        def _rec(s):
            # Detect a (Λ, V) pair
            if isinstance(s, tuple) and len(s) == 2 and all(hasattr(x, "shape") for x in s):
                lam, vel = s
                lam = factor * lam
                vel = jnp.zeros_like(vel) if zero_velocity else factor * vel
                return (lam, vel)

            # Recurse through lists/tuples of children
            if isinstance(s, (list, tuple)):
                return type(s)(_rec(x) for x in s)

            # Anything else (e.g., Bias state tensor) -> leave unchanged
            return s

        return _rec(state)

class Conv2D(Atom):
    # no stride and padding for simplicity
    def __init__(self, d_in, d_out, kernel_size, retract_enabled: bool = True):
        super().__init__()
        self.d_in  = d_in
        self.d_out = d_out
        self.k = kernel_size # add kernel size
        self.smooth = True
        self.mass = 1 
        self.sensitivity = 1
        self.retract_enabled = retract_enabled

    def _scale(self, target_norm: float = 1.0):
        # same factor you already use, just centralized
        return jnp.sqrt(self.d_out / self.d_in) / (self.k ** 2) * target_norm

    def _flatten_kernel(self, kernel: Array) -> Array:
        """Reshape [k, k, d_in, d_out] into [(k*k*d_in), d_out]."""
        return kernel.reshape(self.k * self.k * self.d_in, self.d_out)

    def _reshape_kernel(self, matrix: Array) -> Array:
        """Inverse of _flatten_kernel."""
        return matrix.reshape(self.k, self.k, self.d_in, self.d_out)

    def _project_flat(self, matrix: Array) -> Array:
        """Apply matrix_sign in the appropriate orientation for flattened kernels."""
        transpose = matrix.shape[0] < matrix.shape[1]
        mat = matrix.T if transpose else matrix
        proj = matrix_sign(mat)
        return proj.T if transpose else proj

    


    def forward(self, x, w):
        # x shape is [N, H, W, C]
        weights = w[0]  # shape is [k, k, d_in, d_out]

        return jax.lax.conv_general_dilated(
            lhs=x,
            rhs=weights,
            window_strides=(1,1),
            padding="SAME",
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

    def initialize(self, key):
        shape = (self.k, self.k, self.d_in, self.d_out)           # [k, k, d_in, d_out]
        weight = jax.random.normal(key, shape=shape)
        weight_flat = self._flatten_kernel(weight)
        weight_proj = self._project_flat(weight_flat)
        return [self._scale() * self._reshape_kernel(weight_proj)]

    def project(self, w):
        weight = w[0]                                              # [k, k, d_in, d_out]
        weight_flat = self._flatten_kernel(weight)
        weight_proj = self._project_flat(weight_flat)
        return [self._scale() * self._reshape_kernel(weight_proj)]

    # --- dualize: flattened msign(grad) + fixed scale (times target_norm) ---
    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]                                           # [k, k, d_in, d_out]
        grad_flat = self._flatten_kernel(grad)
        out_flat = matrix_sign(grad_flat)
        out = self._reshape_kernel(out_flat)
        return [self._scale(target_norm) * out]

    def retract(self, w):
        if not self.retract_enabled:
            return w  # no-op

        W = w[0]  # [k,k,d_in,d_out]
        W_flat = self._flatten_kernel(W)
        W_proj = self._project_flat(W_flat)
        W_ret = self._reshape_kernel(W_proj)
        # if you also carry a gain parameter (W, g), keep g unchanged
        return [W_ret] if len(w) == 1 else [W_ret, *w[1:]]


    def init_dual_state(self, w):
        # Λ lives in the smaller side’s space so the matmul contracts correctly
        rows = self.k * self.k * self.d_in
        cols = self.d_out
        dim = rows if rows < cols else cols
        dtype = w[0].dtype if w else jnp.float32
        lam0 = jnp.zeros((dim, dim), dtype=dtype)
        vel0 = jnp.zeros_like(lam0)
        return [(lam0, vel0)]



    def online_dual_ascent(
        self, state, w, grad_w, *, target_norm: float = 1.0, alpha: float = 1e-2, beta: float = 0.9
    ):
        W = w[0]      # [k, k, d_in, d_out]
        G = grad_w[0] # [k, k, d_in, d_out]
        if not state:
            Λ, V = self.init_dual_state(w)[0]
        else:
            Λ, V = state[0]

        alpha = jnp.asarray(alpha, dtype=W.dtype)
        beta  = jnp.asarray(beta,  dtype=W.dtype)
        target_scale = jnp.asarray(self._scale(target_norm), dtype=W.dtype)

        W_flat = self._flatten_kernel(W)
        G_flat = self._flatten_kernel(G)

        transpose = W_flat.shape[0] < W_flat.shape[1]
        W_t = W_flat.T if transpose else W_flat
        G_t = G_flat.T if transpose else G_flat

        tangent_t, Λn, Vn = _online_dual_ascent_step(
            W_t, G_t, Λ, V, target_norm=target_scale, alpha=alpha, beta=beta
        )
        tangent_flat = tangent_t.T if transpose else tangent_t
        tangent = self._reshape_kernel(tangent_flat)

        return [tangent], [(Λn, Vn)]



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
