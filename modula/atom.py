import jax
import jax.numpy as jnp

from modula.abstract import Atom
from modula.manifold import (
    Array,
    admm_dual_ascent_tangent,
    dampen_dual_state,
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


class BatchNorm2D(Atom):
    """Batch normalization for 2D feature maps with constrained scale parameter.
    
    Scale parameter γ is constrained to have norm ≤ 1 to maintain Lipschitz constraint.
    Input shape: [N, H, W, C]
    
    Weight format: Single array [scale, shift] concatenated, shape [2, C]
    """
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1
        
        # Running statistics (not trainable)
        self.running_mean = None
        self.running_var = None
        self.num_batches_tracked = 0

    def forward(self, x, w):
        # x shape: [N, H, W, C]
        # w[0] shape: [2, C] where w[0][0] is scale, w[0][1] is shift
        params = w[0]  # [2, C]
        scale = params[0]  # [C]
        shift = params[1]  # [C]
        
        # Compute batch statistics
        mean = jnp.mean(x, axis=(0, 1, 2), keepdims=True)  # [1, 1, 1, C]
        var = jnp.var(x, axis=(0, 1, 2), keepdims=True)    # [1, 1, 1, C]
        
        # Normalize
        x_norm = (x - mean) / jnp.sqrt(var + self.eps)
        
        # Scale and shift (broadcast over batch and spatial dimensions)
        return scale[None, None, None, :] * x_norm + shift[None, None, None, :]

    def initialize(self, key):
        # Initialize as [2, C] array
        params = jnp.stack([
            jnp.ones((self.num_features,), dtype=jnp.float32) * 0.1,  # scale
            jnp.zeros((self.num_features,), dtype=jnp.float32)  # shift
        ])
        return [params]

    def project(self, w):
        params = w[0]  # [2, C]
        scale = params[0]
        shift = params[1]
        
        # Constrain scale to have norm ≤ 1
        scale_norm = jnp.linalg.norm(scale)
        scale = jnp.where(scale_norm > 1.0, scale / scale_norm, scale)
        
        return [jnp.stack([scale, shift])]

    def retract(self, w):
        # Same as project - constrain scale norm
        return self.project(w)

    def dualize(self, grad_w, target_norm=1.0):
        grad_params = grad_w[0]  # [2, C]
        grad_scale = grad_params[0]
        grad_shift = grad_params[1]
        
        # Normalize the full gradient tensor jointly to preserve relative magnitudes
        # This maintains the relationship between scale and shift gradients
        full_grad_norm = jnp.linalg.norm(grad_params)
        normalized_grad = grad_params / (full_grad_norm + 1e-12) * target_norm
        
        return [normalized_grad]

    def dual_ascent(self, w, grad_w, target_norm=1.0):
        return self.dualize(grad_w, target_norm)

    def init_dual_state(self, w):
        # Simple state matching the weight shape
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

class Conv2D(Atom):
    def __init__(self, d_in, d_out, kernel_size, stride=1, retract_enabled: bool = True):
        super().__init__()
        self.d_in  = d_in
        self.d_out = d_out
        self.k = kernel_size
        self.stride = stride
        self.smooth = True
        self.mass = 1 
        self.sensitivity = 1
        self.retract_enabled = retract_enabled


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
            window_strides=(self.stride, self.stride),
            padding="SAME",
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

    def initialize(self, key):
        shape = (self.k, self.k, self.d_in, self.d_out)           # [k, k, d_in, d_out]
        weight = jax.random.normal(key, shape=shape)
        weight_flat = self._flatten_kernel(weight)
        weight_proj = self._project_flat(weight_flat)
        return [self._reshape_kernel(weight_proj)]

    def project(self, w):
        weight = w[0]                                              # [k, k, d_in, d_out]
        weight_flat = self._flatten_kernel(weight)
        weight_proj = self._project_flat(weight_flat)
        return [self._reshape_kernel(weight_proj)]

    # --- dualize: flattened msign(grad), leaving layer-shape scaling to the outer update ---
    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]                                           # [k, k, d_in, d_out]
        grad_flat = self._flatten_kernel(grad)
        out_flat = matrix_sign(grad_flat)
        out = self._reshape_kernel(out_flat)
        return [target_norm * out]

    def dual_ascent(self, w, grad_w, target_norm=1.0):
        weight_flat = self._flatten_kernel(w[0])
        grad_flat = self._flatten_kernel(grad_w[0])
        tangent_flat = dual_ascent_tangent(weight_flat, grad_flat, alpha=0.01, steps=100, tol=1e-6)
        return [self._reshape_kernel(tangent_flat)]

    def admm_dual_ascent(self, w, grad_w, *, target_norm=1.0, steps=10, rho=4.0):
        weight_flat = self._flatten_kernel(w[0])
        grad_flat = self._flatten_kernel(grad_w[0])
        tangent_flat = admm_dual_ascent_tangent(weight_flat, grad_flat, steps=steps, rho=rho)
        return [self._reshape_kernel(tangent_flat)]

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

        W_flat = self._flatten_kernel(W)
        G_flat = self._flatten_kernel(G)

        transpose = W_flat.shape[0] < W_flat.shape[1]
        W_t = W_flat.T if transpose else W_flat
        G_t = G_flat.T if transpose else G_flat

        tangent_t, Λn, Vn = online_dual_ascent_step(W_t, G_t, Λ, V, alpha=alpha, beta=beta)
        tangent_flat = tangent_t.T if transpose else tangent_t
        tangent = self._reshape_kernel(tangent_flat)

        return [tangent], [(Λn, Vn)]


class Conv2DTranspose(Atom):
    """Transposed convolution (deconvolution) for upsampling."""
    
    def __init__(self, d_in, d_out, kernel_size, stride=1, retract_enabled: bool = True, use_weight_norm: bool = False):
        super().__init__()
        self.d_in  = d_in
        self.d_out = d_out
        self.k = kernel_size
        self.stride = stride
        self.smooth = True
        self.mass = 1 
        self.sensitivity = 1
        self.retract_enabled = retract_enabled
        self.use_weight_norm = use_weight_norm


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
        # x shape is [N, H, W, C_in]
        if self.use_weight_norm and len(w) > 1:
            # w = [W, g] where W is the direction and g is the magnitude per output channel
            W = w[0]  # [k, k, d_in, d_out]
            g = w[1]  # [d_out]
            # Normalize per output filter
            W_flat = W.reshape(-1, self.d_out)  # [k*k*d_in, d_out]
            W_norms = jnp.linalg.norm(W_flat, axis=0, keepdims=True)  # [1, d_out]
            # CRITICAL: stop_gradient prevents exploding gradients through normalization
            W_normalized = W_flat / jax.lax.stop_gradient(W_norms + 1e-8)
            W_normalized = W_normalized.reshape(self.k, self.k, self.d_in, self.d_out)
            
            # Apply gain per channel
            weights = W_normalized * g[None, None, None, :]
        else:
            weights = w[0]  # shape is [k, k, d_in, d_out]

        return jax.lax.conv_transpose(
            lhs=x,
            rhs=weights,
            strides=(self.stride, self.stride),
            padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

    def initialize(self, key):
        shape = (self.k, self.k, self.d_in, self.d_out)
        weight = jax.random.normal(key, shape=shape)
        weight_flat = self._flatten_kernel(weight)
        weight_proj = self._project_flat(weight_flat)
        W_init = self._reshape_kernel(weight_proj)
        
        if self.use_weight_norm:
            # Initialize gain per output channel from the projected kernel norm
            W_flat = W_init.reshape(-1, self.d_out)
            g_init = jnp.linalg.norm(W_flat, axis=0) + 1e-8
            return [W_init, g_init]
        return [W_init]

    def project(self, w):
        weight = w[0]
        weight_flat = self._flatten_kernel(weight)
        weight_proj = self._project_flat(weight_flat)
        W_proj = self._reshape_kernel(weight_proj)
        
        if self.use_weight_norm:
            # Keep gain positive and bounded
            g = jnp.maximum(w[1], 1e-8)
            return [W_proj, g]
        return [W_proj]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        grad_flat = self._flatten_kernel(grad)
        out_flat = matrix_sign(grad_flat)
        out = self._reshape_kernel(out_flat)
        d_W = target_norm * out
        
        if self.use_weight_norm and len(grad_w) > 1:
            # For weight norm, compute proper gradient for gain parameter
            # Normalize the gradient by its norm to match target_norm
            grad_g = grad_w[1]
            grad_g_norm = jnp.linalg.norm(grad_g)
            d_g = target_norm * grad_g / (grad_g_norm + 1e-12)
            return [d_W, d_g]
        return [d_W]

    def dual_ascent(self, w, grad_w, target_norm=1.0):
        weight_flat = self._flatten_kernel(w[0])
        grad_flat = self._flatten_kernel(grad_w[0])
        tangent_flat = dual_ascent_tangent(weight_flat, grad_flat, alpha=0.01, steps=100, tol=1e-6)
        tangent = self._reshape_kernel(tangent_flat)

        if self.use_weight_norm and len(grad_w) > 1:
            grad_g = grad_w[1]
            grad_g_norm = jnp.linalg.norm(grad_g)
            tangent_g = grad_g / (grad_g_norm + 1e-12)
            return [tangent, tangent_g]
        return [tangent]

    def admm_dual_ascent(self, w, grad_w, *, target_norm=1.0, steps=10, rho=4.0):
        weight_flat = self._flatten_kernel(w[0])
        grad_flat = self._flatten_kernel(grad_w[0])
        tangent_flat = admm_dual_ascent_tangent(weight_flat, grad_flat, steps=steps, rho=rho)
        tangent = self._reshape_kernel(tangent_flat)

        if self.use_weight_norm and len(grad_w) > 1:
            grad_g = grad_w[1]
            grad_g_norm = jnp.linalg.norm(grad_g)
            tangent_g = grad_g / (grad_g_norm + 1e-12)
            return [tangent, tangent_g]
        return [tangent]

    def retract(self, w):
        if not self.retract_enabled:
            return w

        W = w[0]
        W_flat = self._flatten_kernel(W)
        W_proj = self._project_flat(W_flat)
        W_ret = self._reshape_kernel(W_proj)
        
        if self.use_weight_norm:
            # Keep gain unchanged during retraction
            g = jnp.maximum(w[1], 1e-8)
            return [W_ret, g]
        return [W_ret]

    def init_dual_state(self, w):
        rows = self.k * self.k * self.d_in
        cols = self.d_out
        dim = rows if rows < cols else cols
        dtype = w[0].dtype if w else jnp.float32
        lam0 = jnp.zeros((dim, dim), dtype=dtype)
        vel0 = jnp.zeros_like(lam0)
        
        if self.use_weight_norm:
            # Add state for gain parameter - vector of zeros matching g shape
            g_state = jnp.zeros_like(w[1])
            return [(lam0, vel0), g_state]
        return [(lam0, vel0)]

    def online_dual_ascent(
        self, state, w, grad_w, *, target_norm: float = 1.0, alpha: float = 1e-2, beta: float = 0.9
    ):
        W = w[0]
        G = grad_w[0]
        if not state:
            state_init = self.init_dual_state(w)
            Λ, V = state_init[0]
        else:
            Λ, V = state[0]

        alpha = jnp.asarray(alpha, dtype=W.dtype)
        beta  = jnp.asarray(beta,  dtype=W.dtype)

        W_flat = self._flatten_kernel(W)
        G_flat = self._flatten_kernel(G)

        transpose = W_flat.shape[0] < W_flat.shape[1]
        W_t = W_flat.T if transpose else W_flat
        G_t = G_flat.T if transpose else G_flat

        tangent_t, Λn, Vn = online_dual_ascent_step(W_t, G_t, Λ, V, alpha=alpha, beta=beta)
        tangent_flat = tangent_t.T if transpose else tangent_t
        tangent = self._reshape_kernel(tangent_flat)

        if self.use_weight_norm:
            # Proper gradient handling for gain parameter
            grad_g = grad_w[1]
            grad_g_norm = jnp.linalg.norm(grad_g)
            tangent_g = grad_g / (grad_g_norm + 1e-12)
            g_state_next = state[1] if state else jnp.zeros_like(w[1])
            return [tangent, tangent_g], [(Λn, Vn), g_state_next]
        
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
