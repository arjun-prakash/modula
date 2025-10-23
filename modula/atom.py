import jax
import jax.numpy as jnp

from modula.abstract import Atom

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

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        d_weight = orthogonalize(grad) * jnp.sqrt(self.fanout / self.fanin) * target_norm
        return [d_weight]


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
    # no stride and padding for simplicity
    def __init__(self, d_in, d_out, kernel_size):
        super().__init__()
        self.d_in  = d_in
        self.d_out = d_out
        self.k = kernel_size # add kernel size
        self.smooth = True
        self.mass = 1 # based on paper, this is hyperparameter, but kept as 1 in consistency w. Linear
        self.sensitivity = 1

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
        # Create weights directly in [k, k, d_in, d_out] format
        shape = (self.k, self.k, self.d_in, self.d_out)
        weight = jax.random.normal(key, shape=shape)

        # Orthogonalize over [d_in, d_out] at each [k, k] position
        # Apply orthogonalize to axis (2, 3) which are [d_in, d_out]
        def ortho_at_position(w_slice):
            # w_slice shape: [d_in, d_out]
            return orthogonalize(w_slice)
        
        # Vectorize over the spatial dimensions (axes 0 and 1)
        vectorized_ortho = jax.vmap(jax.vmap(ortho_at_position, in_axes=0, out_axes=0), in_axes=0, out_axes=0)
        
        # Mod. norm p17: scale factor for normalized weights
        scale_factor = jnp.sqrt(self.d_out / self.d_in) / (self.k**2)
        
        weight = vectorized_ortho(weight) * scale_factor

        return [weight]

    def project(self, w):
        weight = w[0]  # shape is [k, k, d_in, d_out]
        
        # Define a function to apply at each spatial position
        def project_at_position(w_slice):
            # w_slice shape: [d_in, d_out]
            return orthogonalize(w_slice) * jnp.sqrt(self.d_out / self.d_in) / (self.k**2)
        
        # Vectorize over spatial dimensions
        vectorized_project = jax.vmap(jax.vmap(project_at_position, in_axes=0, out_axes=0), in_axes=0, out_axes=0)
        
        return [vectorized_project(weight)]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]  # shape is [k, k, d_in, d_out]
        
        # Define a function to apply at each spatial position
        def dualize_at_position(g_slice):
            # g_slice shape: [d_in, d_out]
            return orthogonalize(g_slice) * jnp.sqrt(self.d_out / self.d_in) / (self.k**2) * target_norm

        # Vectorize over spatial dimensions
        vectorized_dualize = jax.vmap(jax.vmap(dualize_at_position, in_axes=0, out_axes=0), in_axes=0, out_axes=0)
        
        return [vectorized_dualize(grad)]

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
