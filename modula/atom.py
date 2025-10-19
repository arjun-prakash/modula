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
        self.mass = 1 
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
        shape = (self.d_in, self.d_out, self.k, self.k)
        weight = jax.random.normal(key, shape=shape)

        vectorized_ortho = jax.vmap(
            jax.vmap(orthogonalize, in_axes=2, out_axes=2),
            in_axes=3, 
            out_axes=3
        )

        # Mod. norm p17: scale factor for normalized weights
        scale_factor = jnp.sqrt(self.d_out / self.d_in) / self.k

        weight = vectorized_ortho(weight) * scale_factor

        # Permute weights to [k, k, d_in, d_out] for jax conv
        weight_permuted = jnp.transpose(weight, (2, 3, 0, 1))

        return [weight_permuted]

    def project(self, w):
        weight = w[0]  # shape is [k, k, d_in, d_out]
        
        # Define a function to apply to each spatial slice
        def project_slice(g):
            return orthogonalize(g) * jnp.sqrt(self.d_out / self.d_in) / (self.k**2)
        
        weight_transposed = jnp.transpose(weight, (2, 3, 0, 1))
        
        vectorized_project = jax.vmap(jax.vmap(project_slice, in_axes=2, out_axes=2), in_axes=3, out_axes=3)
        
        weight_projected = vectorized_project(weight_transposed)
        weight_projected = jnp.transpose(weight_projected, (2, 3, 0, 1))
        
        return [weight_projected]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]  # shape is [k, k, d_in, d_out]
        
        # Transpose to [d_in, d_out, k, k] for orthogonalization
        grad_transposed = jnp.transpose(grad, (2, 3, 0, 1))
        
        # Modular duality P7, example 7
        # Define a function to apply to each spatial slice
        def dualize_slice(g):
            return orthogonalize(g) * jnp.sqrt(self.d_out / self.d_in) / (self.k**2) * target_norm

        # Create a vectorized version that maps over spatial dimensions
        vectorized_dualize = jax.vmap(jax.vmap(dualize_slice, in_axes=2, out_axes=2), in_axes=3, out_axes=3)

        # Apply dualization and transpose back to [k, k, d_in, d_out]
        d_weight = vectorized_dualize(grad_transposed)
        d_weight = jnp.transpose(d_weight, (2, 3, 0, 1))
        
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
