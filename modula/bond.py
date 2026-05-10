import jax
import jax.numpy as jnp

from modula.abstract import Bond

class ReLU(Bond):
    def __init__(self):
        super().__init__()
        self.smooth = False
        self.sensitivity = 1

    def forward(self, x, w):
        return jnp.maximum(0, x)


class LeakyReLU(Bond):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.smooth = False
        self.sensitivity = 1
        self.negative_slope = negative_slope

    def forward(self, x, w):
        return jnp.where(x >= 0, x, self.negative_slope * x)


class GeLU(Bond):
    def __init__(self):
        super().__init__()
        self.smooth = False
        self.sensitivity = 1

    def forward(self, x, w):
        return jax.nn.gelu(x) / 1.1289  # 1.1289 is the max derivative of gelu(x)

class Flatten(Bond):
    """Flattens all non-batch dimensions.
    
    Input shape: [N, ...]
    Output shape: [N, prod(...)]
    """
    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1
    
    def forward(self, x, w):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

class Select(Bond):
    """Select a component from a tuple of inputs."""

    def __init__(self, index: int):
        super().__init__()
        self.index = index
        self.smooth = True
        self.sensitivity = 1

    def forward(self, x, w):
        return x[self.index]


class HadamardProduct(Bond):
    """Elementwise product of two tensors with identical shape."""

    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1

    def forward(self, x, w):
        a, b = x
        return a * b
