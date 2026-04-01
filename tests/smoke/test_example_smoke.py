import importlib.util
import sys
import unittest
from pathlib import Path

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"


def load_example_module(module_name: str):
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    module_path = EXAMPLES_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"smoke_{module_name}", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAS_JAX, "requires JAX and modula dependencies")
class ExampleSmokeTest(unittest.TestCase):
    def test_mnist_one_step_manifold_variants(self):
        module = load_example_module("mnist")
        model = module.build_model(4, 2, 3)
        key = jax.random.PRNGKey(0)
        inputs = jax.random.normal(jax.random.fold_in(key, 1), (4, 4))
        targets = jax.random.normal(jax.random.fold_in(key, 2), (4, 2))

        for method in ("manifold", "manifold_admm"):
            weights, loss = module.train_single_run(
                model,
                key,
                method,
                1e-2,
                1,
                2,
                7.0,
                inputs,
                targets,
            )
            self.assertTrue(jnp.isfinite(loss))
            self.assertTrue(all(jnp.all(jnp.isfinite(weight)) for weight in weights))

    def test_cnn_mnist_one_step_online(self):
        module = load_example_module("cnn_mnist")
        model = module.build_model(28 * 28, 10, 16)
        key = jax.random.PRNGKey(1)
        inputs = jax.random.normal(jax.random.fold_in(key, 1), (4, 28, 28, 1))
        targets = jax.random.normal(jax.random.fold_in(key, 2), (4, 10))

        weights, loss = module.train_single_run(
            model,
            key,
            "manifold_online",
            1e-2,
            1,
            2,
            11.0,
            inputs,
            targets,
        )
        self.assertTrue(jnp.isfinite(loss))
        self.assertTrue(all(jnp.all(jnp.isfinite(weight)) for weight in weights))


if __name__ == "__main__":
    unittest.main()
