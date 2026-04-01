import unittest

try:
    import jax
    import jax.numpy as jnp

    from modula.atom import Conv2D, Conv2DTranspose, Linear
    from modula.manifold import matrix_sign, tangent_constraint_residual

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


@unittest.skipUnless(HAS_JAX, "requires JAX and modula dependencies")
class ManifoldAtomSmokeTest(unittest.TestCase):
    def _assert_finite_tree(self, tree):
        for leaf in tree:
            self.assertTrue(jnp.all(jnp.isfinite(leaf)))

    def _matrix_residual(self, weight, tangent):
        return float(tangent_constraint_residual(weight, tangent))

    def _conv_residual(self, atom, weight, tangent):
        return self._matrix_residual(atom._flatten_kernel(weight), atom._flatten_kernel(tangent))

    def _assert_state_symmetric(self, state_pair):
        lam, vel = state_pair
        self.assertTrue(jnp.allclose(lam, lam.T, atol=1e-5))
        self.assertTrue(jnp.allclose(vel, vel.T, atol=1e-5))

    def test_linear_dual_admm_and_online(self):
        key = jax.random.PRNGKey(0)
        atom = Linear(8, 4)
        weight = atom.initialize(key)
        grad = [jax.random.normal(jax.random.fold_in(key, 1), weight[0].shape)]

        naive = matrix_sign(grad[0])
        naive_residual = self._matrix_residual(weight[0], naive)

        tangent = atom.dual_ascent(weight, grad)[0]
        self.assertEqual(tangent.shape, weight[0].shape)
        self._assert_finite_tree([tangent])
        self.assertLess(self._matrix_residual(weight[0], tangent), naive_residual)

        tangent_admm = atom.admm_dual_ascent(weight, grad)[0]
        self.assertEqual(tangent_admm.shape, weight[0].shape)
        self._assert_finite_tree([tangent_admm])
        self.assertTrue(jnp.isfinite(self._matrix_residual(weight[0], tangent_admm)))

        state = atom.init_dual_state(weight)
        first_residual = None
        for _ in range(6):
            tangents, state = atom.online_dual_ascent(state, weight, grad, alpha=5e-2, beta=0.0)
            tangent_online = tangents[0]
            self._assert_finite_tree([tangent_online])
            self._assert_state_symmetric(state[0])
            residual = self._matrix_residual(weight[0], tangent_online)
            if first_residual is None:
                first_residual = residual
        self.assertLessEqual(residual, first_residual + 1e-6)

    def test_conv2d_dual_admm_and_online(self):
        key = jax.random.PRNGKey(1)
        atom = Conv2D(3, 8, 3)
        weight = atom.initialize(key)
        grad = [jax.random.normal(jax.random.fold_in(key, 1), weight[0].shape)]

        naive = atom._reshape_kernel(matrix_sign(atom._flatten_kernel(grad[0])))
        naive_residual = self._conv_residual(atom, weight[0], naive)

        tangent = atom.dual_ascent(weight, grad)[0]
        self.assertEqual(tangent.shape, weight[0].shape)
        self._assert_finite_tree([tangent])
        self.assertLess(self._conv_residual(atom, weight[0], tangent), naive_residual)

        tangent_admm = atom.admm_dual_ascent(weight, grad)[0]
        self.assertEqual(tangent_admm.shape, weight[0].shape)
        self._assert_finite_tree([tangent_admm])
        self.assertTrue(jnp.isfinite(self._conv_residual(atom, weight[0], tangent_admm)))

        state = atom.init_dual_state(weight)
        first_residual = None
        for _ in range(6):
            tangents, state = atom.online_dual_ascent(state, weight, grad, alpha=5e-2, beta=0.0)
            tangent_online = tangents[0]
            self._assert_finite_tree([tangent_online])
            self._assert_state_symmetric(state[0])
            residual = self._conv_residual(atom, weight[0], tangent_online)
            if first_residual is None:
                first_residual = residual
        self.assertLessEqual(residual, first_residual + 1e-6)

    def test_conv2d_transpose_weight_norm_paths(self):
        key = jax.random.PRNGKey(2)
        atom = Conv2DTranspose(4, 3, 3, use_weight_norm=True)
        weight = atom.initialize(key)
        grad = [
            jax.random.normal(jax.random.fold_in(key, 1), weight[0].shape),
            jax.random.normal(jax.random.fold_in(key, 2), weight[1].shape),
        ]

        naive = atom._reshape_kernel(matrix_sign(atom._flatten_kernel(grad[0])))
        naive_residual = self._conv_residual(atom, weight[0], naive)

        tangents = atom.dual_ascent(weight, grad)
        self.assertEqual(len(tangents), 2)
        self.assertEqual(tangents[0].shape, weight[0].shape)
        self.assertEqual(tangents[1].shape, weight[1].shape)
        self._assert_finite_tree(tangents)
        self.assertLess(self._conv_residual(atom, weight[0], tangents[0]), naive_residual)

        tangents_admm = atom.admm_dual_ascent(weight, grad)
        self.assertEqual(len(tangents_admm), 2)
        self.assertEqual(tangents_admm[0].shape, weight[0].shape)
        self.assertEqual(tangents_admm[1].shape, weight[1].shape)
        self._assert_finite_tree(tangents_admm)
        self.assertTrue(jnp.isfinite(self._conv_residual(atom, weight[0], tangents_admm[0])))

        state = atom.init_dual_state(weight)
        first_residual = None
        for _ in range(6):
            tangents_online, state = atom.online_dual_ascent(state, weight, grad, alpha=5e-2, beta=0.0)
            self.assertEqual(len(tangents_online), 2)
            self._assert_finite_tree(tangents_online)
            self._assert_state_symmetric(state[0])
            residual = self._conv_residual(atom, weight[0], tangents_online[0])
            if first_residual is None:
                first_residual = residual
        self.assertLessEqual(residual, first_residual + 1e-6)


if __name__ == "__main__":
    unittest.main()
