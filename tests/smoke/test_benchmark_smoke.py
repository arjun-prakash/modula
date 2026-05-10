import json
import math
import tempfile
import unittest
import importlib.util
import contextlib
import io
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from benchmark import common as benchmark_common
from benchmark import cifar10_mlp_mup as cifar10_mlp_benchmark
from benchmark import cifar10_mlp_sp as cifar10_mlp_sp_benchmark
from benchmark.run_logging import NoOpLogger, create_run_logger
from modula.atom import Linear, RMSRadiusLinear, StandardParamLinear


class BenchmarkSmokeTest(unittest.TestCase):
    def test_noop_logger(self):
        logger = create_run_logger(
            use_wandb=False,
            project="benchmark-tests",
            entity=None,
            name="noop",
            config={"dataset": "synthetic"},
        )
        self.assertIsInstance(logger, NoOpLogger)
        logger.log({"epoch": 0.0, "loss": 1.0})
        logger.finish()

    def test_cifar_manifold_scaling_has_no_target_norm_allocator(self):
        from benchmark import mup as benchmark_mup

        trunk, _ = benchmark_mup.build_cifar_mlp_models(10, hidden_size=16)
        weights = trunk.initialize(jax.random.PRNGKey(0))
        tangents = [jnp.ones_like(weight) for weight in weights]

        none_directions = benchmark_common._manifold_directions(trunk, tangents, scaling="none")
        for direction, tangent in zip(none_directions, tangents):
            self.assertTrue(bool(jnp.allclose(direction, tangent)))

        fan_ratio_directions = benchmark_common._manifold_directions(trunk, tangents, scaling="fan_ratio")
        atoms = benchmark_common._iter_weighted_atoms(trunk)
        for atom, direction, tangent in zip(atoms, fan_ratio_directions, tangents):
            expected = math.sqrt(atom.fanout / atom.fanin)
            self.assertTrue(bool(jnp.allclose(direction, expected * tangent)))

    def test_benchmark_parameterization_modules_exist(self):
        self.assertIsNotNone(importlib.util.find_spec("benchmark.sp"))
        self.assertIsNotNone(importlib.util.find_spec("benchmark.mup"))
        self.assertIsNotNone(importlib.util.find_spec("benchmark.scaling"))

    def test_shared_scaling_has_no_selected_atom_divisor(self):
        from benchmark import scaling as benchmark_scaling

        atom = Linear(16, 64)
        self.assertEqual(benchmark_scaling.manifold_update_scale(atom, scaling="none"), 1.0)
        self.assertEqual(benchmark_scaling.manifold_update_scale(atom, scaling="fan_ratio"), 0.5)
        self.assertEqual(benchmark_scaling.manifold_update_scale(atom, scaling="fan_max"), 8.0)

    def test_cifar_normalization_matches_torchvision_centering(self):
        images = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float32)

        centered = benchmark_common.normalize_cifar_images(images, normalization="minus_one_one")
        zero_one = benchmark_common.normalize_cifar_images(images, normalization="zero_one")

        self.assertTrue(bool(jnp.allclose(centered, jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32))))
        self.assertTrue(bool(jnp.allclose(zero_one, images)))

    def test_rms_radius_linear_geometry(self):
        atom = RMSRadiusLinear(16, 64)
        expected_radius = math.sqrt(atom.fanout / atom.fanin)

        weights = atom.initialize(jax.random.PRNGKey(0))
        for candidate in (weights, atom.project(weights), atom.retract(weights)):
            matrix = candidate[0]
            deviation = benchmark_common.stiefel_deviation(atom, matrix)
            self.assertIsNotNone(deviation)
            self.assertLess(deviation, 2e-2)
            rms_norm = benchmark_common._linear_rms_to_rms_norm(atom, matrix)
            self.assertIsNotNone(rms_norm)
            self.assertTrue(math.isclose(rms_norm, 1.0, rel_tol=2e-2, abs_tol=2e-2))

    def test_cifar_mlp_sp_linear_initialization_is_not_stiefel(self):
        from benchmark import sp as benchmark_sp

        hidden_size = 128
        trunk, head = benchmark_sp.build_cifar_mlp_models(
            10,
            hidden_size=hidden_size,
        )

        trunk_weights = trunk.initialize(jax.random.PRNGKey(0))
        head_weights = head.initialize(jax.random.PRNGKey(1))
        hidden_weight = trunk_weights[1]
        output_weight = head_weights[0]
        hidden_atom = benchmark_common._iter_weighted_atoms(trunk)[1]
        head_atom = benchmark_common._iter_weighted_atoms(head)[0]

        self.assertEqual(hidden_weight.shape, (hidden_size, hidden_size))
        self.assertIsInstance(hidden_atom, StandardParamLinear)
        self.assertIsInstance(head_atom, StandardParamLinear)
        self.assertGreater(
            benchmark_common.stiefel_deviation(hidden_atom, hidden_weight),
            0.05,
        )
        self.assertTrue(
            math.isclose(
                float(jnp.std(hidden_weight)),
                1.0 / math.sqrt(hidden_size),
                rel_tol=0.2,
            )
        )
        self.assertTrue(
            math.isclose(
                float(jnp.std(output_weight)),
                1.0 / math.sqrt(hidden_size),
                rel_tol=0.35,
            )
        )

    def test_cifar_mlp_sp_manifold_routes_use_original_linear_atoms(self):
        from benchmark import sp as benchmark_sp

        trunk, head = benchmark_sp.build_cifar_mlp_models(
            10,
            hidden_size=16,
            parameterization="unit_stiefel",
        )

        trunk_atoms = benchmark_common._iter_weighted_atoms(trunk)
        head_atoms = benchmark_common._iter_weighted_atoms(head)
        self.assertIsInstance(trunk_atoms[1], Linear)
        self.assertNotIsInstance(trunk_atoms[1], StandardParamLinear)
        self.assertNotIsInstance(trunk_atoms[1], RMSRadiusLinear)
        self.assertIsInstance(head_atoms[0], Linear)
        self.assertNotIsInstance(head_atoms[0], StandardParamLinear)
        self.assertNotIsInstance(head_atoms[0], RMSRadiusLinear)

    def test_cifar_mlp_mup_routes_use_manifold_atoms(self):
        from benchmark import mup as benchmark_mup

        unit_trunk, unit_head = benchmark_mup.build_cifar_mlp_models(
            10,
            hidden_size=16,
            parameterization="unit_stiefel",
        )
        rms_trunk, rms_head = benchmark_mup.build_cifar_mlp_models(
            10,
            hidden_size=16,
            parameterization="rms_radius",
        )

        self.assertIsInstance(benchmark_common._iter_weighted_atoms(unit_trunk)[1], Linear)
        self.assertNotIsInstance(benchmark_common._iter_weighted_atoms(unit_trunk)[1], StandardParamLinear)
        self.assertIsInstance(benchmark_common._iter_weighted_atoms(unit_head)[0], Linear)
        self.assertIsInstance(benchmark_common._iter_weighted_atoms(rms_trunk)[1], RMSRadiusLinear)
        self.assertIsInstance(benchmark_common._iter_weighted_atoms(rms_head)[0], Linear)
        self.assertNotIsInstance(benchmark_common._iter_weighted_atoms(rms_head)[0], RMSRadiusLinear)

    def test_mup_head_scaling_uses_base_width(self):
        from benchmark import mup as benchmark_mup

        _, head = benchmark_mup.build_cifar_mlp_models(
            10,
            hidden_size=512,
            parameterization="rms_radius",
        )

        self.assertTrue(math.isclose(benchmark_mup.head_adam_update_scale(head, base_width=256), 0.5))
        self.assertTrue(math.isclose(benchmark_mup.head_init_scale(head, base_width=256), math.sqrt(0.5)))

    def test_head_adam_update_scale_scales_adam_updates(self):
        weights = [jnp.array([[1.0, -2.0, 3.0]], dtype=jnp.float32)]
        grads = [jnp.array([[0.5, -0.25, 0.75]], dtype=jnp.float32)]
        learning_rate = 0.01
        update_scale = 0.25

        optimizer = optax.adam(learning_rate)
        state = optimizer.init(weights)
        base_updates, _ = optimizer.update(grads, state, params=weights)
        scaled_updates, _ = benchmark_common._scaled_head_optimizer_update(
            "adam",
            optimizer,
            state,
            grads,
            weights,
            learning_rate=learning_rate,
            adam_weight_decay=0.0,
            head_adam_update_scale=update_scale,
        )

        self.assertTrue(bool(jnp.allclose(scaled_updates[0], update_scale * base_updates[0])))

    def test_head_adam_update_scale_leaves_adamw_decay_unscaled(self):
        weights = [jnp.array([[1.0, -2.0, 3.0]], dtype=jnp.float32)]
        grads = [jnp.array([[0.5, -0.25, 0.75]], dtype=jnp.float32)]
        learning_rate = 0.01
        weight_decay = 0.2
        update_scale = 0.25

        optimizer = optax.adam(learning_rate)
        state = optimizer.init(weights)
        adam_updates, _ = optimizer.update(grads, state, params=weights)
        scaled_updates, _ = benchmark_common._scaled_head_optimizer_update(
            "adamw",
            optimizer,
            state,
            grads,
            weights,
            learning_rate=learning_rate,
            adam_weight_decay=weight_decay,
            head_adam_update_scale=update_scale,
        )

        expected_decay_updates = [-learning_rate * weight_decay * weights[0]]
        expected_updates = [update_scale * adam_updates[0] + expected_decay_updates[0]]
        self.assertTrue(bool(jnp.allclose(scaled_updates[0], expected_updates[0])))

    def test_cifar10_mlp_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            results_path = tmp_path / "cifar10_mlp_results.json"
            plots_dir = tmp_path / "plots"

            cifar10_mlp_benchmark.main(
                [
                    "--smoke-test",
                    "--synthetic-data",
                    "--steps",
                    "2",
                    "--learning-rates",
                    "1e-2",
                    "--hidden-sizes",
                    "16",
                    "32",
                    "--trunk",
                    "wide3",
                    "--methods",
                    "adamw",
                    "manifold",
                    "--linear-normalizations",
                    "unit_stiefel_none",
                    "rms_radius",
                    "--muon-scaling",
                    "fan_max",
                    "--results-path",
                    str(results_path),
                    "--plots-dir",
                    str(plots_dir),
                ]
            )

            self.assertTrue(results_path.exists())
            self.assertTrue((plots_dir / "cifar10_mlp_best_accuracy_vs_runtime.png").exists())

            payload = json.loads(results_path.read_text())
            self.assertEqual(payload["dataset"], "cifar10_mlp")
            self.assertEqual(payload["config"]["loss"], "cross_entropy")
            self.assertEqual(payload["config"]["epochs"], None)
            self.assertEqual(payload["config"]["sgd_momentum"], 0.9)
            self.assertEqual(payload["config"]["cifar_normalization"], "minus_one_one")
            self.assertEqual(payload["config"]["mup_base_width"], 256)
            self.assertEqual(payload["config"]["muon_scaling"], "fan_max")
            self.assertEqual(payload["config"]["hidden_sizes"], [16, 32])
            self.assertEqual(payload["config"]["trunk"], "wide3")
            self.assertEqual(payload["config"]["linear_normalizations"], ["unit_stiefel_none", "rms_radius"])
            self.assertNotIn("target_norm", payload["config"])
            self.assertIn("muon_weight_decay", payload["config"])
            self.assertIn("manifold_weight_decay", payload["config"])
            for method in ("adamw", "manifold"):
                self.assertEqual(len(payload["methods"][method]["runs"]), 4)
                self.assertEqual(
                    {run["hidden_size"] for run in payload["methods"][method]["runs"]},
                    {16, 32},
                )
                self.assertEqual(
                    {run["trunk"] for run in payload["methods"][method]["runs"]},
                    {"wide3"},
                )
                self.assertEqual(
                    {run["linear_normalization"] for run in payload["methods"][method]["runs"]},
                    {"unit_stiefel_none", "rms_radius"},
                )
                self.assertIn("lr_transfer", payload["methods"][method])
                self.assertIn("unit_stiefel_none", payload["methods"][method]["lr_transfer"])
                self.assertIn("rms_radius", payload["methods"][method]["lr_transfer"])
                best = payload["methods"][method]["best"]
                self.assertIn(best["hidden_size"], (16, 32))
                self.assertEqual(best["trunk"], "wide3")
                self.assertIn(best["linear_normalization"], ("unit_stiefel_none", "rms_radius"))
                self.assertEqual(best["loss_name"], "cross_entropy")
                self.assertTrue(math.isfinite(best["train_accuracy"]))
                self.assertTrue(math.isfinite(best["test_accuracy"]))
                self.assertTrue(math.isfinite(best["final_loss"]))
                self.assertTrue(math.isfinite(best["full_train_loss"]))
                self.assertGreater(best["final_epoch"], 0.0)
                self.assertGreater(best["training_time_seconds"], 0.0)
                self.assertGreater(best["seconds_per_step"], 0.0)
                self.assertTrue(math.isfinite(best["trunk_stiefel_deviation_mean"]))
                self.assertTrue(math.isfinite(best["trunk_stiefel_deviation_max"]))
                self.assertGreaterEqual(best["trunk_stiefel_deviation_mean"], 0.0)
                self.assertGreaterEqual(best["trunk_stiefel_deviation_max"], 0.0)
                self.assertTrue(math.isfinite(best["trunk_rms_to_rms_deviation_mean"]))
                self.assertTrue(math.isfinite(best["head_adam_update_scale"]))
                self.assertTrue(best["head_adam_update_scale"] > 0.0)

            for method in ("adamw", "manifold"):
                for run in payload["methods"][method]["runs"]:
                    expected_scale = 256 / (4 * run["hidden_size"])
                    self.assertTrue(
                        math.isclose(
                            run["head_adam_update_scale"],
                            expected_scale,
                            rel_tol=1e-12,
                        )
                    )

    def test_cifar10_mlp_accepts_plain_sgd(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            results_path = tmp_path / "cifar10_mlp_sgd_results.json"
            plots_dir = tmp_path / "plots"

            cifar10_mlp_benchmark.main(
                [
                    "--smoke-test",
                    "--synthetic-data",
                    "--steps",
                    "2",
                    "--learning-rates",
                    "1e-2",
                    "--hidden-sizes",
                    "16",
                    "--methods",
                    "sgd",
                    "--sgd-momentum",
                    "0.3",
                    "--linear-normalizations",
                    "unit_stiefel_none",
                    "--results-path",
                    str(results_path),
                    "--plots-dir",
                    str(plots_dir),
                ]
            )

            payload = json.loads(results_path.read_text())
            self.assertEqual(payload["config"]["methods"], ["sgd"])
            self.assertEqual(payload["config"]["sgd_momentum"], 0.3)
            self.assertEqual(len(payload["methods"]["sgd"]["runs"]), 1)
            self.assertEqual(payload["methods"]["sgd"]["best"]["linear_normalization"], "unit_stiefel_none")
            self.assertTrue(math.isfinite(payload["methods"]["sgd"]["best"]["final_loss"]))
            self.assertTrue(math.isfinite(payload["methods"]["sgd"]["best"]["full_train_loss"]))

    def test_cifar10_mlp_epochs_resolve_steps_for_mup_and_sp(self):
        for benchmark_module, dataset_name in (
            (cifar10_mlp_benchmark, "cifar10_mlp"),
            (cifar10_mlp_sp_benchmark, "cifar10_mlp_sp"),
        ):
            with self.subTest(dataset=dataset_name), tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                results_path = tmp_path / f"{dataset_name}_results.json"
                plots_dir = tmp_path / "plots"
                args = [
                    "--smoke-test",
                    "--synthetic-data",
                    "--epochs",
                    "0.25",
                    "--batch-size",
                    "8",
                    "--learning-rates",
                    "1e-2",
                    "--hidden-sizes",
                    "16",
                    "--methods",
                    "sgd",
                    "--results-path",
                    str(results_path),
                    "--plots-dir",
                    str(plots_dir),
                ]
                if dataset_name == "cifar10_mlp":
                    args.extend(["--linear-normalizations", "unit_stiefel_none"])

                benchmark_module.main(args)

                payload = json.loads(results_path.read_text())
                best = payload["methods"]["sgd"]["best"]
                self.assertEqual(payload["config"]["epochs"], 0.25)
                self.assertEqual(payload["config"]["batch_size"], 8)
                self.assertEqual(payload["config"]["steps"], 2)
                self.assertEqual(best["steps"], 2)
                self.assertTrue(math.isclose(best["final_epoch"], 0.25, rel_tol=1e-6))

    def test_cifar10_mlp_sp_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            results_path = tmp_path / "cifar10_mlp_sp_results.json"
            plots_dir = tmp_path / "plots"

            cifar10_mlp_sp_benchmark.main(
                [
                    "--smoke-test",
                    "--synthetic-data",
                    "--steps",
                    "2",
                    "--learning-rates",
                    "1e-2",
                    "--hidden-sizes",
                    "16",
                    "--methods",
                    "sgd",
                    "--results-path",
                    str(results_path),
                    "--plots-dir",
                    str(plots_dir),
                ]
            )

            self.assertTrue(results_path.exists())
            self.assertTrue((plots_dir / "cifar10_mlp_sp_best_accuracy_vs_runtime.png").exists())

            payload = json.loads(results_path.read_text())
            self.assertEqual(payload["dataset"], "cifar10_mlp_sp")
            self.assertEqual(payload["config"]["methods"], ["sgd"])
            self.assertEqual(payload["config"]["epochs"], None)
            self.assertEqual(payload["config"]["sgd_momentum"], 0.9)
            self.assertEqual(payload["config"]["cifar_normalization"], "minus_one_one")
            self.assertEqual(payload["config"]["hidden_sizes"], [16])
            self.assertEqual(payload["methods"]["sgd"]["best"]["parameterization"], "sp")
            self.assertEqual(payload["methods"]["sgd"]["best"]["linear_normalization"], "sp")
            self.assertFalse(payload["methods"]["sgd"]["best"]["project_trunk_after_update"])
            self.assertTrue(math.isfinite(payload["methods"]["sgd"]["best"]["final_loss"]))
            self.assertTrue(math.isfinite(payload["methods"]["sgd"]["best"]["full_train_loss"]))

    def test_cifar10_mlp_sp_accepts_manifold_methods_and_rejects_muon(self):
        args = cifar10_mlp_sp_benchmark.parse_args(
            [
                "--methods",
                "manifold",
                "manifold_online",
                "manifold_admm",
            ]
        )
        self.assertEqual(args.methods, ["manifold", "manifold_online", "manifold_admm"])

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cifar10_mlp_sp_benchmark.parse_args(["--methods", "muon"])

    def test_cifar10_mlp_sp_manifold_smoke_uses_unit_stiefel_without_scaling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            results_path = tmp_path / "cifar10_mlp_sp_manifold_results.json"
            plots_dir = tmp_path / "plots"

            cifar10_mlp_sp_benchmark.main(
                [
                    "--smoke-test",
                    "--synthetic-data",
                    "--steps",
                    "2",
                    "--learning-rates",
                    "1e-2",
                    "--hidden-sizes",
                    "16",
                    "--methods",
                    "manifold",
                    "--manifold-scaling",
                    "fan_max",
                    "--results-path",
                    str(results_path),
                    "--plots-dir",
                    str(plots_dir),
                ]
            )

            payload = json.loads(results_path.read_text())
            best = payload["methods"]["manifold"]["best"]
            self.assertEqual(best["parameterization"], "unit_stiefel")
            self.assertEqual(best["linear_normalization"], "unit_stiefel")
            self.assertEqual(best["effective_manifold_scaling"], "none")
            self.assertEqual(best["head_adam_update_scale"], 1.0)

if __name__ == "__main__":
    unittest.main()
