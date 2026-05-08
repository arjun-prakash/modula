import json
import math
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp

from benchmark import common as benchmark_common
from benchmark import cifar10 as cifar10_benchmark
from benchmark import cifar10_mlp as cifar10_mlp_benchmark
from benchmark import cifar100 as cifar100_benchmark
from benchmark import gpt as gpt_benchmark
from benchmark.run_logging import NoOpLogger, create_run_logger
from modula.atom import Linear, RMSRadiusLinear


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
        trunk, _ = benchmark_common.build_cifar_mlp_models(10, hidden_size=16)
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

    def test_gpt_manifold_scaling_has_no_selected_atom_divisor(self):
        atom = Linear(16, 64)
        self.assertEqual(gpt_benchmark.manifold_update_scale(atom, scaling="none"), 1.0)
        self.assertEqual(gpt_benchmark.manifold_update_scale(atom, scaling="fan_ratio"), 0.5)
        self.assertEqual(gpt_benchmark.manifold_update_scale(atom, scaling="fan_max"), 8.0)

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
        hidden_size = 128
        trunk, head = benchmark_common.build_cifar_mlp_models(
            10,
            hidden_size=hidden_size,
            linear_normalization="sp",
        )

        trunk_weights = trunk.initialize(jax.random.PRNGKey(0))
        head_weights = head.initialize(jax.random.PRNGKey(1))
        hidden_weight = trunk_weights[1]
        output_weight = head_weights[0]

        self.assertEqual(hidden_weight.shape, (hidden_size, hidden_size))
        self.assertGreater(
            benchmark_common.stiefel_deviation(benchmark_common._iter_weighted_atoms(trunk)[1], hidden_weight),
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

    def test_cifar10_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            results_path = tmp_path / "cifar10_results.json"
            plots_dir = tmp_path / "plots"

            cifar10_benchmark.main(
                [
                    "--smoke-test",
                    "--synthetic-data",
                    "--steps",
                    "2",
                    "--learning-rates",
                    "1e-2",
                    "--methods",
                    "adam",
                    "adamw",
                    "muon",
                    "manifold",
                    "manifold_online",
                    "manifold_admm",
                    "--results-path",
                    str(results_path),
                    "--plots-dir",
                    str(plots_dir),
                ]
            )

            self.assertTrue(results_path.exists())
            self.assertTrue((plots_dir / "cifar10_best_accuracy_vs_runtime.png").exists())

            payload = json.loads(results_path.read_text())
            self.assertEqual(payload["dataset"], "cifar10")
            self.assertEqual(payload["config"]["loss"], "cross_entropy")
            self.assertEqual(payload["config"]["muon_scaling"], "fan_ratio")
            self.assertNotIn("target_norm", payload["config"])
            self.assertIn("muon_weight_decay", payload["config"])
            self.assertIn("manifold_weight_decay", payload["config"])
            for method in ("adam", "adamw", "muon", "manifold", "manifold_online", "manifold_admm"):
                best = payload["methods"][method]["best"]
                self.assertEqual(best["loss_name"], "cross_entropy")
                self.assertTrue(math.isfinite(best["train_accuracy"]))
                self.assertTrue(math.isfinite(best["test_accuracy"]))
                self.assertTrue(math.isfinite(best["final_loss"]))
                self.assertGreater(best["final_epoch"], 0.0)
                self.assertGreater(best["training_time_seconds"], 0.0)
                self.assertGreater(best["seconds_per_step"], 0.0)
                self.assertTrue(math.isfinite(best["trunk_stiefel_deviation_mean"]))
                self.assertTrue(math.isfinite(best["trunk_stiefel_deviation_max"]))
                self.assertGreaterEqual(best["trunk_stiefel_deviation_mean"], 0.0)
                self.assertGreaterEqual(best["trunk_stiefel_deviation_max"], 0.0)

    def test_cifar100_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            results_path = tmp_path / "cifar100_results.json"
            plots_dir = tmp_path / "plots"

            cifar100_benchmark.main(
                [
                    "--smoke-test",
                    "--synthetic-data",
                    "--steps",
                    "2",
                    "--learning-rates",
                    "5e-3",
                    "--methods",
                    "adam",
                    "adamw",
                    "muon",
                    "manifold",
                    "manifold_online",
                    "manifold_admm",
                    "--muon-scaling",
                    "fan_max",
                    "--results-path",
                    str(results_path),
                    "--plots-dir",
                    str(plots_dir),
                ]
            )

            self.assertTrue(results_path.exists())
            self.assertTrue((plots_dir / "cifar100_best_accuracy_vs_runtime.png").exists())

            payload = json.loads(results_path.read_text())
            self.assertEqual(payload["dataset"], "cifar100")
            self.assertEqual(payload["config"]["loss"], "cross_entropy")
            self.assertEqual(payload["config"]["muon_scaling"], "fan_max")
            self.assertNotIn("target_norm", payload["config"])
            self.assertIn("muon_weight_decay", payload["config"])
            self.assertIn("manifold_weight_decay", payload["config"])
            for method in ("adam", "adamw", "muon", "manifold", "manifold_online", "manifold_admm"):
                best = payload["methods"][method]["best"]
                self.assertEqual(best["loss_name"], "cross_entropy")
                self.assertTrue(math.isfinite(best["train_accuracy"]))
                self.assertTrue(math.isfinite(best["test_accuracy"]))
                self.assertTrue(math.isfinite(best["final_loss"]))
                self.assertGreater(best["final_epoch"], 0.0)
                self.assertGreater(best["training_time_seconds"], 0.0)
                self.assertGreater(best["seconds_per_step"], 0.0)
                self.assertTrue(math.isfinite(best["trunk_stiefel_deviation_mean"]))
                self.assertTrue(math.isfinite(best["trunk_stiefel_deviation_max"]))
                self.assertGreaterEqual(best["trunk_stiefel_deviation_mean"], 0.0)
                self.assertGreaterEqual(best["trunk_stiefel_deviation_max"], 0.0)

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
                self.assertGreater(best["final_epoch"], 0.0)
                self.assertGreater(best["training_time_seconds"], 0.0)
                self.assertGreater(best["seconds_per_step"], 0.0)
                self.assertTrue(math.isfinite(best["trunk_stiefel_deviation_mean"]))
                self.assertTrue(math.isfinite(best["trunk_stiefel_deviation_max"]))
                self.assertGreaterEqual(best["trunk_stiefel_deviation_mean"], 0.0)
                self.assertGreaterEqual(best["trunk_stiefel_deviation_max"], 0.0)
                self.assertTrue(math.isfinite(best["trunk_rms_to_rms_deviation_mean"]))

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
            self.assertEqual(len(payload["methods"]["sgd"]["runs"]), 1)
            self.assertEqual(payload["methods"]["sgd"]["best"]["linear_normalization"], "unit_stiefel_none")
            self.assertTrue(math.isfinite(payload["methods"]["sgd"]["best"]["final_loss"]))

    def test_gpt_policy_selection(self):
        self.assertEqual(gpt_benchmark.selected_atom_keys("none", 2), ())

        mlp_names = [name for _, name in gpt_benchmark.selected_atom_keys("mlp_only", 2)]
        self.assertEqual(mlp_names, ["mlp_up", "mlp_down", "mlp_up", "mlp_down"])

        value_out_mlp_names = {
            name for _, name in gpt_benchmark.selected_atom_keys("attention_value_out_mlp", 1)
        }
        self.assertNotIn("q", value_out_mlp_names)
        self.assertNotIn("k", value_out_mlp_names)
        self.assertEqual(value_out_mlp_names, {"v", "attn_out", "mlp_up", "mlp_down"})

        all_names = [name for _, name in gpt_benchmark.selected_atom_keys("all_blocks", 1)]
        self.assertEqual(all_names, list(gpt_benchmark.BLOCK_LAYER_NAMES))

    def test_gpt_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "gpt_results.json"

            gpt_benchmark.main(
                [
                    "--smoke-test",
                    "--synthetic-data",
                    "--steps",
                    "2",
                    "--learning-rates",
                    "1e-2",
                    "--methods",
                    "adam",
                    "adamw",
                    "manifold_online",
                    "manifold_admm",
                    "--manifold-scaling",
                    "fan_max",
                    "--layer-policies",
                    "none",
                    "mlp_only",
                    "--results-path",
                    str(results_path),
                ]
            )

            self.assertTrue(results_path.exists())
            self.assertFalse((Path("benchmark") / "transformer_manifold_muon_plan.md").exists())
            payload = json.loads(results_path.read_text())
            self.assertEqual(payload["dataset"], "synthetic_gpt")
            self.assertNotIn("target_norm", payload["config"])
            self.assertNotIn("blocks_mass", payload["config"])
            self.assertEqual(payload["config"]["manifold_scaling"], "fan_max")
            self.assertIn("manifold_weight_decay", payload["config"])

            adam_best = payload["methods"]["adam"]["policies"]["none"]["best"]
            self.assertTrue(math.isfinite(adam_best["train_loss"]))
            self.assertTrue(math.isfinite(adam_best["val_loss"]))
            self.assertTrue(math.isfinite(adam_best["final_batch_loss"]))
            self.assertGreater(adam_best["final_epoch"], 0.0)
            self.assertGreater(adam_best["training_time_seconds"], 0.0)
            self.assertGreater(adam_best["seconds_per_step"], 0.0)
            self.assertGreater(adam_best["tokens_per_second"], 0.0)

            adamw_best = payload["methods"]["adamw"]["policies"]["none"]["best"]
            self.assertTrue(math.isfinite(adamw_best["train_loss"]))
            self.assertTrue(math.isfinite(adamw_best["val_loss"]))
            self.assertTrue(math.isfinite(adamw_best["final_batch_loss"]))
            self.assertGreater(adamw_best["final_epoch"], 0.0)
            self.assertGreater(adamw_best["training_time_seconds"], 0.0)
            self.assertGreater(adamw_best["seconds_per_step"], 0.0)
            self.assertGreater(adamw_best["tokens_per_second"], 0.0)

            for method in ("manifold_online", "manifold_admm"):
                policies = payload["methods"][method]["policies"]
                self.assertEqual(policies["none"]["runs"], [])
                for policy in ("mlp_only",):
                    best = policies[policy]["best"]
                    self.assertTrue(math.isfinite(best["train_loss"]))
                    self.assertTrue(math.isfinite(best["val_loss"]))
                    self.assertTrue(math.isfinite(best["final_batch_loss"]))
                    self.assertGreater(best["final_epoch"], 0.0)
                    self.assertGreater(best["training_time_seconds"], 0.0)
                    self.assertGreater(best["seconds_per_step"], 0.0)
                    self.assertGreater(best["tokens_per_second"], 0.0)
                    self.assertTrue(math.isfinite(best["stiefel_deviation_mean"]))
                    self.assertTrue(math.isfinite(best["stiefel_deviation_max"]))
                    self.assertGreaterEqual(best["stiefel_deviation_mean"], 0.0)
                    self.assertGreaterEqual(best["stiefel_deviation_max"], 0.0)


if __name__ == "__main__":
    unittest.main()
