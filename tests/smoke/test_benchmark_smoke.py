import json
import math
import tempfile
import unittest
from pathlib import Path

from benchmark import cifar10 as cifar10_benchmark
from benchmark import cifar100 as cifar100_benchmark
from benchmark.run_logging import NoOpLogger, create_run_logger


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
        logger.log({"step": 0, "loss": 1.0})
        logger.finish()

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
            for method in ("adam", "manifold", "manifold_online", "manifold_admm"):
                best = payload["methods"][method]["best"]
                self.assertTrue(math.isfinite(best["train_accuracy"]))
                self.assertTrue(math.isfinite(best["test_accuracy"]))
                self.assertTrue(math.isfinite(best["final_loss"]))
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
            self.assertTrue((plots_dir / "cifar100_best_accuracy_vs_runtime.png").exists())

            payload = json.loads(results_path.read_text())
            self.assertEqual(payload["dataset"], "cifar100")
            for method in ("adam", "manifold", "manifold_online", "manifold_admm"):
                best = payload["methods"][method]["best"]
                self.assertTrue(math.isfinite(best["train_accuracy"]))
                self.assertTrue(math.isfinite(best["test_accuracy"]))
                self.assertTrue(math.isfinite(best["final_loss"]))
                self.assertGreater(best["training_time_seconds"], 0.0)
                self.assertGreater(best["seconds_per_step"], 0.0)
                self.assertTrue(math.isfinite(best["trunk_stiefel_deviation_mean"]))
                self.assertTrue(math.isfinite(best["trunk_stiefel_deviation_max"]))
                self.assertGreaterEqual(best["trunk_stiefel_deviation_mean"], 0.0)
                self.assertGreaterEqual(best["trunk_stiefel_deviation_max"], 0.0)


if __name__ == "__main__":
    unittest.main()
