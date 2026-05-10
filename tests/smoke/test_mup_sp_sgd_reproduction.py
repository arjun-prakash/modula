import ast
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPRO_DIR = REPO_ROOT / "reproductions" / "mup_mlp_sp_sgd"
ENTRYPOINT = REPRO_DIR / "train.py"
README = REPRO_DIR / "README.md"


class MuPSpSgdReproductionTest(unittest.TestCase):
    def test_entrypoint_is_parseable_and_documents_source(self):
        source = ENTRYPOINT.read_text()

        ast.parse(source)
        self.assertTrue(README.exists())
        self.assertIn("microsoft/mup/examples/MLP/main.py", README.read_text())

    def test_model_matches_original_sp_sgd_mlp_shape(self):
        source = ENTRYPOINT.read_text()

        expected_patterns = [
            r"self\.fc_1\s*=\s*nn\.Linear\(3072,\s*width,\s*bias=False\)",
            r"self\.fc_2\s*=\s*nn\.Linear\(width,\s*width,\s*bias=False\)",
            r"self\.fc_3\s*=\s*MuReadout\(width,\s*num_classes,\s*bias=False,\s*output_mult=",
            r"nn\.init\.zeros_\(self\.fc_3\.weight\)",
            r"self\.nonlin\(self\.fc_1\(x\)\s*\*\s*self\.input_mult\*\*0\.5\)",
            r"self\.nonlin\(self\.fc_2\(out\)\)",
        ]
        for pattern in expected_patterns:
            with self.subTest(pattern=pattern):
                self.assertRegex(source, pattern)

    def test_sp_path_uses_original_optimizer_and_data_defaults(self):
        source = ENTRYPOINT.read_text()

        self.assertIn("set_base_shapes(mynet, None)", source)
        self.assertIn("MuSGD(mynet.parameters(), lr=args.lr, momentum=args.momentum)", source)
        self.assertRegex(source, r"--batch_size['\"],\s*type=int,\s*default=64")
        self.assertRegex(source, r"--epochs['\"],\s*type=int,\s*default=20")
        self.assertRegex(source, r"--momentum['\"],\s*type=float,\s*default=0\.9")
        self.assertRegex(source, r"--lr['\"],\s*type=float,\s*default=0\.1")
        self.assertIn("transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))", source)

    def test_training_reports_full_epoch_train_loss(self):
        source = ENTRYPOINT.read_text()

        self.assertIn("train_loss += loss.item() * data.shape[0]", source)
        self.assertIn("train_loss /= len(train_loader.dataset)", source)
        self.assertIn("logs.tsv", source)
        self.assertIn("train_loss", source)
        self.assertIn("test_loss", source)
        self.assertIsNotNone(re.search(r"for epoch in range\(1,\s*args\.epochs\s*\+\s*1\)", source))

    def test_log_file_can_be_named_from_cli(self):
        source = ENTRYPOINT.read_text()

        self.assertRegex(source, r"--log_file['\"],\s*type=str,\s*default=['\"]logs\.tsv['\"]")
        self.assertIn("Path(os.path.expanduser(args.log_dir)) / args.log_file", source)

    def test_defaults_use_paper_relu_xent_multipliers(self):
        source = ENTRYPOINT.read_text()

        self.assertRegex(source, r"--input_mult['\"],\s*type=float,\s*default=0\.00390625")
        self.assertRegex(source, r"--output_mult['\"],\s*type=float,\s*default=32\.0")
        readme = README.read_text()
        self.assertIn("`--input_mult 0.00390625`", readme)
        self.assertIn("`--output_mult 32.0`", readme)

    def test_final_train_loss_chart_is_written(self):
        source = ENTRYPOINT.read_text()

        self.assertRegex(source, r"--chart_file['\"],\s*type=str,\s*default=['\"]final_train_loss\.tsv['\"]")
        self.assertIn("def write_final_train_loss_chart(chart_path, logs):", source)
        self.assertIn("math.log2(lr)", source)
        self.assertIn('final["train_loss"]', source)
        self.assertIn("write_final_train_loss_chart(chart_path, logs)", source)
        self.assertIn("final_train_loss.tsv", README.read_text())
