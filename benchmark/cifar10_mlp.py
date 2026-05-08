if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import jax
import numpy as np

from benchmark.common import (
    LINEAR_NORMALIZATION_CHOICES,
    add_common_arguments,
    apply_smoke_test_overrides,
    build_cifar_mlp_models,
    canonicalize_methods,
    make_results_config,
    make_run_config,
    plot_best_accuracy_vs_runtime,
    plot_cifar_mlp_lr_transfer,
    prepare_dataset,
    print_run_summary,
    save_results,
    train_single_run,
)
from benchmark.run_logging import create_run_logger


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CIFAR-10 MLP benchmark with Muon-style manifold methods")
    add_common_arguments(
        parser,
        dataset_name="cifar10_mlp",
        default_learning_rates=[1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
        default_steps=2000,
        default_batch_size=128,
        default_eval_every=100,
        default_results_path=Path("results/benchmark_cifar10_mlp_results.json"),
        default_plots_dir=Path("plots") / "benchmark_cifar10_mlp",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[64],
        help="Hidden sizes to sweep for both CIFAR-10 MLP hidden layers",
    )
    parser.add_argument(
        "--trunk",
        type=str,
        default="default",
        choices=("default", "wide3"),
        help="CIFAR-10 MLP trunk architecture",
    )
    parser.add_argument(
        "--linear-normalizations",
        type=str,
        nargs="+",
        default=["unit_stiefel"],
        choices=LINEAR_NORMALIZATION_CHOICES,
        help="Linear-layer normalization variants for CIFAR-10 MLP trunk layers",
    )
    args = parser.parse_args(argv)
    if any(hidden_size <= 0 for hidden_size in args.hidden_sizes):
        parser.error("--hidden-sizes must contain only positive integers")
    return args


def _normalization_run_config(linear_normalization: str, args):
    if linear_normalization == "sp":
        return "sp", args.manifold_scaling, args.muon_scaling, False
    if linear_normalization == "unit_stiefel":
        return "unit_stiefel", args.manifold_scaling, args.muon_scaling, False
    if linear_normalization == "unit_stiefel_none":
        return "unit_stiefel", "none", "none", False
    if linear_normalization == "unit_stiefel_fan_ratio":
        return "unit_stiefel", "fan_ratio", "fan_ratio", False
    if linear_normalization == "rms_radius":
        return "rms_radius", "fan_ratio", "fan_ratio", True
    raise ValueError(f"Unknown linear normalization: {linear_normalization}")


def main(argv=None):
    args = parse_args(argv)
    args.methods = canonicalize_methods(args.methods)
    apply_smoke_test_overrides(args)

    dataset = prepare_dataset(
        "cifar10",
        synthetic_data=args.synthetic_data,
        smoke_test=args.smoke_test,
        seed=args.seed,
    )
    results = {method: [] for method in args.methods}
    best_runs = {}
    base_key = jax.random.PRNGKey(args.seed)

    for method_idx, method in enumerate(args.methods):
        method_key = jax.random.fold_in(base_key, method_idx)
        for norm_idx, linear_normalization in enumerate(args.linear_normalizations):
            norm_key = jax.random.fold_in(method_key, norm_idx)
            atom_normalization, manifold_scaling, muon_scaling, project_trunk_after_update = _normalization_run_config(
                linear_normalization,
                args,
            )
            for hidden_idx, hidden_size in enumerate(args.hidden_sizes):
                hidden_key = jax.random.fold_in(norm_key, hidden_idx)
                trunk, head = build_cifar_mlp_models(
                    dataset.num_classes,
                    hidden_size,
                    trunk=args.trunk,
                    linear_normalization=atom_normalization,
                )
                for lr_idx, learning_rate in enumerate(args.learning_rates):
                    run_key = jax.random.fold_in(hidden_key, lr_idx)
                    run_seed = int(jax.random.randint(run_key, (), 0, np.iinfo(np.int32).max))
                    run_name = (
                        f"cifar10-mlp-{args.trunk}-{linear_normalization}-{method}-"
                        f"h{hidden_size}-lr{learning_rate:.3g}"
                    )
                    run_config = make_run_config(
                        args,
                        dataset_name="cifar10_mlp",
                        num_classes=dataset.num_classes,
                        method=method,
                        learning_rate=learning_rate,
                        hidden_size=hidden_size,
                    )
                    run_config.update(
                        {
                            "linear_normalization": linear_normalization,
                            "effective_manifold_scaling": manifold_scaling,
                            "effective_muon_scaling": muon_scaling,
                            "project_trunk_after_update": bool(project_trunk_after_update),
                        }
                    )
                    logger = create_run_logger(
                        use_wandb=args.use_wandb,
                        project=args.wandb_project,
                        entity=args.wandb_entity,
                        name=run_name,
                        config=run_config,
                        group=args.wandb_group or "cifar10-mlp-benchmark",
                        tags=[
                            "cifar10",
                            "mlp",
                            args.trunk,
                            method,
                            linear_normalization,
                            f"hidden_size={hidden_size}",
                        ],
                    )
                    try:
                        run_result = train_single_run(
                            trunk,
                            head,
                            dataset,
                            batch_size=args.batch_size,
                            steps=args.steps,
                            learning_rate=learning_rate,
                            eval_every=args.eval_every,
                            eval_train_samples=args.eval_train_samples,
                            seed=run_seed,
                            method=method,
                            dual_alpha=args.dual_alpha,
                            dual_beta=args.dual_beta,
                            admm_steps=args.admm_steps,
                            admm_rho=args.admm_rho,
                            adam_weight_decay=args.adam_weight_decay,
                            manifold_momentum=args.manifold_momentum,
                            manifold_weight_decay=args.manifold_weight_decay,
                            manifold_scaling=manifold_scaling,
                            muon_scaling=muon_scaling,
                            muon_momentum=args.muon_momentum,
                            muon_weight_decay=args.muon_weight_decay,
                            loss=args.loss,
                            logger=logger,
                            show_progress=not args.smoke_test,
                            project_trunk_after_update=project_trunk_after_update,
                        )
                    finally:
                        logger.finish()

                    run_result["learning_rate"] = float(learning_rate)
                    run_result["hidden_size"] = int(hidden_size)
                    run_result["trunk"] = str(args.trunk)
                    run_result["linear_normalization"] = str(linear_normalization)
                    run_result["effective_manifold_scaling"] = str(manifold_scaling)
                    run_result["effective_muon_scaling"] = str(muon_scaling)
                    run_result["project_trunk_after_update"] = bool(project_trunk_after_update)
                    results[method].append(run_result)

                    best = best_runs.get(method)
                    if best is None or run_result["test_accuracy"] > best["test_accuracy"]:
                        best_runs[method] = dict(run_result)

                    print_run_summary(method, run_result)

    plot_best_accuracy_vs_runtime(best_runs, args.plots_dir, "cifar10_mlp")
    plot_cifar_mlp_lr_transfer(results, args.plots_dir)
    save_results(
        "cifar10_mlp",
        make_results_config(args, dataset_name="cifar10_mlp", num_classes=dataset.num_classes),
        results,
        best_runs,
        args.results_path,
    )


if __name__ == "__main__":
    main()
