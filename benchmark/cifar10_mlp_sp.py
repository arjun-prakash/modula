if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import jax
import numpy as np

from benchmark import sp as benchmark_sp
from benchmark.common import (
    MANIFOLD_METHODS,
    add_common_arguments,
    apply_smoke_test_overrides,
    canonicalize_methods,
    make_results_config,
    make_run_config,
    plot_best_accuracy_vs_runtime,
    plot_cifar_mlp_lr_transfer,
    prepare_dataset,
    print_run_summary,
    resolve_training_steps,
    save_results,
    train_single_run,
    validate_common_arguments,
)
from benchmark.run_logging import create_run_logger

SP_BASELINE_METHOD_CHOICES = ("sgd", "adam", "adamw")
SP_METHOD_CHOICES = (*SP_BASELINE_METHOD_CHOICES, *MANIFOLD_METHODS)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CIFAR-10 MLP benchmark for standard parameterization")
    add_common_arguments(
        parser,
        dataset_name="cifar10_mlp_sp",
        default_learning_rates=[1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
        default_steps=4000,
        default_batch_size=64,
        default_eval_every=100,
        default_results_path=Path("results/benchmark_cifar10_mlp_sp_results.json"),
        default_plots_dir=Path("plots") / "benchmark_cifar10_mlp_sp",
    )
    parser.set_defaults(methods=list(SP_METHOD_CHOICES))
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
    args = parser.parse_args(argv)
    validate_common_arguments(parser, args)
    if any(hidden_size <= 0 for hidden_size in args.hidden_sizes):
        parser.error("--hidden-sizes must contain only positive integers")
    invalid_methods = sorted(set(args.methods) - set(SP_METHOD_CHOICES))
    if invalid_methods:
        parser.error(f"SP benchmark only supports methods: {' '.join(SP_METHOD_CHOICES)}")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.methods = canonicalize_methods(args.methods)
    apply_smoke_test_overrides(args)

    dataset = prepare_dataset(
        "cifar10",
        synthetic_data=args.synthetic_data,
        smoke_test=args.smoke_test,
        seed=args.seed,
        cifar_normalization=args.cifar_normalization,
    )
    resolve_training_steps(args, dataset)
    results = {method: [] for method in args.methods}
    best_runs = {}
    base_key = jax.random.PRNGKey(args.seed)

    for method_idx, method in enumerate(args.methods):
        method_key = jax.random.fold_in(base_key, method_idx)
        for hidden_idx, hidden_size in enumerate(args.hidden_sizes):
            hidden_key = jax.random.fold_in(method_key, hidden_idx)
            is_manifold_method = method in MANIFOLD_METHODS
            parameterization = "unit_stiefel" if is_manifold_method else "sp"
            effective_manifold_scaling = "none" if is_manifold_method else args.manifold_scaling
            head_adam_update_scale = 1.0
            trunk, head = benchmark_sp.build_cifar_mlp_models(
                dataset.num_classes,
                hidden_size,
                trunk=args.trunk,
                parameterization=parameterization,
            )
            for lr_idx, learning_rate in enumerate(args.learning_rates):
                run_key = jax.random.fold_in(hidden_key, lr_idx)
                run_seed = int(jax.random.randint(run_key, (), 0, np.iinfo(np.int32).max))
                run_name = f"cifar10-mlp-sp-{args.trunk}-{method}-h{hidden_size}-lr{learning_rate:.3g}"
                run_config = make_run_config(
                    args,
                    dataset_name="cifar10_mlp_sp",
                    num_classes=dataset.num_classes,
                    method=method,
                    learning_rate=learning_rate,
                    hidden_size=hidden_size,
                )
                run_config.update(
                    {
                        "parameterization": parameterization,
                        "linear_normalization": parameterization,
                        "effective_manifold_scaling": effective_manifold_scaling,
                        "effective_muon_scaling": args.muon_scaling,
                        "project_trunk_after_update": False,
                        "head_adam_update_scale": float(head_adam_update_scale),
                    }
                )
                logger = create_run_logger(
                    use_wandb=args.use_wandb,
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=run_name,
                    config=run_config,
                    group=args.wandb_group or "cifar10-mlp-sp-benchmark",
                    tags=[
                        "cifar10",
                        "mlp",
                        "sp",
                        args.trunk,
                        method,
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
                        manifold_scaling=effective_manifold_scaling,
                        muon_scaling=args.muon_scaling,
                        muon_momentum=args.muon_momentum,
                        muon_weight_decay=args.muon_weight_decay,
                        sgd_momentum=args.sgd_momentum,
                        loss=args.loss,
                        logger=logger,
                        show_progress=not args.smoke_test,
                        head_adam_update_scale=head_adam_update_scale,
                    )
                finally:
                    logger.finish()

                run_result["learning_rate"] = float(learning_rate)
                run_result["hidden_size"] = int(hidden_size)
                run_result["trunk"] = str(args.trunk)
                run_result["parameterization"] = str(parameterization)
                run_result["linear_normalization"] = str(parameterization)
                run_result["effective_manifold_scaling"] = str(effective_manifold_scaling)
                run_result["effective_muon_scaling"] = str(args.muon_scaling)
                run_result["project_trunk_after_update"] = False
                run_result["head_adam_update_scale"] = float(head_adam_update_scale)
                results[method].append(run_result)

                best = best_runs.get(method)
                if best is None or run_result["test_accuracy"] > best["test_accuracy"]:
                    best_runs[method] = dict(run_result)

                print_run_summary(method, run_result)

    plot_best_accuracy_vs_runtime(best_runs, args.plots_dir, "cifar10_mlp_sp")
    plot_cifar_mlp_lr_transfer(results, args.plots_dir)
    save_results(
        "cifar10_mlp_sp",
        make_results_config(args, dataset_name="cifar10_mlp_sp", num_classes=dataset.num_classes),
        results,
        best_runs,
        args.results_path,
    )


if __name__ == "__main__":
    main()
