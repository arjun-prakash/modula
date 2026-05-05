if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import jax
import numpy as np

from benchmark.common import (
    add_common_arguments,
    apply_smoke_test_overrides,
    build_cifar_models,
    canonicalize_methods,
    make_results_config,
    make_run_config,
    plot_best_accuracy_vs_runtime,
    prepare_dataset,
    print_run_summary,
    save_results,
    train_single_run,
)
from benchmark.run_logging import create_run_logger


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Multiclass CIFAR-10 benchmark with Muon-style manifold methods")
    add_common_arguments(
        parser,
        dataset_name="cifar10",
        default_learning_rates=[1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
        default_steps=2000,
        default_batch_size=128,
        default_eval_every=100,
        default_results_path=Path("results/benchmark_cifar10_results.json"),
        default_plots_dir=Path("plots") / "benchmark_cifar10",
    )
    return parser.parse_args(argv)


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
    trunk, head = build_cifar_models(dataset.num_classes)

    results = {method: [] for method in args.methods}
    best_runs = {}
    base_key = jax.random.PRNGKey(args.seed)

    for method_idx, method in enumerate(args.methods):
        method_key = jax.random.fold_in(base_key, method_idx)
        for lr_idx, learning_rate in enumerate(args.learning_rates):
            run_key = jax.random.fold_in(method_key, lr_idx)
            run_seed = int(jax.random.randint(run_key, (), 0, np.iinfo(np.int32).max))
            run_name = f"cifar10-{method}-lr{learning_rate:.3g}"
            logger = create_run_logger(
                use_wandb=args.use_wandb,
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=make_run_config(
                    args,
                    dataset_name=dataset.name,
                    num_classes=dataset.num_classes,
                    method=method,
                    learning_rate=learning_rate,
                ),
                group=args.wandb_group or "cifar10-benchmark",
                tags=[dataset.name, method],
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
                    manifold_scaling=args.manifold_scaling,
                    muon_scaling=args.muon_scaling,
                    muon_momentum=args.muon_momentum,
                    muon_weight_decay=args.muon_weight_decay,
                    loss=args.loss,
                    logger=logger,
                    show_progress=not args.smoke_test,
                )
            finally:
                logger.finish()

            run_result["learning_rate"] = float(learning_rate)
            results[method].append(run_result)

            best = best_runs.get(method)
            if best is None or run_result["test_accuracy"] > best["test_accuracy"]:
                best_runs[method] = dict(run_result)

            print_run_summary(method, run_result)

    plot_best_accuracy_vs_runtime(best_runs, args.plots_dir, dataset.name)
    save_results(
        dataset.name,
        make_results_config(args, dataset_name=dataset.name, num_classes=dataset.num_classes),
        results,
        best_runs,
        args.results_path,
    )


if __name__ == "__main__":
    main()
