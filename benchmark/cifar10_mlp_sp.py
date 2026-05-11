if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from benchmark import sp as benchmark_sp
from benchmark.common import (
    add_common_arguments,
    apply_smoke_test_overrides,
    make_run_config,
    prepare_dataset,
    print_run_summary,
    save_result,
    train_single_run,
    validate_common_arguments,
)

DATASET_NAME = "cifar10_mlp_sp"
PARAMETERIZATION = "sp"
METHOD = "sgd"

build_models = benchmark_sp.build_cifar_mlp_models


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CIFAR-10 MLP run with standard parameterization")
    add_common_arguments(
        parser,
        default_learning_rate=0.1,
        default_steps=4000,
        default_batch_size=64,
        default_eval_every=100,
        default_output_path=Path("results/cifar10_mlp_sp.json"),
    )
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size for both MLP hidden layers")
    args = parser.parse_args(argv)
    validate_common_arguments(parser, args)
    if args.hidden_size <= 0:
        parser.error("--hidden-size must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    apply_smoke_test_overrides(args)

    dataset = prepare_dataset(
        "cifar10",
        synthetic_data=args.synthetic_data,
        smoke_test=args.smoke_test,
        seed=args.seed,
    )
    trunk, head = build_models(dataset.num_classes, args.hidden_size)
    run_config = make_run_config(
        args,
        dataset_name=DATASET_NAME,
        num_classes=dataset.num_classes,
        method=METHOD,
        hidden_size=args.hidden_size,
        parameterization=PARAMETERIZATION,
    )
    run_result = train_single_run(
        trunk,
        head,
        dataset,
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        seed=args.seed,
        method=METHOD,
        show_progress=not args.smoke_test,
    )
    run_result.update(
        {
            "learning_rate": float(args.learning_rate),
            "hidden_size": int(args.hidden_size),
            "parameterization": PARAMETERIZATION,
            "linear_normalization": PARAMETERIZATION,
            "project_trunk_after_update": False,
            "head_adam_update_scale": 1.0,
            "head_init_scale": 1.0,
        }
    )

    print_run_summary(METHOD, run_result)
    save_result(DATASET_NAME, run_config, run_result, args.output)


if __name__ == "__main__":
    main()
