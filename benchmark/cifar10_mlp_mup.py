if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from benchmark import mup as benchmark_mup
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

DATASET_NAME = "cifar10_mlp_mup"
PARAMETERIZATION = benchmark_mup.PARAMETERIZATION
METHOD = "manifold"

build_models = benchmark_mup.build_cifar_mlp_models
head_adam_update_scale = benchmark_mup.head_adam_update_scale
head_init_scale = benchmark_mup.head_init_scale


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CIFAR-10 MLP run with muP-style Modula parameterization")
    add_common_arguments(
        parser,
        default_learning_rate=0.01,
        default_steps=4000,
        default_batch_size=64,
        default_eval_every=100,
        default_output_path=Path("results/cifar10_mlp_mup.json"),
    )
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size for both MLP hidden layers")
    parser.add_argument(
        "--mup-base-width",
        type=int,
        default=256,
        help="Base hidden width used for muP readout initialization and optimizer scaling",
    )
    args = parser.parse_args(argv)
    validate_common_arguments(parser, args)
    if args.hidden_size <= 0:
        parser.error("--hidden-size must be positive")
    if args.mup_base_width <= 0:
        parser.error("--mup-base-width must be positive")
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
    head_update_scale = head_adam_update_scale(head, base_width=args.mup_base_width)
    head_scale = head_init_scale(head, base_width=args.mup_base_width)
    run_config = make_run_config(
        args,
        dataset_name=DATASET_NAME,
        num_classes=dataset.num_classes,
        method=METHOD,
        hidden_size=args.hidden_size,
        parameterization=PARAMETERIZATION,
        mup_base_width=args.mup_base_width,
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
        project_trunk_after_update=True,
        head_adam_update_scale=head_update_scale,
        head_init_scale=head_scale,
    )
    run_result.update(
        {
            "learning_rate": float(args.learning_rate),
            "hidden_size": int(args.hidden_size),
            "parameterization": PARAMETERIZATION,
            "linear_normalization": PARAMETERIZATION,
            "project_trunk_after_update": True,
            "head_adam_update_scale": float(head_update_scale),
            "head_init_scale": float(head_scale),
        }
    )

    print_run_summary(METHOD, run_result)
    save_result(DATASET_NAME, run_config, run_result, args.output)


if __name__ == "__main__":
    main()
