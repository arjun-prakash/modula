#!/usr/bin/env python3
"""
main.py - Main entry point for running training and experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent
from typing import Optional

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
FILES_DIR = BASE_DIR / "files"


def resolve_files_path(path: Optional[Path], *, default_name: str) -> Path:
    """Return a path inside FILES_DIR, preserving relative structure."""
    candidate = path.expanduser() if path else Path(default_name)
    if candidate.is_absolute():
        candidate = Path(candidate.name)
    return FILES_DIR / candidate


from deeponet import DeepONet, DeepONetConfig
from training import METHOD_CHOICES, TrainingConfig, TrainingResult, train_single_run


def plot_training_history(result: TrainingResult, output_path: Path) -> None:
    if not result.loss_history:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(result.loss_history, label="total", color="steelblue", linewidth=2.5)
    ax.plot(result.mse_history, label="mse", color="darkorange", alpha=0.75)
    ax.plot(result.dyn_history, label="dynamic", color="seagreen", alpha=0.75)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Physics-Informed DeepONet Training", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Training curve saved: {output_path}")

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Physics-Informed DeepONet for Drone Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """Examples:
  python main.py --train                                # Train a single run
  python main.py --train --method dualize               # Train with dualize updates
  python main.py --compare                              # Run comparison (normal + storm)
  python main.py --multidrone                           # Multi-drone V-stack experiment
  python main.py --all                                  # Train (single method) + run all experiments\n"""
        ),
    )

    # Mode selection
    parser.add_argument("--train", action="store_true", help="Train the DeepONet network")
    parser.add_argument("--compare", action="store_true", help="Run comparison experiments")
    parser.add_argument("--multidrone", action="store_true", help="Run multi-drone V-stack experiment")
    parser.add_argument("--animated", action="store_true", help="Run animated comparison with moving targets")
    parser.add_argument("--all", action="store_true", help="Run training (single run) and all experiments")

    # Training hyperparameters
    parser.add_argument("--method", choices=METHOD_CHOICES, default="descent", help="Training method for a single run")
    parser.add_argument("--learning-rate", type=float, default=5e-2, help="Learning rate for a single run")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps per run")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lambda-dyn", type=float, default=0.3, help="Physics loss weight λ")
    parser.add_argument("--target-norm", type=float, default=1.0, help="Target norm for dual updates")
    parser.add_argument("--dual-alpha", type=float, default=2e-5, help="Dual-ascent step size α")
    parser.add_argument("--dual-beta", type=float, default=0.9, help="Dual-ascent momentum β")
    parser.add_argument("--position-scale", type=float, default=4.0, help="Position sampling scale for batch generation")
    parser.add_argument("--velocity-scale", type=float, default=1.5, help="Velocity sampling scale for batch generation")
    parser.add_argument("--control-scale", type=float, default=6.0, help="Control sampling scale for batch generation")
    parser.add_argument("--storm-probability", type=float, default=0.5, help="Probability of storm-mode gust fields")
    parser.add_argument("--max-bursts", type=int, default=4, help="Maximum bursts spawned per sample")
    parser.add_argument("--max-encoded-bursts", type=int, default=2, help="Encoded bursts per sample (feature vector)")
    parser.add_argument("--time-horizon", type=float, default=10.0, help="Time horizon for random sampling (seconds)")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--log-interval", type=int, default=50, help="Frequency (steps) to refresh tqdm metrics")

    # Model hyperparameters
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden width for branch/trunk networks")
    parser.add_argument("--branch-mass", type=float, default=1.0, help="Mass allocation for the branch network")
    parser.add_argument("--trunk-mass", type=float, default=1.0, help="Mass allocation for the trunk network")

    # Simulation parameters
    parser.add_argument("--duration", type=float, default=25.0, help="Simulation duration in seconds")
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation timestep in seconds")
    parser.add_argument("--num-drones", type=int, default=5, help="Number of drones for multi-drone test")

    # Outputs
    parser.add_argument("--save-weights", type=Path, default=Path("deeponet_weights.npz"), help="Path to save/load weights")

    args = parser.parse_args()

    save_weights_path = resolve_files_path(args.save_weights, default_name="deeponet_weights.npz")
    training_curve_path = resolve_files_path(Path("training_loss.png"), default_name="training_loss.png")

    comparison_normal_path = resolve_files_path(Path("comparison_normal.png"), default_name="comparison_normal.png")
    comparison_storm_path = resolve_files_path(Path("comparison_storm.png"), default_name="comparison_storm.png")
    vstack_normal_path = resolve_files_path(Path("vstack_normal.png"), default_name="vstack_normal.png")
    vstack_extreme_path = resolve_files_path(Path("vstack_extreme.png"), default_name="vstack_extreme.png")

    args.save_weights = save_weights_path

    if not (args.train or args.compare or args.multidrone or args.animated or args.all):
        parser.print_help()
        return

    print("\n" + "=" * 70)
    print(" PHYSICS-INFORMED DEEPONET FOR DRONE GUST CONTROL")
    print("=" * 70 + "\n")

    network: Optional[DeepONet] = None
    trained_result: Optional[TrainingResult] = None

    step_counter = 1

    if args.train or args.all:
        print(f"STEP {step_counter}: Training DeepONet Network")
        print("-" * 70)

        training_config = TrainingConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lambda_dyn=args.lambda_dyn,
            target_norm=args.target_norm,
            dual_alpha=args.dual_alpha,
            dual_beta=args.dual_beta,
            position_scale=args.position_scale,
            velocity_scale=args.velocity_scale,
            control_scale=args.control_scale,
            storm_probability=args.storm_probability,
            max_bursts=args.max_bursts,
            max_encoded_bursts=args.max_encoded_bursts,
            time_horizon=args.time_horizon,
            seed=args.seed,
            log_interval=args.log_interval,
        )
        deeponet_config = DeepONetConfig(
            hidden_dim=args.hidden_dim,
            branch_mass=args.branch_mass,
            trunk_mass=args.trunk_mass,
        )

        trained_result = train_single_run(
            args.method,
            config=training_config,
            deeponet_config=deeponet_config,
            learning_rate=args.learning_rate,
            save_path=save_weights_path,
        )

        network = DeepONet(trained_result.deeponet_config)
        network.set_weights(trained_result.weights)

        if save_weights_path:
            print(f"\n✓ Saved weights to {save_weights_path}\n")

        if trained_result.loss_history:
            plot_training_history(trained_result, training_curve_path)

        print("\n✓ Training complete!\n")
        step_counter += 1

    if network is None:
        network = DeepONet()
        network.load_weights(save_weights_path)
        print(f"✓ Loaded weights from {save_weights_path}\n")


    # ========== COMPARISON EXPERIMENTS ==========
    if args.compare or args.all:
        print(f"\nSTEP {step_counter}: Comparison Experiments")
        print("-" * 70)

        from experiment_comparison import (
            plot_comparison_analysis,
            run_comparison_simulation,
        )

        # Normal mode
        print("\n[1/2] Normal Mode Comparison")
        times, tp, tb, gf, field = run_comparison_simulation(
            network,
            duration=args.duration,
            dt=args.dt,
            storm_mode=False,
        )

        fig1 = plot_comparison_analysis(times, tp, tb, gf, storm_mode=False)
        plt.savefig(comparison_normal_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {comparison_normal_path}")

        plt.close(fig1)

        # Storm mode
        print("\n[2/2] Storm Mode Comparison")
        times, tp, tb, gf, field = run_comparison_simulation(
            network,
            duration=args.duration + 5,
            dt=args.dt,
            storm_mode=True,
        )

        fig2 = plot_comparison_analysis(times, tp, tb, gf, storm_mode=True)
        plt.savefig(comparison_storm_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {comparison_storm_path}")

        plt.close(fig2)

        print("\n✓ Comparison experiments complete!\n")
        step_counter += 1

    # ========== MULTI-DRONE EXPERIMENT ==========
    if args.multidrone or args.all:
        print(f"\nSTEP {step_counter}: Multi-Drone V-Stack Experiment")
        print("-" * 70)

        from experiment_multidrone import (
            plot_vstack_analysis,
            run_vstack_simulation,
        )

        # Standard wind
        print("\n[1/2] V-Stack with 18 m/s wind")
        times, trajs = run_vstack_simulation(
            network,
            num_drones=args.num_drones,
            spacing=1.0,
            duration=20.0,
            dt=args.dt,
            wind_magnitude=18.0,
        )

        fig3 = plot_vstack_analysis(
            times,
            trajs,
            num_drones=args.num_drones,
            spacing=1.0,
        )
        plt.savefig(vstack_normal_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {vstack_normal_path}")

        plt.close(fig3)

        # Extreme wind
        print("\n[2/2] V-Stack with 25 m/s EXTREME wind")
        times2, trajs2 = run_vstack_simulation(
            network,
            num_drones=args.num_drones,
            spacing=1.0,
            duration=20.0,
            dt=args.dt,
            wind_magnitude=25.0,
        )

        fig4 = plot_vstack_analysis(
            times2,
            trajs2,
            num_drones=args.num_drones,
            spacing=1.0,
        )
        plt.suptitle(
            "V-Stack: EXTREME WIND (25 m/s) ⚠️",
            fontsize=16,
            fontweight="bold",
            y=0.995,
            color="darkred",
        )
        plt.savefig(vstack_extreme_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {vstack_extreme_path}")

        plt.close(fig4)

        print("\n✓ Multi-drone experiments complete!\n")
        step_counter += 1

    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print(" ALL REQUESTED TASKS COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")

    if trained_result is not None:
        print(f"  • {save_weights_path} (network weights)")
        if trained_result.loss_history:
            print(f"  • {training_curve_path} (training curves)")
    if args.compare or args.all:
        print(f"  • {comparison_normal_path} (normal mode comparison)")
        print(f"  • {comparison_storm_path} (storm mode comparison)")
    if args.multidrone or args.all:
        print(f"  • {vstack_normal_path} (multi-drone formation)")
        print(f"  • {vstack_extreme_path} (extreme wind test)")
    print("\nFor interactive 3D visualization, open:")
    print("  • 3D Extreme Turbulence Visualization.html (in browser)")
    print("=" * 70 + "\n")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
