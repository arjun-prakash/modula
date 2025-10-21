#!/usr/bin/env python3
"""
main.py - Main entry point for running all experiments
"""

import sys
import argparse
import matplotlib.pyplot as plt
from deeponet import DeepONet
from training import train_deeponet_physics


def main():
    parser = argparse.ArgumentParser(
        description='Physics-Informed DeepONet for Drone Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train                    # Train the network
  python main.py --compare                  # Run comparison (normal + storm)
  python main.py --multidrone               # Multi-drone V-stack experiment
  python main.py --all                      # Run all experiments
  
  python main.py --train --epochs 500       # Custom training
  python main.py --compare --duration 30    # Longer simulation
        """
    )
    
    # Mode selection
    parser.add_argument('--train', action='store_true',
                       help='Train the DeepONet network')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison experiments')
    parser.add_argument('--multidrone', action='store_true',
                       help='Run multi-drone V-stack experiment')
    parser.add_argument('--animated', action='store_true',
                       help='Run animated comparison with moving targets')
    parser.add_argument('--all', action='store_true',
                       help='Run all experiments')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of training epochs (default: 300)')
    parser.add_argument('--samples', type=int, default=200,
                       help='Samples per epoch (default: 200)')
    parser.add_argument('--batch-size', type=int, default=20,
                       help='Batch size (default: 20)')
    parser.add_argument('--lr', type=float, default=5e-3,
                       help='Learning rate (default: 5e-3)')
    parser.add_argument('--lambda-dyn', type=float, default=0.3,
                       help='Physics loss weight (default: 0.3)')
    
    # Simulation parameters
    parser.add_argument('--duration', type=float, default=25.0,
                       help='Simulation duration in seconds (default: 25)')
    parser.add_argument('--dt', type=float, default=0.02,
                       help='Timestep in seconds (default: 0.02)')
    parser.add_argument('--num-drones', type=int, default=5,
                       help='Number of drones for multi-drone test (default: 5)')
    
    # Output
    parser.add_argument('--save-weights', type=str, default='deeponet_weights.npz',
                       help='Path to save/load weights (default: deeponet_weights.npz)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable interactive plotting')
    
    args = parser.parse_args()
    
    # If no mode selected, show help
    if not (args.train or args.compare or args.multidrone or args.all):
        parser.print_help()
        return
    
    print("\n" + "="*70)
    print(" PHYSICS-INFORMED DEEPONET FOR DRONE GUST CONTROL")
    print("="*70 + "\n")
    
    network = None
    
    # ========== TRAINING ==========
    if args.train or args.all:
        print("STEP 1: Training DeepONet Network")
        print("-" * 70)
        
        network, losses = train_deeponet_physics(
            epochs=args.epochs,
            samples_per_epoch=args.samples,
            batch_size=args.batch_size,
            lr0=args.lr,
            lambda_dyn=args.lambda_dyn,
            save_path=args.save_weights
        )
        
        # Plot training curve
        if not args.no_plot:
            plt.figure(figsize=(10, 5))
            plt.plot(losses, linewidth=2.5, color='steelblue')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Physics-Informed DeepONet Training', 
                     fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig('training_loss.png', dpi=150)
            print("\n✓ Training curve saved: training_loss.png")
            plt.show()
        
        print("\n✓ Training complete!\n")
    
    # ========== LOAD NETWORK ==========
    if not network:
        print("Loading trained network...")
        network = DeepONet()
        try:
            network.load_weights(args.save_weights)
            print(f"✓ Loaded weights from {args.save_weights}\n")
        except:
            print(f"✗ Could not load weights from {args.save_weights}")
            print("Please train the network first using --train")
            return
    
    # ========== COMPARISON EXPERIMENTS ==========
    if args.compare or args.all:
        print("\nSTEP 2: Comparison Experiments")
        print("-" * 70)
        
        from experiment_comparison import (
            run_comparison_simulation, 
            plot_comparison_analysis
        )
        
        # Normal mode
        print("\n[1/2] Normal Mode Comparison")
        times, tp, tb, gf, field = run_comparison_simulation(
            network, 
            duration=args.duration, 
            dt=args.dt, 
            storm_mode=False
        )
        
        fig1 = plot_comparison_analysis(times, tp, tb, gf, storm_mode=False)
        plt.savefig('comparison_normal.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: comparison_normal.png")
        
        if not args.no_plot:
            plt.show()
        plt.close()
        
        # Storm mode
        print("\n[2/2] Storm Mode Comparison")
        times, tp, tb, gf, field = run_comparison_simulation(
            network, 
            duration=args.duration + 5, 
            dt=args.dt, 
            storm_mode=True
        )
        
        fig2 = plot_comparison_analysis(times, tp, tb, gf, storm_mode=True)
        plt.savefig('comparison_storm.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: comparison_storm.png")
        
        if not args.no_plot:
            plt.show()
        plt.close()
        
        print("\n✓ Comparison experiments complete!\n")
    
    # ========== MULTI-DRONE EXPERIMENT ==========
    if args.multidrone or args.all:
        print("\nSTEP 3: Multi-Drone V-Stack Experiment")
        print("-" * 70)
        
        from experiment_multidrone import (
            run_vstack_simulation,
            plot_vstack_analysis
        )
        
        # Standard wind
        print("\n[1/2] V-Stack with 18 m/s wind")
        times, trajs = run_vstack_simulation(
            network,
            num_drones=args.num_drones,
            spacing=1.0,
            duration=20.0,
            dt=args.dt,
            wind_magnitude=18.0
        )
        
        fig3 = plot_vstack_analysis(times, trajs, 
                                   num_drones=args.num_drones, 
                                   spacing=1.0)
        plt.savefig('vstack_normal.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: vstack_normal.png")
        
        if not args.no_plot:
            plt.show()
        plt.close()
        
        # Extreme wind
        print("\n[2/2] V-Stack with 25 m/s EXTREME wind")
        times2, trajs2 = run_vstack_simulation(
            network,
            num_drones=args.num_drones,
            spacing=1.0,
            duration=20.0,
            dt=args.dt,
            wind_magnitude=25.0
        )
        
        fig4 = plot_vstack_analysis(times2, trajs2, 
                                   num_drones=args.num_drones, 
                                   spacing=1.0)
        plt.suptitle('V-Stack: EXTREME WIND (25 m/s) ⚠️', 
                    fontsize=16, fontweight='bold', 
                    y=0.995, color='darkred')
        plt.savefig('vstack_extreme.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: vstack_extreme.png")
        
        if not args.no_plot:
            plt.show()
        plt.close()
        
        print("\n✓ Multi-drone experiments complete!\n")
    
    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print(" ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    if args.train or args.all:
        print(f"  • {args.save_weights} (network weights)")
        print("  • training_loss.png (training curve)")
    if args.compare or args.all:
        print("  • comparison_normal.png (normal mode comparison)")
        print("  • comparison_storm.png (storm mode comparison)")
    if args.multidrone or args.all:
        print("  • vstack_normal.png (multi-drone formation)")
        print("  • vstack_extreme.png (extreme wind test)")
    print("\nFor interactive 3D visualization, open:")
    print("  • 3D Extreme Turbulence Visualization.html (in browser)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()