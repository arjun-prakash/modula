#!/usr/bin/env python3
"""
experiment_comparison.py - Static comparison of PINN vs Baseline
"""

import numpy as np
import matplotlib.pyplot as plt
from physics_models import GustField, QuadcopterDynamics
from controllers import PIDController, encode_gust_features, encode_trunk_vec


def run_comparison_simulation(network, duration=25.0, dt=0.02, storm_mode=False):
    """
    Compare PINN-augmented vs baseline PID controller
    
    Args:
        network: trained DeepONet
        duration: simulation time (seconds)
        dt: timestep (seconds)
        storm_mode: bool, use extreme conditions
        
    Returns:
        times: time array
        traj_pinn: PINN trajectory [N, 6]
        traj_baseline: baseline trajectory [N, 6]
        gust_forces: disturbance forces [N, 3]
        gust_field: GustField object
    """
    print(f"\nRunning comparison simulation...")
    print(f"Mode: {'STORM ⚠️' if storm_mode else 'NORMAL'}")
    print(f"Duration: {duration}s, dt: {dt}s")
    
    # Create world
    gust_field = GustField(
        max_bursts=4 if not storm_mode else 8,
        spawn_radius=8.0,
        storm_mode=storm_mode
    )

    # Two identical quads
    quad_pinn = QuadcopterDynamics(mass=1.0)
    quad_baseline = QuadcopterDynamics(mass=1.0)

    # Two identical controllers
    controller_pinn = PIDController(kp=15.0, ki=0.3, kd=8.0)
    controller_baseline = PIDController(kp=15.0, ki=0.3, kd=8.0)

    # Same initial conditions
    state_pinn = np.array([3.0, 2.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    state_baseline = state_pinn.copy()
    target = np.zeros(3, dtype=np.float32)

    # Storage
    n = int(duration / dt)
    traj_pinn = np.zeros((n, 6), dtype=np.float32)
    traj_base = np.zeros((n, 6), dtype=np.float32)
    gust_forces = np.zeros((n, 3), dtype=np.float32)
    times = np.zeros(n, dtype=np.float32)

    # Simulation loop
    for i in range(n):
        t = i * dt
        times[i] = t

        # Update gust field
        avg_pos = 0.5 * (state_pinn[:3] + state_baseline[:3])
        gust_field.update(dt, avg_pos)

        # True disturbances
        gust_pinn = gust_field.get_force(state_pinn[:3], t)
        gust_base = gust_field.get_force(state_baseline[:3], t)
        gust_forces[i] = gust_pinn

        # PINN-augmented control
        gf_vec = encode_gust_features(state_pinn[:3], gust_field, t, max_bursts=2)
        trunk = encode_trunk_vec(state_pinn, gust_field, t)
        d_hat = network.forward(gf_vec.reshape(1, -1), trunk.reshape(1, -1))[0]

        u_pid = controller_pinn.compute(state_pinn[:3], target, dt)
        u_total = u_pid - d_hat  # Feedforward cancellation

        dstate_pinn = quad_pinn.compute_dynamics(state_pinn, u_total, gust_pinn)
        state_pinn = state_pinn + dstate_pinn * dt
        traj_pinn[i] = state_pinn

        # Baseline control
        u_base = controller_baseline.compute(state_baseline[:3], target, dt)
        dstate_base = quad_baseline.compute_dynamics(state_baseline, u_base, gust_base)
        state_baseline = state_baseline + dstate_base * dt
        traj_base[i] = state_baseline

    print("Simulation complete!")
    return times, traj_pinn, traj_base, gust_forces, gust_field


def plot_comparison_analysis(times, traj_pinn, traj_baseline, gust_forces, 
                            storm_mode=False):
    """Create comprehensive comparison plots"""
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

    # 3D trajectory
    ax1 = fig.add_subplot(gs[0:2, 0:2], projection='3d')
    ax1.plot(traj_pinn[:, 0], traj_pinn[:, 1], traj_pinn[:, 2],
             'b-', linewidth=2.5, alpha=0.7, label='PINN-Augmented')
    ax1.plot(traj_baseline[:, 0], traj_baseline[:, 1], traj_baseline[:, 2],
             'r-', linewidth=2.5, alpha=0.7, label='Baseline PID')
    ax1.scatter([0], [0], [0], c='green', marker='X', s=500,
                edgecolors='darkgreen', linewidths=2, label='Target', zorder=10)
    ax1.scatter([traj_pinn[0, 0]], [traj_pinn[0, 1]], [traj_pinn[0, 2]],
                c='cyan', marker='o', s=200, edgecolors='blue',
                linewidths=2, label='Start', zorder=10)
    ax1.set_xlabel('X (m)', fontsize=11)
    ax1.set_ylabel('Y (m)', fontsize=11)
    ax1.set_zlabel('Z (m)', fontsize=11)
    ax1.set_title('3D Trajectory Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Tracking error
    ax2 = fig.add_subplot(gs[0, 2:])
    error_pinn = np.linalg.norm(traj_pinn[:, :3], axis=1)
    error_baseline = np.linalg.norm(traj_baseline[:, :3], axis=1)
    ax2.plot(times, error_pinn, 'b-', linewidth=2.5, 
             label='PINN-Augmented', alpha=0.8)
    ax2.plot(times, error_baseline, 'r-', linewidth=2.5, 
             label='Baseline PID', alpha=0.8)
    ax2.fill_between(times, error_pinn, alpha=0.2, color='blue')
    ax2.fill_between(times, error_baseline, alpha=0.2, color='red')
    ax2.axhline(y=0.5, color='green', linestyle='--', 
                alpha=0.6, linewidth=1.5, label='Good (<0.5m)')
    ax2.set_ylabel('Distance from Target (m)', fontsize=11)
    ax2.set_title('Tracking Error Comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Velocity magnitude
    ax3 = fig.add_subplot(gs[1, 2:])
    vel_pinn = np.linalg.norm(traj_pinn[:, 3:6], axis=1)
    vel_baseline = np.linalg.norm(traj_baseline[:, 3:6], axis=1)
    ax3.plot(times, vel_pinn, 'b-', linewidth=2.5, 
             label='PINN-Augmented', alpha=0.8)
    ax3.plot(times, vel_baseline, 'r-', linewidth=2.5, 
             label='Baseline PID', alpha=0.8)
    ax3.set_ylabel('Velocity Magnitude (m/s)', fontsize=11)
    ax3.set_title('Velocity Comparison', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Top-down view
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(traj_pinn[:, 0], traj_pinn[:, 1], 'b-', 
             linewidth=2.5, alpha=0.7, label='PINN')
    ax4.plot(traj_baseline[:, 0], traj_baseline[:, 1], 'r-', 
             linewidth=2.5, alpha=0.7, label='Baseline')
    ax4.scatter([0], [0], c='green', marker='X', s=400,
                edgecolors='darkgreen', linewidths=2, label='Target', zorder=10)
    ax4.set_xlabel('X (m)', fontsize=11)
    ax4.set_ylabel('Y (m)', fontsize=11)
    ax4.set_title('Top-Down View (XY)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')

    # Statistics
    ax5 = fig.add_subplot(gs[2, 1])
    metrics = ['Mean', 'Max', 'Final', 'RMS']
    pinn_stats = [
        np.mean(error_pinn),
        np.max(error_pinn),
        error_pinn[-1],
        np.sqrt(np.mean(error_pinn**2))
    ]
    baseline_stats = [
        np.mean(error_baseline),
        np.max(error_baseline),
        error_baseline[-1],
        np.sqrt(np.mean(error_baseline**2))
    ]
    x = np.arange(len(metrics))
    width = 0.35
    ax5.bar(x - width/2, pinn_stats, width, label='PINN', 
            color='blue', alpha=0.7)
    ax5.bar(x + width/2, baseline_stats, width, label='Baseline', 
            color='red', alpha=0.7)
    ax5.set_ylabel('Error (m)', fontsize=11)
    ax5.set_title('Error Statistics', fontsize=13, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics, fontsize=10)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')

    # Gust profile
    ax6 = fig.add_subplot(gs[2, 2:])
    gust_mag = np.linalg.norm(gust_forces, axis=1)
    ax6.plot(times, gust_mag, 'orange', linewidth=2.5, alpha=0.8, label='Total Gust')
    ax6.fill_between(times, 0, gust_mag, alpha=0.3, color='orange')
    if storm_mode:
        base_wind_mag = np.sqrt(8**2 + 6**2 + 2**2)
        ax6.axhline(y=base_wind_mag, color='red', linestyle='--', 
                    linewidth=2, alpha=0.7, label=f'Base Wind (~{base_wind_mag:.1f}N)')
        ax6.legend(fontsize=10)
    ax6.set_xlabel('Time (s)', fontsize=11)
    ax6.set_ylabel('Force (N)', fontsize=11)
    ax6.set_title('Gust Disturbance Profile', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    title = 'PINN-Augmented vs Baseline PID Controller'
    if storm_mode:
        title += ' - STORM MODE ⚠️'
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

    # Print statistics
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(f"{'Metric':<20} {'PINN':<15} {'Baseline':<15} {'Improvement':<15}")
    print("-"*70)
    for i, metric in enumerate(metrics):
        improvement = (1 - pinn_stats[i]/baseline_stats[i]) * 100
        print(f"{metric:<20} {pinn_stats[i]:<15.3f} {baseline_stats[i]:<15.3f} "
              f"{improvement:>13.1f}%")
    print("="*70)

    return fig


if __name__ == "__main__":
    from deeponet import DeepONet
    
    # Load trained network
    network = DeepONet()
    try:
        network.load_weights("deeponet_weights.npz")
        print("Loaded trained weights")
    except:
        print("No trained weights found. Train first using training.py")
        exit(1)
    
    # Normal mode test
    times, tp, tb, gf, field = run_comparison_simulation(
        network, duration=25.0, dt=0.02, storm_mode=False
    )
    fig1 = plot_comparison_analysis(times, tp, tb, gf, storm_mode=False)
    plt.savefig('comparison_normal.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Storm mode test
    times, tp, tb, gf, field = run_comparison_simulation(
        network, duration=30.0, dt=0.02, storm_mode=True
    )
    fig2 = plot_comparison_analysis(times, tp, tb, gf, storm_mode=True)
    plt.savefig('comparison_storm.png', dpi=150, bbox_inches='tight')
    plt.show()