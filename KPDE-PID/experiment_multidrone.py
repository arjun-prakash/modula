#!/usr/bin/env python3
"""
experiment_multidrone.py - Multi-drone V-stack formation control
"""

import numpy as np
import matplotlib.pyplot as plt
from physics_models import ConstantWindField, QuadcopterDynamics
from controllers import PIDController, encode_gust_features, encode_trunk_vec


def run_vstack_simulation(network, num_drones=5, spacing=1.0, 
                          duration=20.0, dt=0.02, wind_magnitude=18.0):
    """
    Simulate vertical stack of drones under strong horizontal wind
    
    Args:
        network: trained DeepONet
        num_drones: number of drones in stack
        spacing: vertical spacing between drones (m)
        duration: simulation time (s)
        dt: timestep (s)
        wind_magnitude: horizontal wind speed (m/s)
        
    Returns:
        times: time array
        trajectories: dict with 'pinn' and 'baseline' arrays [N, T, 6]
    """
    print(f"\n{'='*70}")
    print(f"V-STACK MULTI-DRONE SIMULATION")
    print(f"{'='*70}")
    print(f"Drones: {num_drones}")
    print(f"Spacing: {spacing}m")
    print(f"Wind: {wind_magnitude} m/s horizontal (+X)")
    print(f"Duration: {duration}s")
    
    # Strong constant wind
    wind = ConstantWindField(
        wind_vec=np.array([wind_magnitude, 0.0, 0.0], dtype=np.float32)
    )
    
    quad = QuadcopterDynamics(mass=1.0)
    target = np.zeros(3, dtype=np.float32)

    # Controllers
    ctrls_pinn = [PIDController(kp=15.0, ki=0.3, kd=8.0) 
                  for _ in range(num_drones)]
    ctrls_base = [PIDController(kp=15.0, ki=0.3, kd=8.0) 
                  for _ in range(num_drones)]

    # Initial states (vertical stack)
    states_pinn = np.zeros((num_drones, 6), dtype=np.float32)
    states_base = np.zeros((num_drones, 6), dtype=np.float32)
    
    for i in range(num_drones):
        z0 = 0.5 + i * spacing
        states_pinn[i, :3] = np.array([3.0, 0.0, z0], dtype=np.float32)
        states_base[i, :3] = states_pinn[i, :3].copy()

    # Storage
    steps = int(duration / dt)
    times = np.arange(steps, dtype=np.float32) * dt
    traj_pinn = np.zeros((num_drones, steps, 6), dtype=np.float32)
    traj_base = np.zeros((num_drones, steps, 6), dtype=np.float32)

    # Simulation loop
    for k in range(steps):
        t = times[k]
        
        for i in range(num_drones):
            # True disturbances
            d_true_p = wind.get_force(states_pinn[i, :3], t)
            d_true_b = wind.get_force(states_base[i, :3], t)

            # PINN-augmented
            gf_vec = encode_gust_features(states_pinn[i, :3], wind, t, max_bursts=2)
            trunk = encode_trunk_vec(states_pinn[i, :], wind, t)
            d_hat = network.forward(gf_vec.reshape(1, -1), trunk.reshape(1, -1))[0]

            u_pid = ctrls_pinn[i].compute(states_pinn[i, :3], target, dt)
            u_total = u_pid - d_hat

            dstate_p = quad.compute_dynamics(states_pinn[i], u_total, d_true_p)
            states_pinn[i] = states_pinn[i] + dstate_p * dt
            traj_pinn[i, k] = states_pinn[i]

            # Baseline
            u_b = ctrls_base[i].compute(states_base[i, :3], target, dt)
            dstate_b = quad.compute_dynamics(states_base[i], u_b, d_true_b)
            states_base[i] = states_base[i] + dstate_b * dt
            traj_base[i, k] = states_base[i]

        if k % 100 == 0:
            print(f"Progress: {100*k/steps:.1f}%", end='\r')

    print(f"Progress: 100.0%")
    print(f"{'='*70}\n")
    
    return times, {'pinn': traj_pinn, 'baseline': traj_base}


def plot_vstack_analysis(times, trajs, num_drones, spacing):
    """Create comprehensive V-stack visualization"""
    tp = trajs['pinn']
    tb = trajs['baseline']
    N, T, _ = tp.shape

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Top-down view (all drones)
    ax1 = fig.add_subplot(gs[0, 0])
    colors_p = plt.cm.Blues(np.linspace(0.4, 1, N))
    colors_b = plt.cm.Reds(np.linspace(0.4, 1, N))
    
    for i in range(N):
        ax1.plot(tp[i, :, 0], tp[i, :, 1], '-', color=colors_p[i], 
                linewidth=2, alpha=0.7, label=f'PINN-{i}' if i < 2 else '')
        ax1.plot(tb[i, :, 0], tb[i, :, 1], '--', color=colors_b[i], 
                linewidth=2, alpha=0.7, label=f'Base-{i}' if i < 2 else '')
    
    ax1.scatter([0], [0], c='green', marker='X', s=300, 
                edgecolors='darkgreen', linewidths=2, zorder=10)
    ax1.set_xlabel('X (m)', fontsize=11)
    ax1.set_ylabel('Y (m)', fontsize=11)
    ax1.set_title('Top-Down View (XY Plane)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 2. Side view (XZ plane)
    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(N):
        ax2.plot(tp[i, :, 0], tp[i, :, 2], '-', color=colors_p[i], 
                linewidth=2, alpha=0.7)
        ax2.plot(tb[i, :, 0], tb[i, :, 2], '--', color=colors_b[i], 
                linewidth=2, alpha=0.7)
    ax2.scatter([0], [0], c='green', marker='X', s=300, 
                edgecolors='darkgreen', linewidths=2, zorder=10)
    ax2.set_xlabel('X (m)', fontsize=11)
    ax2.set_ylabel('Z (m)', fontsize=11)
    ax2.set_title('Side View (XZ Plane)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Final formation (Z positions)
    ax3 = fig.add_subplot(gs[0, 2])
    z_p = tp[:, -1, 2]
    z_b = tb[:, -1, 2]
    x = np.arange(N)
    width = 0.35
    
    ax3.barh(x - width/2, z_p, width, label='PINN', color='blue', alpha=0.7)
    ax3.barh(x + width/2, z_b, width, label='Baseline', color='red', alpha=0.7)
    ax3.axvline(x=0, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_yticks(x)
    ax3.set_yticklabels([f'Drone {i}' for i in range(N)])
    ax3.set_xlabel('Final Z Position (m)', fontsize=11)
    ax3.set_title('Formation Integrity', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Drift over time (X displacement)
    ax4 = fig.add_subplot(gs[1, 0])
    for i in range(N):
        drift_p = tp[i, :, 0]
        drift_b = tb[i, :, 0]
        ax4.plot(times, drift_p, '-', color=colors_p[i], linewidth=1.5, alpha=0.7)
        ax4.plot(times, drift_b, '--', color=colors_b[i], linewidth=1.5, alpha=0.7)
    ax4.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('X Drift (m)', fontsize=11)
    ax4.set_title('Horizontal Drift (Wind Effect)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Formation spacing errors
    ax5 = fig.add_subplot(gs[1, 1])
    spacing_errors_p = []
    spacing_errors_b = []
    
    for k in range(T):
        if N > 1:
            z_diffs_p = np.diff(tp[:, k, 2])
            z_diffs_b = np.diff(tb[:, k, 2])
            spacing_errors_p.append(np.std(z_diffs_p - spacing))
            spacing_errors_b.append(np.std(z_diffs_b - spacing))
    
    ax5.plot(times, spacing_errors_p, 'b-', linewidth=2.5, 
            label='PINN', alpha=0.8)
    ax5.plot(times, spacing_errors_b, 'r-', linewidth=2.5, 
            label='Baseline', alpha=0.8)
    ax5.set_xlabel('Time (s)', fontsize=11)
    ax5.set_ylabel('Spacing Error Std Dev (m)', fontsize=11)
    ax5.set_title('Formation Spacing Consistency', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # 6. Total position error
    ax6 = fig.add_subplot(gs[1, 2])
    error_p = np.linalg.norm(tp[:, :, :3], axis=2)  # [N, T]
    error_b = np.linalg.norm(tb[:, :, :3], axis=2)
    
    mean_error_p = np.mean(error_p, axis=0)
    mean_error_b = np.mean(error_b, axis=0)
    
    ax6.plot(times, mean_error_p, 'b-', linewidth=2.5, 
            label='PINN Mean', alpha=0.8)
    ax6.plot(times, mean_error_b, 'r-', linewidth=2.5, 
            label='Baseline Mean', alpha=0.8)
    ax6.fill_between(times, mean_error_p, alpha=0.2, color='blue')
    ax6.fill_between(times, mean_error_b, alpha=0.2, color='red')
    ax6.set_xlabel('Time (s)', fontsize=11)
    ax6.set_ylabel('Mean Position Error (m)', fontsize=11)
    ax6.set_title('Average Tracking Performance', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    plt.suptitle(f'V-Stack Multi-Drone Formation Control ({N} Drones, {spacing}m Spacing)', 
                fontsize=16, fontweight='bold', y=0.995)

    # Print statistics
    print("\n" + "="*70)
    print("MULTI-DRONE PERFORMANCE SUMMARY")
    print("="*70)
    
    final_drift_p = np.mean(np.abs(tp[:, -1, 0]))
    final_drift_b = np.mean(np.abs(tb[:, -1, 0]))
    
    final_z_error_p = np.std(tp[:, -1, 2] - np.arange(N) * spacing)
    final_z_error_b = np.std(tb[:, -1, 2] - np.arange(N) * spacing)
    
    print(f"Final X Drift (Mean Abs):")
    print(f"  PINN:     {final_drift_p:.2f} m")
    print(f"  Baseline: {final_drift_b:.2f} m")
    print(f"  Improvement: {(1 - final_drift_p/final_drift_b)*100:.1f}%")
    print()
    print(f"Formation Spacing Error (Std Dev):")
    print(f"  PINN:     {final_z_error_p:.3f} m")
    print(f"  Baseline: {final_z_error_b:.3f} m")
    print(f"  Improvement: {(1 - final_z_error_p/final_z_error_b)*100:.1f}%")
    print("="*70 + "\n")

    return fig


if __name__ == "__main__":
    from deeponet import DeepONet
    
    # Load trained network
    network = DeepONet()
    try:
        network.load_weights("deeponet_weights.npz")
        print("✓ Loaded trained weights")
    except:
        print("✗ No trained weights found. Train first using training.py")
        exit(1)
    
    # Run V-stack experiment
    times, trajs = run_vstack_simulation(
        network,
        num_drones=5,
        spacing=1.0,
        duration=20.0,
        dt=0.02,
        wind_magnitude=18.0
    )
    
    # Visualize results
    fig = plot_vstack_analysis(times, trajs, num_drones=5, spacing=1.0)
    plt.savefig('vstack_multidrone.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Additional test: Higher wind
    print("\nRunning extreme wind test (25 m/s)...")
    times2, trajs2 = run_vstack_simulation(
        network,
        num_drones=5,
        spacing=1.0,
        duration=20.0,
        dt=0.02,
        wind_magnitude=25.0
    )
    
    fig2 = plot_vstack_analysis(times2, trajs2, num_drones=5, spacing=1.0)
    plt.suptitle('V-Stack Multi-Drone: EXTREME WIND (25 m/s) ⚠️', 
                fontsize=16, fontweight='bold', y=0.995, color='darkred')
    plt.savefig('vstack_extreme_wind.png', dpi=150, bbox_inches='tight')
    plt.show()