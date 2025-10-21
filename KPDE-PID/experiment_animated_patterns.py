#!/usr/bin/env python3
"""
experiment_animated_patterns.py - Animated comparison with moving target patterns
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from physics_models import GustField, QuadcopterDynamics
from controllers import PIDController, encode_gust_features, encode_trunk_vec


class TargetPattern:
    """Generate various target trajectories"""
    
    @staticmethod
    def stationary(t):
        return np.array([0.0, 0.0, 2.0], dtype=np.float32)
    
    @staticmethod
    def circular(t):
        r = 5.0
        w = 0.3
        return np.array([
            r * np.cos(w * t),
            r * np.sin(w * t),
            2.0 + 0.5 * np.sin(0.5 * t)
        ], dtype=np.float32)
    
    @staticmethod
    def figure8(t):
        r = 4.5
        w = 0.4
        return np.array([
            r * np.sin(w * t),
            r * np.sin(w * t) * np.cos(w * t),
            2.0 + 1.0 * np.sin(0.6 * t)
        ], dtype=np.float32)
    
    @staticmethod
    def vertical(t):
        return np.array([
            0.0,
            0.0,
            2.0 + 3.0 * np.sin(0.5 * t)
        ], dtype=np.float32)
    
    @staticmethod
    def box3d(t):
        phase = (t * 0.3) % 4
        side = 6.0
        if phase < 1:
            return np.array([side * phase, side, 2.0], dtype=np.float32)
        elif phase < 2:
            return np.array([side, side * (2 - phase), 2.0 + 2.0 * (phase - 1)], dtype=np.float32)
        elif phase < 3:
            return np.array([side * (3 - phase), -side, 4.0 - 2.0 * (phase - 2)], dtype=np.float32)
        else:
            return np.array([-side, -side + side * 2 * (phase - 3), 2.0], dtype=np.float32)
    
    @staticmethod
    def lissajous(t):
        return np.array([
            5.0 * np.sin(0.5 * t),
            5.0 * np.sin(0.35 * t + np.pi/3),
            2.0 + 1.5 * np.sin(0.7 * t)
        ], dtype=np.float32)


def create_animated_comparison_with_patterns(network, pattern_name='circular', 
                                            duration=30.0, dt=0.05, storm_mode=True):
    """
    Create animated comparison with moving target and real-time error plots
    
    Args:
        network: trained DeepONet
        pattern_name: one of ['stationary', 'circular', 'figure8', 'vertical', 'box3d', 'lissajous']
        duration: simulation time (s)
        dt: timestep (s)
        storm_mode: use extreme wind conditions
    """
    print(f"\n{'='*70}")
    print(f"ANIMATED COMPARISON: {pattern_name.upper()} TARGET PATTERN")
    print(f"{'='*70}")
    
    # Get target function
    target_func = getattr(TargetPattern, pattern_name)
    
    # Create world
    gust_field = GustField(
        max_bursts=8 if storm_mode else 5,
        spawn_radius=10.0,
        storm_mode=storm_mode
    )
    
    # Two drones
    quad_pinn = QuadcopterDynamics(mass=1.0)
    quad_base = QuadcopterDynamics(mass=1.0)
    
    ctrl_pinn = PIDController(kp=15.0, ki=0.3, kd=8.0)
    ctrl_base = PIDController(kp=15.0, ki=0.3, kd=8.0)
    
    # Initial state
    state_pinn = np.array([4.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float32)
    state_base = state_pinn.copy()
    
    # Storage
    n_steps = int(duration / dt)
    traj_pinn = [state_pinn[:3].copy()]
    traj_base = [state_base[:3].copy()]
    target_traj = []
    error_pinn = []
    error_base = []
    times = [0]
    gust_log = []
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 3D trajectory plot
    ax_3d = fig.add_subplot(gs[:2, :2], projection='3d')
    ax_3d.set_xlim([-8, 8])
    ax_3d.set_ylim([-8, 8])
    ax_3d.set_zlim([0, 8])
    ax_3d.set_xlabel('X (m)', fontsize=10)
    ax_3d.set_ylabel('Y (m)', fontsize=10)
    ax_3d.set_zlabel('Z (m)', fontsize=10)
    title_str = f'Target Pattern: {pattern_name.capitalize()}'
    if storm_mode:
        title_str += ' (STORM MODE ⚠️)'
    ax_3d.set_title(title_str, fontsize=12, fontweight='bold')
    
    line_pinn, = ax_3d.plot([], [], [], 'b-', linewidth=2, alpha=0.7, label='PINN')
    line_base, = ax_3d.plot([], [], [], 'r-', linewidth=2, alpha=0.7, label='Baseline')
    line_target, = ax_3d.plot([], [], [], 'm--', linewidth=1.5, alpha=0.5, label='Target Path')
    
    drone_pinn = ax_3d.scatter([], [], [], c='cyan', s=300, marker='o', 
                               edgecolors='blue', linewidths=2, label='PINN Drone')
    drone_base = ax_3d.scatter([], [], [], c='red', s=300, marker='o',
                               edgecolors='darkred', linewidths=2, label='Base Drone')
    target_point = ax_3d.scatter([], [], [], c='lime', s=400, marker='X',
                                 edgecolors='darkgreen', linewidths=2, label='Target')
    
    ax_3d.legend(loc='upper right', fontsize=9)
    ax_3d.grid(True, alpha=0.3)
    
    # Error plot
    ax_err = fig.add_subplot(gs[0, 2])
    ax_err.set_xlim([0, duration])
    ax_err.set_ylim([0, 10])
    ax_err.set_xlabel('Time (s)', fontsize=10)
    ax_err.set_ylabel('Tracking Error (m)', fontsize=10)
    ax_err.set_title('Real-Time Error Comparison', fontsize=11, fontweight='bold')
    ax_err.grid(True, alpha=0.3)
    
    line_err_pinn, = ax_err.plot([], [], 'b-', linewidth=2.5, label='PINN', alpha=0.8)
    line_err_base, = ax_err.plot([], [], 'r-', linewidth=2.5, label='Baseline', alpha=0.8)
    ax_err.legend(fontsize=9)
    
    # Cumulative error plot
    ax_cum = fig.add_subplot(gs[1, 2])
    ax_cum.set_xlim([0, duration])
    ax_cum.set_ylim([0, 100])
    ax_cum.set_xlabel('Time (s)', fontsize=10)
    ax_cum.set_ylabel('Cumulative Error (m·s)', fontsize=10)
    ax_cum.set_title('Cumulative Tracking Error', fontsize=11, fontweight='bold')
    ax_cum.grid(True, alpha=0.3)
    
    line_cum_pinn, = ax_cum.plot([], [], 'b-', linewidth=2.5, label='PINN', alpha=0.8)
    line_cum_base, = ax_cum.plot([], [], 'r-', linewidth=2.5, label='Baseline', alpha=0.8)
    ax_cum.legend(fontsize=9)
    
    # Statistics panel
    ax_stats = fig.add_subplot(gs[2, 2])
    ax_stats.axis('off')
    stats_text = ax_stats.text(0.1, 0.9, '', fontsize=10, verticalalignment='top',
                               family='monospace',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Info text
    info_text = fig.text(0.02, 0.98, '', fontsize=10, va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def init():
        line_pinn.set_data([], [])
        line_pinn.set_3d_properties([])
        line_base.set_data([], [])
        line_base.set_3d_properties([])
        line_target.set_data([], [])
        line_target.set_3d_properties([])
        return []
    
    def update(frame):
        nonlocal state_pinn, state_base
        
        t = frame * dt
        
        # Get current target
        target = target_func(t)
        target_traj.append(target.copy())
        
        # Update gust field
        avg_pos = 0.5 * (state_pinn[:3] + state_base[:3])
        gust_field.update(dt, avg_pos)
        
        # True disturbances
        gust_pinn = gust_field.get_force(state_pinn[:3], t)
        gust_base = gust_field.get_force(state_base[:3], t)
        gust_log.append(np.linalg.norm(gust_pinn))
        
        # PINN control
        gf_vec = encode_gust_features(state_pinn[:3], gust_field, t, max_bursts=2)
        trunk = encode_trunk_vec(state_pinn, gust_field, t)
        d_hat = network.forward(gf_vec.reshape(1, -1), trunk.reshape(1, -1))[0]
        
        u_pid_pinn = ctrl_pinn.compute(state_pinn[:3], target, dt)
        u_total_pinn = u_pid_pinn - d_hat
        
        dstate_pinn = quad_pinn.compute_dynamics(state_pinn, u_total_pinn, gust_pinn)
        state_pinn = state_pinn + dstate_pinn * dt
        
        # Baseline control
        u_pid_base = ctrl_base.compute(state_base[:3], target, dt)
        dstate_base = quad_base.compute_dynamics(state_base, u_pid_base, gust_base)
        state_base = state_base + dstate_base * dt
        
        # Log trajectories and errors
        traj_pinn.append(state_pinn[:3].copy())
        traj_base.append(state_base[:3].copy())
        times.append(t)
        
        err_p = np.linalg.norm(state_pinn[:3] - target)
        err_b = np.linalg.norm(state_base[:3] - target)
        error_pinn.append(err_p)
        error_base.append(err_b)
        
        # Update 3D plot
        arr_p = np.array(traj_pinn)
        arr_b = np.array(traj_base)
        arr_t = np.array(target_traj)
        
        line_pinn.set_data(arr_p[:, 0], arr_p[:, 1])
        line_pinn.set_3d_properties(arr_p[:, 2])
        
        line_base.set_data(arr_b[:, 0], arr_b[:, 1])
        line_base.set_3d_properties(arr_b[:, 2])
        
        line_target.set_data(arr_t[:, 0], arr_t[:, 1])
        line_target.set_3d_properties(arr_t[:, 2])
        
        drone_pinn._offsets3d = ([state_pinn[0]], [state_pinn[1]], [state_pinn[2]])
        drone_base._offsets3d = ([state_base[0]], [state_base[1]], [state_base[2]])
        target_point._offsets3d = ([target[0]], [target[1]], [target[2]])
        
        # Update error plots
        T = np.array(times)
        E_p = np.array(error_pinn)
        E_b = np.array(error_base)
        
        line_err_pinn.set_data(T, E_p)
        line_err_base.set_data(T, E_b)
        
        # Cumulative error
        cum_p = np.cumsum(E_p) * dt
        cum_b = np.cumsum(E_b) * dt
        
        line_cum_pinn.set_data(T, cum_p)
        line_cum_base.set_data(T, cum_b)
        
        # Update statistics
        mean_err_p = np.mean(E_p)
        mean_err_b = np.mean(E_b)
        max_err_p = np.max(E_p)
        max_err_b = np.max(E_b)
        improvement = (1 - mean_err_p / mean_err_b) * 100 if mean_err_b > 0 else 0
        
        stats_str = (
            f"STATISTICS (t={t:.1f}s)\n"
            f"{'─'*25}\n"
            f"Mean Error:\n"
            f"  PINN:     {mean_err_p:.2f} m\n"
            f"  Baseline: {mean_err_b:.2f} m\n"
            f"\n"
            f"Max Error:\n"
            f"  PINN:     {max_err_p:.2f} m\n"
            f"  Baseline: {max_err_b:.2f} m\n"
            f"\n"
            f"Cumulative:\n"
            f"  PINN:     {cum_p[-1]:.1f} m·s\n"
            f"  Baseline: {cum_b[-1]:.1f} m·s\n"
            f"\n"
            f"Improvement: {improvement:.1f}%"
        )
        stats_text.set_text(stats_str)
        
        # Update info
        mode_str = "STORM" if storm_mode else "NORMAL"
        info_str = (
            f"[{mode_str}] Pattern: {pattern_name.upper()} | "
            f"Time: {t:.1f}s | "
            f"Bursts: {len(gust_field.bursts)} | "
            f"Wind: {gust_log[-1]:.1f} N\n"
            f"PINN Error: {err_p:.2f} m | "
            f"Baseline Error: {err_b:.2f} m | "
            f"Improvement: {((err_b - err_p)/err_b*100):.1f}%"
        )
        info_text.set_text(info_str)
        
        return []
    
    anim = FuncAnimation(fig, update, init_func=init, frames=n_steps,
                        interval=50, blit=False, repeat=False)
    
    print(f"Animation created: {n_steps} frames at {1/dt:.0f} fps")
    print(f"Duration: {duration}s")
    
    return anim, fig


if __name__ == "__main__":
    from deeponet import DeepONet
    import sys
    
    # Load trained network
    network = DeepONet()
    try:
        network.load_weights("deeponet_weights.npz")
        print("✓ Loaded trained weights")
    except:
        print("✗ No trained weights found. Train first using training.py")
        sys.exit(1)
    
    # Get pattern from command line or use default
    patterns = ['stationary', 'circular', 'figure8', 'vertical', 'box3d', 'lissajous']
    
    if len(sys.argv) > 1 and sys.argv[1] in patterns:
        pattern = sys.argv[1]
    else:
        pattern = 'figure8'
        print(f"\nUsage: python {sys.argv[0]} [pattern]")
        print(f"Available patterns: {', '.join(patterns)}")
        print(f"Using default: {pattern}\n")
    
    # Create animation
    print(f"\nGenerating animation for {pattern} pattern...")
    anim, fig = create_animated_comparison_with_patterns(
        network,
        pattern_name=pattern,
        duration=30.0,
        dt=0.05,
        storm_mode=True
    )
    
    # Save animation (requires ffmpeg or pillow)
    try:
        filename = f'animated_{pattern}_storm.gif'
        anim.save(filename, writer='pillow', fps=20, dpi=100)
        print(f"\n✓ Animation saved: {filename}")
    except:
        print("\n⚠ Could not save animation (install pillow: pip install pillow)")
    
    plt.show()
    
    print("\n" + "="*70)
    print("To test other patterns, run:")
    for p in patterns:
        print(f"  python experiment_animated_patterns.py {p}")
    print("="*70)