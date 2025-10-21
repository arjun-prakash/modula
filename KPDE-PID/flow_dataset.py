#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================================
# GUST FIELD MODEL - Pulsed-Jet Bursts
# ============================================================================
class GustField:
    def __init__(self, max_bursts=5, spawn_radius=10.0, storm_mode=False):
        self.max_bursts = max_bursts
        self.spawn_radius = spawn_radius
        self.bursts = []
        self.storm_mode = storm_mode

        # Storm-specific parameters
        if storm_mode:
            self.base_wind = np.array([8.0, -6.0, 2.0], dtype=np.float32)
            self.wind_turbulence_freq = 0.5
            self.wind_turbulence_amp = 5.0
            self.max_bursts = max(self.max_bursts, 8)
        else:
            self.base_wind = np.zeros(3, dtype=np.float32)
            self.wind_turbulence_freq = 0.0
            self.wind_turbulence_amp = 0.0

    def update(self, dt, drone_pos):
        """Update burst dynamics and spawn new bursts."""
        self.bursts = [b for b in self.bursts if b['lifetime'] > 0]
        for burst in self.bursts:
            burst['lifetime'] -= dt

        spawn_prob = 0.15 if self.storm_mode else 0.1
        if len(self.bursts) < self.max_bursts and np.random.rand() < spawn_prob:
            self._spawn_burst(drone_pos)

    def _spawn_burst(self, drone_pos):
        """Create a new pulsed-jet burst directed at drone."""
        angle_theta = np.random.uniform(0, 2*np.pi)
        angle_phi = np.random.uniform(0, np.pi)
        distance = np.random.uniform(2.0, self.spawn_radius)

        position = drone_pos + distance * np.array([
            np.sin(angle_phi) * np.cos(angle_theta),
            np.sin(angle_phi) * np.sin(angle_theta),
            np.cos(angle_phi)
        ])

        if self.storm_mode:
            intensity = np.random.uniform(30.0, 70.0)
            lifetime = np.random.uniform(1.5, 4.0)
        else:
            intensity = np.random.uniform(15.0, 40.0)
            lifetime = np.random.uniform(0.5, 2.0)

        self.bursts.append({
            'position': position.astype(np.float32),
            'intensity': float(intensity),
            'lifetime': float(lifetime),
            'pulse_freq': float(np.random.uniform(2.0, 8.0))
        })

    def get_force(self, drone_pos, t):
        """Total gust force (base wind + active bursts)."""
        total_force = np.zeros(3, dtype=np.float32)

        if self.storm_mode:
            turbulence = self.wind_turbulence_amp * np.array([
                np.sin(2 * np.pi * self.wind_turbulence_freq * t),
                np.cos(2 * np.pi * self.wind_turbulence_freq * t * 1.3),
                np.sin(2 * np.pi * self.wind_turbulence_freq * t * 0.7)
            ], dtype=np.float32)
            total_force += self.base_wind + turbulence

        for burst in self.bursts:
            delta = drone_pos - burst['position']
            dist = np.linalg.norm(delta)
            if dist < 1e-3:
                continue
            direction = delta / dist
            pulse = 0.5 * (1 + np.sin(2 * np.pi * burst['pulse_freq'] * t))
            magnitude = burst['intensity'] * pulse / (1.0 + dist**2)
            total_force += (magnitude * direction).astype(np.float32)

        return total_force


# ============================================================================
# CONSTANT HORIZONTAL WIND FIELD (no bursts)
# ============================================================================
class ConstantWindField:
    """Simple field that returns a fixed horizontal wind; no bursts."""
    def __init__(self, wind_vec=np.array([15.0, 0.0, 0.0], dtype=np.float32)):
        self.wind_vec = wind_vec.astype(np.float32)
        self.bursts = []           # keep interface for encoders
        self.storm_mode = True     # so trunk encoder includes base wind
        self.base_wind = self.wind_vec
        self.wind_turbulence_freq = 0.0
        self.wind_turbulence_amp = 0.0

    def update(self, dt, pos):
        pass

    def get_force(self, pos, t):
        return self.wind_vec


# ============================================================================
# QUADCOPTER DYNAMICS
# ============================================================================
class QuadcopterDynamics:
    def __init__(self, mass=1.0, drag_coeff=0.1):
        self.mass = float(mass)
        self.drag = float(drag_coeff)
        self.g = 9.81

    def compute_dynamics(self, state, control_force, disturbance_force):
        """
        State: [x, y, z, vx, vy, vz]
        Returns: state derivative
        """
        vel = state[3:6]
        gravity = np.array([0, 0, -self.mass * self.g], dtype=np.float32)
        drag_force = -self.drag * vel
        total_force = control_force + disturbance_force + gravity + drag_force
        accel = total_force / self.mass
        return np.concatenate([vel, accel]).astype(np.float32)


# ============================================================================
# PID CONTROLLER
# ============================================================================
class PIDController:
    def __init__(self, kp=20.0, ki=0.5, kd=10.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral = np.zeros(3, dtype=np.float32)
        self.prev_error = np.zeros(3, dtype=np.float32)

    def compute(self, current_pos, target_pos, dt):
        """Compute PID control force."""
        error = (target_pos - current_pos).astype(np.float32)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -5, 5)  # anti-windup
        derivative = (error - self.prev_error) / dt if dt > 0 else np.zeros(3, dtype=np.float32)
        self.prev_error = error.copy()
        force = self.kp * error + self.ki * self.integral + self.kd * derivative
        return np.clip(force, -50, 50).astype(np.float32)

    def reset(self):
        self.integral[:] = 0.0
        self.prev_error[:] = 0.0


# ============================================================================
# DeepONet
# ============================================================================
class DeepONet:
    def __init__(self, branch_input_dim=10, trunk_input_dim=10, hidden_dim=64, output_dim=3):
        """
        Branch: encodes gust features (10-dim; see encoder below)
        Trunk:  encodes [state(6) + time_norm(1) + base_wind(3)] = 10 dims
        Output: predicted disturbance force (3)
        """
        rng = np.random.default_rng(0)

        def xavier_in(din, dout):
            return (rng.standard_normal((din, dout)).astype(np.float32) * np.sqrt(2.0 / din))

        self.W_branch1 = xavier_in(branch_input_dim, hidden_dim); self.b_branch1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W_branch2 = xavier_in(hidden_dim, hidden_dim);       self.b_branch2 = np.zeros(hidden_dim, dtype=np.float32)

        self.W_trunk1  = xavier_in(trunk_input_dim, hidden_dim);  self.b_trunk1  = np.zeros(hidden_dim, dtype=np.float32)
        self.W_trunk2  = xavier_in(hidden_dim, hidden_dim);       self.b_trunk2  = np.zeros(hidden_dim, dtype=np.float32)

        self.W_out     = xavier_in(hidden_dim, output_dim);       self.b_out     = np.zeros(output_dim, dtype=np.float32)

    @staticmethod
    def _relu(x): 
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_deriv_from_pre(z): 
        return (z > 0.0).astype(np.float32)

    def forward(self, gust_features, trunk_vec, return_cache=False):
        """gust_features: (B,10), trunk_vec: (B,10)."""
        gf = gust_features
        tv = trunk_vec
        if gf.ndim == 1: gf = gf.reshape(1, -1)
        if tv.ndim == 1: tv = tv.reshape(1, -1)

        # Branch
        z_b1 = gf @ self.W_branch1 + self.b_branch1
        b1   = self._relu(z_b1)
        z_b2 = b1 @ self.W_branch2 + self.b_branch2
        b2   = self._relu(z_b2)

        # Trunk
        z_t1 = tv @ self.W_trunk1 + self.b_trunk1
        t1   = self._relu(z_t1)
        z_t2 = t1 @ self.W_trunk2 + self.b_trunk2
        t2   = self._relu(z_t2)

        combined = b2 * t2
        out = combined @ self.W_out + self.b_out

        if not return_cache:
            return out
        cache = dict(gust_features=gf, trunk_vec=tv,
                     z_b1=z_b1, b1=b1, z_b2=z_b2, b2=b2,
                     z_t1=z_t1, t1=t1, z_t2=z_t2, t2=t2, combined=combined)
        return out, cache

    def backward(self, cache, grad_out, learning_rate=0.001, weight_decay=0.0):
        """Backprop for a mini-batch. grad_out shape (B,3)."""
        # Output layer
        grad_W_out = cache['combined'].T @ grad_out
        grad_b_out = np.sum(grad_out, axis=0)
        grad_comb  = grad_out @ self.W_out.T

        # Split to branch & trunk
        grad_b2 = grad_comb * cache['t2']
        grad_t2 = grad_comb * cache['b2']

        # Branch backprop (using PRE-activations)
        grad_b2_pre = grad_b2 * self._relu_deriv_from_pre(cache['z_b2'])
        grad_W_b2 = cache['b1'].T @ grad_b2_pre
        grad_b_b2 = np.sum(grad_b2_pre, axis=0)

        grad_b1 = grad_b2_pre @ self.W_branch2.T
        grad_b1_pre = grad_b1 * self._relu_deriv_from_pre(cache['z_b1'])
        grad_W_b1 = cache['gust_features'].T @ grad_b1_pre
        grad_b_b1 = np.sum(grad_b1_pre, axis=0)

        # Trunk backprop (using PRE-activations)
        grad_t2_pre = grad_t2 * self._relu_deriv_from_pre(cache['z_t2'])
        grad_W_t2 = cache['t1'].T @ grad_t2_pre
        grad_b_t2 = np.sum(grad_t2_pre, axis=0)

        grad_t1 = grad_t2_pre @ self.W_trunk2.T
        grad_t1_pre = grad_t1 * self._relu_deriv_from_pre(cache['z_t1'])
        grad_W_t1 = cache['trunk_vec'].T @ grad_t1_pre
        grad_b_t1 = np.sum(grad_t1_pre, axis=0)

        # Optional weight decay
        def decay(W): 
            return weight_decay * W if weight_decay > 0 else 0.0

        # Update
        self.W_out     -= learning_rate * (grad_W_out + decay(self.W_out))
        self.b_out     -= learning_rate * grad_b_out

        self.W_branch2 -= learning_rate * (grad_W_b2 + decay(self.W_branch2))
        self.b_branch2 -= learning_rate * grad_b_b2
        self.W_branch1 -= learning_rate * (grad_W_b1 + decay(self.W_branch1))
        self.b_branch1 -= learning_rate * grad_b_b1

        self.W_trunk2  -= learning_rate * (grad_W_t2 + decay(self.W_trunk2))
        self.b_trunk2  -= learning_rate * grad_b_t2
        self.W_trunk1  -= learning_rate * (grad_W_t1 + decay(self.W_trunk1))
        self.b_trunk1  -= learning_rate * grad_b_t1


# ============================================================================
# FEATURE ENCODERS
# ============================================================================
def encode_gust_features(drone_pos, gust_field, t, max_bursts=2):
    """
    10-dim vector = 2 bursts × 5 dims each:
      [relx/10, rely/10, relz/10, dist/10, intensity/70] per burst
    Pad with zeros if fewer than 2 bursts.
    """
    feats = np.zeros(10, dtype=np.float32)
    for i, burst in enumerate(gust_field.bursts[:max_bursts]):
        idx = i * 5
        rel = burst['position'] - drone_pos
        dist = np.linalg.norm(rel) + 1e-6
        feats[idx:idx+3] = (rel / 10.0).astype(np.float32)
        feats[idx+3] = float(dist / 10.0)
        feats[idx+4] = float(np.clip(burst['intensity'] / 70.0, 0, 1))
    return feats

def encode_trunk_vec(state6, gust_field, t):
    """
    10-dim trunk: [x,y,z,vx,vy,vz, t_norm, base_wind/10 (3)]
    """
    state6 = state6.astype(np.float32)
    t_norm = np.array([(t % 10.0) / 10.0], dtype=np.float32)
    if getattr(gust_field, 'storm_mode', False):
        bw = getattr(gust_field, 'base_wind', np.zeros(3, dtype=np.float32)).astype(np.float32)
        if getattr(gust_field, 'wind_turbulence_amp', 0.0) != 0.0:
            bw = bw + gust_field.wind_turbulence_amp * np.array([
                np.sin(2*np.pi*gust_field.wind_turbulence_freq * t),
                np.cos(2*np.pi*gust_field.wind_turbulence_freq * t * 1.3),
                np.sin(2*np.pi*gust_field.wind_turbulence_freq * t * 0.7)
            ], dtype=np.float32)
    else:
        bw = np.zeros(3, dtype=np.float32)
    return np.concatenate([state6, t_norm, (bw/10.0).astype(np.float32)], axis=0)


# ============================================================================
# PHYSICS-INFORMED TRAINING (supervised + dynamics residual)
# ============================================================================
def train_deeponet_physics(epochs=300, samples_per_epoch=200, batch_size=20,
                           lr0=5e-3, weight_decay=1e-4, lambda_dyn=0.3):
    """
    Loss = MSE(d_hat, d_true) + lambda_dyn * || m a_true - (u - m g e_z - c_d v + d_hat) ||^2
    where a_true is synthesized from sampled u, v, and world d_true.
    """
    print("Training Physics-Informed DeepONet...")
    network = DeepONet(branch_input_dim=10, trunk_input_dim=10, hidden_dim=64, output_dim=3)
    quad = QuadcopterDynamics(mass=1.0, drag_coeff=0.1)
    g = quad.g

    losses = []
    best = float('inf'); patience = 50; bad = 0

    for epoch in range(epochs):
        epoch_loss = 0.0

        # Collect epoch samples
        GF_list, TRUNK_list, TRUE_list, ATRUE_list, U_list, V_list = [], [], [], [], [], []

        for _ in range(samples_per_epoch):
            drone_pos = np.random.randn(3).astype(np.float32) * 4.0
            drone_vel = (np.random.randn(3) * 1.5).astype(np.float32)
            state = np.concatenate([drone_pos, drone_vel]).astype(np.float32)

            gust_field = GustField(max_bursts=4, storm_mode=(np.random.rand() < 0.5))
            n_bursts = np.random.randint(0, 4)
            for _ in range(n_bursts):
                gust_field._spawn_burst(drone_pos)

            t = float(np.random.rand() * 10.0)
            d_true = gust_field.get_force(drone_pos, t).astype(np.float32)

            u = (np.random.randn(3) * 6.0).astype(np.float32)
            gravity = np.array([0, 0, -quad.mass * g], dtype=np.float32)
            drag = -quad.drag * drone_vel
            a_true = (u + d_true + drag + gravity) / quad.mass

            gf = encode_gust_features(drone_pos, gust_field, t, max_bursts=2)
            trunk = encode_trunk_vec(state, gust_field, t)

            GF_list.append(gf); TRUNK_list.append(trunk); TRUE_list.append(d_true)
            ATRUE_list.append(a_true); U_list.append(u); V_list.append(drone_vel)

        GF = np.array(GF_list); TR = np.array(TRUNK_list); TRUE = np.array(TRUE_list)
        ATRUE = np.array(ATRUE_list); UU = np.array(U_list); VV = np.array(V_list)

        num_batches = samples_per_epoch // batch_size
        for b in range(num_batches):
            s = b * batch_size; e = s + batch_size
            gf = GF[s:e]; tr = TR[s:e]; d_true = TRUE[s:e]
            a_true = ATRUE[s:e]; u = UU[s:e]; v = VV[s:e]

            pred, cache = network.forward(gf, tr, return_cache=True)

            # c = m a_true - (u - m g e_z - c_d v)
            m = quad.mass; cd = quad.drag
            c = (m * a_true) - (u - np.array([0, 0, m * g], dtype=np.float32) - cd * v)

            err = pred - d_true
            mse = np.mean(err**2)
            resid = pred - c
            l_dyn = np.mean(resid**2)

            total = mse + lambda_dyn * l_dyn
            epoch_loss += total

            grad_pred = (2.0 / (batch_size * 3)) * (err + lambda_dyn * (pred - c))
            grad_pred = np.clip(grad_pred, -1.0, 1.0)

            lr = lr0 * (0.98 ** (epoch // 10))
            network.backward(cache, grad_pred, learning_rate=lr, weight_decay=weight_decay)

        avg = epoch_loss / num_batches
        losses.append(avg)

        if avg < best:
            best = avg; bad = 0
        else:
            bad += 1
        if bad >= patience:
            print(f"Early stopping @ epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{epochs}, loss={avg:.6f}, best={best:.6f}, lr={lr:.6f}")

    print(f"Training complete! Final loss: {avg:.6f}, Best loss: {best:.6f}")
    return network, losses


# ============================================================================
# COMPARISON SIM (Plain feed-forward: u_total = u_pid - d_hat)
# ============================================================================
def run_comparison_simulation(network, duration=25.0, dt=0.02, storm_mode=False):
    """Baseline uses PID; PINN uses PID - d_hat (no gating, no EMA)."""
    gust_field = GustField(max_bursts=4 if not storm_mode else 8, spawn_radius=8.0, storm_mode=storm_mode)

    quad_pinn = QuadcopterDynamics(mass=1.0)
    quad_baseline = QuadcopterDynamics(mass=1.0)

    controller_pinn = PIDController(kp=15.0, ki=0.3, kd=8.0)
    controller_baseline = PIDController(kp=15.0, ki=0.3, kd=8.0)

    state_pinn = np.array([3.0, 2.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    state_baseline = state_pinn.copy()
    target = np.zeros(3, dtype=np.float32)

    n = int(duration / dt)
    traj_pinn = np.zeros((n, 6), dtype=np.float32)
    traj_base = np.zeros((n, 6), dtype=np.float32)
    gust_forces = np.zeros((n, 3), dtype=np.float32)
    times = np.zeros(n, dtype=np.float32)

    print(f"\nRunning comparison simulation in {'STORM MODE' if storm_mode else 'NORMAL MODE'}...")
    if storm_mode:
        print("⚠️  EXTREME CONDITIONS")

    for i in range(n):
        t = i * dt
        times[i] = t

        avg_pos = 0.5 * (state_pinn[:3] + state_baseline[:3])
        gust_field.update(dt, avg_pos)

        # TRUE disturbances
        gust_pinn = gust_field.get_force(state_pinn[:3], t)
        gust_base = gust_field.get_force(state_baseline[:3], t)
        gust_forces[i] = gust_pinn

        # PINN-augmented (plain FF)
        gf_vec = encode_gust_features(state_pinn[:3], gust_field, t, max_bursts=2)
        trunk = encode_trunk_vec(state_pinn, gust_field, t)
        d_hat = network.forward(gf_vec.reshape(1, -1), trunk.reshape(1, -1))[0]

        u_pid = controller_pinn.compute(state_pinn[:3], target, dt)
        u_total = u_pid - d_hat

        dstate_pinn = quad_pinn.compute_dynamics(state_pinn, u_total, gust_pinn)
        state_pinn = state_pinn + dstate_pinn * dt
        traj_pinn[i] = state_pinn

        # Baseline
        u_base = controller_baseline.compute(state_baseline[:3], target, dt)
        dstate_base = quad_baseline.compute_dynamics(state_baseline, u_base, gust_base)
        state_baseline = state_baseline + dstate_base * dt
        traj_base[i] = state_baseline

    print("Comparison simulation complete!")
    return times, traj_pinn, traj_base, gust_forces, gust_field


# ============================================================================
# ANIMATED SIM (keeps visuals; plain FF)
# ============================================================================
def create_animated_simulation(network, duration=30.0, dt=0.05, storm_mode=False):
    """Animated visualization; visuals unchanged; uses u_total = u_pid - d_hat."""
    gust_field = GustField(max_bursts=5, spawn_radius=10.0, storm_mode=storm_mode)
    quad = QuadcopterDynamics(mass=1.0)
    controller = PIDController(kp=15.0, ki=0.3, kd=8.0)

    state = np.array([4.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float32)

    target = np.array([2.0, -2.0, 1.5], dtype=np.float32)
    target_radius = 0.8
    targets_reached = 0

    n_steps = int(duration / dt)
    trajectory = [state[:3].copy()]
    velocity = [state[3:6].copy()]
    gust_forces_log = [0.0]
    burst_positions = []
    times = [0]
    target_history = [target.copy()]

    # Figure layout (kept from your style)
    title_suffix = " - STORM MODE ⚠️" if storm_mode else ""
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax_3d.set_xlim([-8, 8]); ax_3d.set_ylim([-8, 8]); ax_3d.set_zlim([-2, 8])
    ax_3d.set_xlabel('X (m)', fontsize=10); ax_3d.set_ylabel('Y (m)', fontsize=10); ax_3d.set_zlabel('Z (m)', fontsize=10)
    ax_3d.set_title(f'3D Trajectory & Gust Field{title_suffix}', fontsize=12, fontweight='bold')

    trajectory_line, = ax_3d.plot([], [], [], 'b-', linewidth=2, alpha=0.6, label='Path')
    drone_point = ax_3d.scatter([], [], [], c='lime', s=300, marker='o',
                                edgecolors='darkgreen', linewidths=2, label='Drone')
    target_point = ax_3d.scatter([target[0]], [target[1]], [target[2]], c='red', s=400, marker='X',
                                 edgecolors='darkred', linewidths=2, label='Target')
    burst_scatter = ax_3d.scatter([], [], [], c='orange', s=150, marker='*',
                                  alpha=0.6, edgecolors='darkorange', linewidths=1.5)
    ax_3d.legend(loc='upper right', fontsize=9); ax_3d.grid(True, alpha=0.3)

    ax_pos = fig.add_subplot(gs[0, 1:])
    ax_pos.set_xlim([0, duration]); ax_pos.set_ylim([-6, 6])
    ax_pos.set_ylabel('Position (m)', fontsize=10)
    ax_pos.set_title('Position vs Time', fontsize=11, fontweight='bold')
    ax_pos.grid(True, alpha=0.3)
    line_x, = ax_pos.plot([], [], 'r-', linewidth=2, label='X', alpha=0.8)
    line_y, = ax_pos.plot([], [], 'g-', linewidth=2, label='Y', alpha=0.8)
    line_z, = ax_pos.plot([], [], 'b-', linewidth=2, label='Z', alpha=0.8)
    ax_pos.axhline(y=0, color='k', linestyle='--', alpha=0.4, linewidth=1)
    ax_pos.legend(loc='upper right', fontsize=9, ncol=3)

    ax_vel = fig.add_subplot(gs[1, 1:])
    ax_vel.set_xlim([0, duration]); ax_vel.set_ylim([-4, 4])
    ax_vel.set_ylabel('Velocity (m/s)', fontsize=10)
    ax_vel.set_title('Velocity vs Time', fontsize=11, fontweight='bold')
    ax_vel.grid(True, alpha=0.3)
    line_vx, = ax_vel.plot([], [], 'r-', linewidth=2, label='Vx', alpha=0.8)
    line_vy, = ax_vel.plot([], [], 'g-', linewidth=2, label='Vy', alpha=0.8)
    line_vz, = ax_vel.plot([], [], 'b-', linewidth=2, label='Vz', alpha=0.8)
    ax_vel.legend(loc='upper right', fontsize=9, ncol=3)

    ax_err = fig.add_subplot(gs[2, 1])
    ax_err.set_xlim([0, duration]); ax_err.set_ylim([0, 12])
    ax_err.set_xlabel('Time (s)', fontsize=10)
    ax_err.set_ylabel('Distance (m)', fontsize=10)
    ax_err.set_title('Tracking Error', fontsize=11, fontweight='bold')
    ax_err.grid(True, alpha=0.3)
    line_err, = ax_err.plot([], [], 'purple', linewidth=2.5, alpha=0.8)
    ax_err.axhline(y=target_radius, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label=f'Target ({target_radius}m)')
    ax_err.legend(fontsize=9)

    ax_gust = fig.add_subplot(gs[2, 2])
    ax_gust.set_xlim([0, duration]); ax_gust.set_ylim([0, 100 if storm_mode else 60])
    ax_gust.set_xlabel('Time (s)', fontsize=10)
    ax_gust.set_ylabel('Force Magnitude (N)', fontsize=10)
    ax_gust.set_title('Gust Disturbance', fontsize=11, fontweight='bold')
    ax_gust.grid(True, alpha=0.3)
    line_gust, = ax_gust.plot([], [], 'orange', linewidth=2.5, alpha=0.8)

    info_text = fig.text(0.02, 0.98, '', fontsize=10, va='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def init():
        trajectory_line.set_data([], [])
        trajectory_line.set_3d_properties([])
        return []

    def update(frame):
        nonlocal state, target, targets_reached
        t = frame * dt

        # Target switching when reached
        dist_to_target = np.linalg.norm(state[:3] - target)
        if dist_to_target < target_radius:
            target = np.random.uniform(-6, 6, 3).astype(np.float32)
            target[2] = np.random.uniform(0.5, 5.0)
            targets_reached += 1
            target_history.append(target.copy())
            print(f"Target reached! New target at [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]. Total: {targets_reached}")

        # World update
        gust_field.update(dt, state[:3])
        true_gust = gust_field.get_force(state[:3], t)

        # PINN prediction
        gust_features = encode_gust_features(state[:3], gust_field, t, max_bursts=2)
        trunk_vec = encode_trunk_vec(state, gust_field, t)
        d_hat = network.forward(gust_features.reshape(1, -1), trunk_vec.reshape(1, -1))[0]

        # Plain feed-forward cancellation
        control_force = controller.compute(state[:3], target, dt)
        u_total = control_force - d_hat

        # Dynamics (world uses TRUE gust)
        dstate = quad.compute_dynamics(state, u_total, true_gust)
        state = state + dstate * dt

        # logs
        trajectory.append(state[:3].copy())
        velocity.append(state[3:6].copy())
        gust_forces_log.append(np.linalg.norm(true_gust))
        times.append(t)
        burst_positions[:] = [b['position'] for b in gust_field.bursts]

        # plots
        traj_array = np.array(trajectory)
        vel_array = np.array(velocity)
        times_array = np.array(times)

        trajectory_line.set_data(traj_array[:, 0], traj_array[:, 1])
        trajectory_line.set_3d_properties(traj_array[:, 2])

        drone_point._offsets3d = ([state[0]], [state[1]], [state[2]])
        target_point._offsets3d = ([target[0]], [target[1]], [target[2]])

        if burst_positions:
            bp_array = np.array(burst_positions)
            burst_scatter._offsets3d = (bp_array[:, 0], bp_array[:, 1], bp_array[:, 2])
        else:
            burst_scatter._offsets3d = ([], [], [])

        line_x.set_data(times_array, traj_array[:, 0])
        line_y.set_data(times_array, traj_array[:, 1])
        line_z.set_data(times_array, traj_array[:, 2])

        line_vx.set_data(times_array, vel_array[:, 0])
        line_vy.set_data(times_array, vel_array[:, 1])
        line_vz.set_data(times_array, vel_array[:, 2])

        errors = np.linalg.norm(traj_array - target, axis=1)
        line_err.set_data(times_array, errors)
        line_gust.set_data(times_array, np.array(gust_forces_log))

        mode_indicator = "⚠️ STORM" if storm_mode else "NORMAL"
        info_str = (f'[{mode_indicator}] Time: {t:.1f}s | Pos: [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}] m\n'
                    f'Target: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}] | '
                    f'Error: {dist_to_target:.2f} m | Targets: {targets_reached} | '
                    f'Bursts: {len(gust_field.bursts)} | Gust: {np.linalg.norm(true_gust):.1f} N')
        info_text.set_text(info_str)
        return []

    anim = FuncAnimation(fig, update, init_func=init, frames=n_steps, interval=50, blit=False, repeat=False)
    return anim, fig


# ============================================================================
# FREE-FLIGHT (NO GUSTS) with MOVING TARGET — animation
# ============================================================================
def create_animated_free_flight_moving_target_no_gusts(duration=30.0, dt=0.05):
    """Animation: one drone, NO gusts, PID tracks a smooth Lissajous target."""
    quad = QuadcopterDynamics(mass=1.0)
    controller = PIDController(kp=15.0, ki=0.3, kd=8.0)

    state = np.array([0.0, 0.0, 1.5, 0.0, 0.0, 0.0], dtype=np.float32)

    n_steps = int(duration / dt)
    traj, vel, times, err_hist, gust_hist = [], [], [], [], []

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax_3d.set_xlim([-8, 8]); ax_3d.set_ylim([-8, 8]); ax_3d.set_zlim([-2, 8])
    ax_3d.set_title('Free Flight (No Gusts): 3D Trajectory', fontsize=12, fontweight='bold')
    path_line, = ax_3d.plot([], [], [], 'b-', lw=2, alpha=0.7, label='Path')
    drone_pt = ax_3d.scatter([], [], [], c='lime', s=300, marker='o',
                             edgecolors='darkgreen', linewidths=2, label='Drone')
    tgt_pt = ax_3d.scatter([], [], [], c='red', s=400, marker='X',
                           edgecolors='darkred', linewidths=2, label='Target')
    ax_3d.legend(fontsize=9); ax_3d.grid(True, alpha=0.3)

    ax_pos = fig.add_subplot(gs[0, 1:])
    ax_pos.set_xlim([0, duration]); ax_pos.set_ylim([-8, 8])
    line_x, = ax_pos.plot([], [], 'r-', lw=2, label='X')
    line_y, = ax_pos.plot([], [], 'g-', lw=2, label='Y')
    line_z, = ax_pos.plot([], [], 'b-', lw=2, label='Z')
    ax_pos.legend(fontsize=9); ax_pos.grid(True, alpha=0.3)

    ax_vel = fig.add_subplot(gs[1, 1:])
    ax_vel.set_xlim([0, duration]); ax_vel.set_ylim([-4, 4])
    line_vx, = ax_vel.plot([], [], 'r-', lw=2, label='Vx')
    line_vy, = ax_vel.plot([], [], 'g-', lw=2, label='Vy')
    line_vz, = ax_vel.plot([], [], 'b-', lw=2, label='Vz')
    ax_vel.legend(fontsize=9); ax_vel.grid(True, alpha=0.3)

    ax_err = fig.add_subplot(gs[2, 1])
    ax_err.set_xlim([0, duration]); ax_err.set_ylim([0, 8])
    line_err, = ax_err.plot([], [], 'purple', lw=2.5, alpha=0.8)
    ax_err.set_title('Tracking Error'); ax_err.grid(True, alpha=0.3)

    ax_gust = fig.add_subplot(gs[2, 2])
    ax_gust.set_xlim([0, duration]); ax_gust.set_ylim([0, 1])
    line_gust, = ax_gust.plot([], [], 'orange', lw=2.5, alpha=0.8)
    ax_gust.set_title('Gust Disturbance (Zero)'); ax_gust.grid(True, alpha=0.3)

    info = fig.text(0.02, 0.98, '', fontsize=10, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def target_at(t):
        x = 5.0 * np.sin(0.4 * t)
        y = 5.0 * np.sin(0.3 * t + np.pi/3)
        z = 2.0 + 1.0 * np.sin(0.5 * t)
        return np.array([x, y, z], dtype=np.float32)

    def init():
        path_line.set_data([], []); path_line.set_3d_properties([])
        return []

    def update(k):
        t = k * dt
        tgt = target_at(t)
        gust = np.zeros(3, dtype=np.float32)

        u = controller.compute(state[:3], tgt, dt)
        dstate = quad.compute_dynamics(state, u, gust)
        state[:] = state + dstate * dt

        traj.append(state[:3].copy()); vel.append(state[3:6].copy())
        times.append(t); err_hist.append(np.linalg.norm(state[:3]-tgt)); gust_hist.append(0.0)

        arr = np.array(traj); arrv = np.array(vel); T = np.array(times)

        path_line.set_data(arr[:,0], arr[:,1]); path_line.set_3d_properties(arr[:,2])
        drone_pt._offsets3d = ([state[0]],[state[1]],[state[2]])
        tgt_pt._offsets3d = ([tgt[0]],[tgt[1]],[tgt[2]])

        line_x.set_data(T, arr[:,0]); line_y.set_data(T, arr[:,1]); line_z.set_data(T, arr[:,2])
        line_vx.set_data(T, arrv[:,0]); line_vy.set_data(T, arrv[:,1]); line_vz.set_data(T, arrv[:,2])
        line_err.set_data(T, np.array(err_hist)); line_gust.set_data(T, np.array(gust_hist))

        info.set_text(f"[FREE FLIGHT] t={t:.1f}s  pos=[{state[0]:.2f},{state[1]:.2f},{state[2]:.2f}]  "
                      f"tgt=[{tgt[0]:.2f},{tgt[1]:.2f},{tgt[2]:.2f}]  err={err_hist[-1]:.2f}m")
        return []

    anim = FuncAnimation(fig, update, init_func=init, frames=n_steps, interval=50, blit=False, repeat=False)
    return anim, fig


# ============================================================================
# V-STACK MULTI-DRONE under STRONG HORIZONTAL WIND (simulation + quick plot)
# ============================================================================
def run_vstack_simulation_constant_wind(network, num_drones=5, spacing=1.0, duration=20.0, dt=0.02):
    """
    Simulate multiple drones in a vertical stack under strong +X wind.
    PINN FF uses u_total = u_pid - d_hat (no gating/EMA).
    Returns times and trajectories (dict with 'pinn' and 'baseline': arrays [N,T,6]).
    """
    wind = ConstantWindField(wind_vec=np.array([18.0, 0.0, 0.0], dtype=np.float32))
    quad = QuadcopterDynamics(mass=1.0)
    target = np.zeros(3, dtype=np.float32)

    ctrls_pinn = [PIDController(kp=15.0, ki=0.3, kd=8.0) for _ in range(num_drones)]
    ctrls_base = [PIDController(kp=15.0, ki=0.3, kd=8.0) for _ in range(num_drones)]

    states_pinn = np.zeros((num_drones, 6), dtype=np.float32)
    states_base = np.zeros((num_drones, 6), dtype=np.float32)
    for i in range(num_drones):
        z0 = 0.5 + i * spacing
        states_pinn[i, :3] = np.array([3.0, 0.0, z0], dtype=np.float32)
        states_base[i, :3] = states_pinn[i, :3].copy()

    steps = int(duration / dt)
    times = np.arange(steps, dtype=np.float32) * dt

    traj_pinn = np.zeros((num_drones, steps, 6), dtype=np.float32)
    traj_base = np.zeros((num_drones, steps, 6), dtype=np.float32)

    for k in range(steps):
        t = times[k]
        for i in range(num_drones):
            d_true_p = wind.get_force(states_pinn[i, :3], t)
            d_true_b = wind.get_force(states_base[i, :3], t)

            # PINN FF (plain)
            gf_vec = encode_gust_features(states_pinn[i, :3], wind, t, max_bursts=2)
            trunk = encode_trunk_vec(states_pinn[i, :], wind, t)
            d_hat = network.forward(gf_vec.reshape(1,-1), trunk.reshape(1,-1))[0]

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

    return times, {'pinn': traj_pinn, 'baseline': traj_base}


def plot_vstack_topdown(times, trajs, title='V-Stack under Strong +X Wind'):
    """Top-down XY for first and last drone, and Z stacks at final time."""
    tp = trajs['pinn']; tb = trajs['baseline']  # [N,T,6]
    N, T, _ = tp.shape

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(tp[0,:,0], tp[0,:,1], 'b-', lw=2, label='PINN d0')
    axs[0].plot(tb[0,:,0], tb[0,:,1], 'r--', lw=2, label='Base d0')
    axs[0].set_aspect('equal', 'box'); axs[0].set_title('Drone 0 (Top-Down)')
    axs[0].grid(True, alpha=0.3); axs[0].legend()

    axs[1].plot(tp[-1,:,0], tp[-1,:,1], 'b-', lw=2, label='PINN dN-1')
    axs[1].plot(tb[-1,:,0], tb[-1,:,1], 'r--', lw=2, label='Base dN-1')
    axs[1].set_aspect('equal', 'box'); axs[1].set_title('Drone N-1 (Top-Down)')
    axs[1].grid(True, alpha=0.3); axs[1].legend()

    z_p = tp[:,-1,2]; z_b = tb[:,-1,2]
    axs[2].plot(np.arange(N), z_p, 'bo-', label='PINN Z@final')
    axs[2].plot(np.arange(N), z_b, 'ro--', label='Base Z@final')
    axs[2].set_title('Final Z across Stack'); axs[2].grid(True, alpha=0.3); axs[2].legend()
    fig.suptitle(title, fontsize=14, fontweight='bold'); plt.tight_layout()
    return fig


# ============================================================================
# Static comparison analysis (kept from your style)
# ============================================================================
def static_comparison_analysis(times, traj_pinn, traj_baseline, gust_forces):
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

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
    ax1.set_xlabel('X (m)', fontsize=11); ax1.set_ylabel('Y (m)', fontsize=11); ax1.set_zlabel('Z (m)', fontsize=11)
    ax1.set_title('3D Trajectory Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right'); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2:])
    error_pinn = np.linalg.norm(traj_pinn[:, :3], axis=1)
    error_baseline = np.linalg.norm(traj_baseline[:, :3], axis=1)
    ax2.plot(times, error_pinn, 'b-', linewidth=2.5, label='PINN-Augmented', alpha=0.8)
    ax2.plot(times, error_baseline, 'r-', linewidth=2.5, label='Baseline PID', alpha=0.8)
    ax2.fill_between(times, error_pinn, alpha=0.2, color='blue')
    ax2.fill_between(times, error_baseline, alpha=0.2, color='red')
    ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.6, linewidth=1.5, label='Good (<0.5m)')
    ax2.set_ylabel('Distance from Target (m)', fontsize=11)
    ax2.set_title('Tracking Error Comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 2:])
    vel_pinn = np.linalg.norm(traj_pinn[:, 3:6], axis=1)
    vel_baseline = np.linalg.norm(traj_baseline[:, 3:6], axis=1)
    ax3.plot(times, vel_pinn, 'b-', linewidth=2.5, label='PINN-Augmented', alpha=0.8)
    ax3.plot(times, vel_baseline, 'r-', linewidth=2.5, label='Baseline PID', alpha=0.8)
    ax3.set_ylabel('Velocity Magnitude (m/s)', fontsize=11)
    ax3.set_title('Velocity Comparison', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10); ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(traj_pinn[:, 0], traj_pinn[:, 1], 'b-', linewidth=2.5, alpha=0.7, label='PINN')
    ax4.plot(traj_baseline[:, 0], traj_baseline[:, 1], 'r-', linewidth=2.5, alpha=0.7, label='Baseline')
    ax4.scatter([0], [0], c='green', marker='X', s=400,
                edgecolors='darkgreen', linewidths=2, label='Target', zorder=10)
    ax4.set_xlabel('X (m)', fontsize=11); ax4.set_ylabel('Y (m)', fontsize=11)
    ax4.set_title('Top-Down View (XY)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3); ax4.axis('equal')

    ax5 = fig.add_subplot(gs[2, 1])
    metrics = ['Mean', 'Max', 'Final', 'RMS']
    pinn_stats = [np.mean(error_pinn), np.max(error_pinn), error_pinn[-1], np.sqrt(np.mean(error_pinn**2))]
    baseline_stats = [np.mean(error_baseline), np.max(error_baseline), error_baseline[-1], np.sqrt(np.mean(error_baseline**2))]
    x = np.arange(len(metrics)); width = 0.35
    ax5.bar(x - width/2, pinn_stats, width, label='PINN', color='blue', alpha=0.7)
    ax5.bar(x + width/2, baseline_stats, width, label='Baseline', color='red', alpha=0.7)
    ax5.set_ylabel('Error (m)', fontsize=11)
    ax5.set_title('Error Statistics', fontsize=13, fontweight='bold')
    ax5.set_xticks(x); ax5.set_xticklabels(metrics, fontsize=10)
    ax5.legend(fontsize=10); ax5.grid(True, alpha=0.3, axis='y')

    ax6 = fig.add_subplot(gs[2, 2:])
    gust_mag = np.linalg.norm(gust_forces, axis=1)
    ax6.plot(times, gust_mag, 'orange', linewidth=2.5, alpha=0.8, label='Total Gust')
    ax6.fill_between(times, 0, gust_mag, alpha=0.3, color='orange')
    if np.mean(gust_mag) > 15:
        base_wind_mag = np.sqrt(8**2 + 6**2 + 2**2)
        ax6.axhline(y=base_wind_mag, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Base Wind (~{base_wind_mag:.1f}N)')
        ax6.legend(fontsize=10)
    ax6.set_xlabel('Time (s)', fontsize=11); ax6.set_ylabel('Force (N)', fontsize=11)
    ax6.set_title('Gust Disturbance Profile', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    plt.suptitle('PINN-Augmented vs Baseline PID Controller Comparison', fontsize=16, fontweight='bold', y=0.995)

    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(f"{'Metric':<20} {'PINN-Augmented':<20} {'Baseline PID':<20} {'Improvement':<15}")
    print("-"*70)
    print(f"{'Mean Error (m)':<20} {pinn_stats[0]:<20.3f} {baseline_stats[0]:<20.3f} {(1-pinn_stats[0]/baseline_stats[0])*100:>13.1f}%")
    print(f"{'Max Error (m)':<20} {pinn_stats[1]:<20.3f} {baseline_stats[1]:<20.3f} {(1-pinn_stats[1]/baseline_stats[1])*100:>13.1f}%")
    print(f"{'Final Error (m)':<20} {pinn_stats[2]:<20.3f} {baseline_stats[2]:<20.3f} {(1-pinn_stats[2]/baseline_stats[2])*100:>13.1f}%")
    print(f"{'RMS Error (m)':<20} {pinn_stats[3]:<20.3f} {baseline_stats[3]:<20.3f} {(1-pinn_stats[3]/baseline_stats[3])*100:>13.1f}%")
    print("="*70)

    return fig

def create_animated_moving_target_extreme_wind(network, duration=30.0, dt=0.05,
                                               wind_mode="storm",
                                               constant_wind_vec=np.array([20.0, -10.0, 3.0], dtype=np.float32),
                                               target_path="lissajous"):
    """
    Animated sim: moving target + EXTREME wind.
    - wind_mode: "storm" (bursts + base wind) or "constant" (strong horizontal wind)
    - target_path: "lissajous" | "circle" | "figure8"
    Uses plain feed-forward cancellation u_total = u_pid - d_hat.
    """
    # ===== World (extreme wind) =====
    if wind_mode == "storm":
        gust_field = GustField(max_bursts=8, spawn_radius=10.0, storm_mode=True)
        title_suffix = " - MOVING TARGET (STORM MODE ⚠️)"
    elif wind_mode == "constant":
        gust_field = ConstantWindField(wind_vec=constant_wind_vec.astype(np.float32))
        title_suffix = " - MOVING TARGET (STRONG CONSTANT WIND ⚠️)"
    else:
        raise ValueError("wind_mode must be 'storm' or 'constant'")

    quad = QuadcopterDynamics(mass=1.0)
    controller = PIDController(kp=15.0, ki=0.3, kd=8.0)

    # ===== Initial state =====
    state = np.array([0.0, -3.0, 1.5, 0.0, 0.0, 0.0], dtype=np.float32)

    # ===== Moving target path =====
    def target_at(t):
        if target_path == "lissajous":
            x = 5.0 * np.sin(0.5 * t)
            y = 5.0 * np.sin(0.35 * t + np.pi/3)
            z = 2.0 + 1.0 * np.sin(0.7 * t)
        elif target_path == "circle":
            r = 5.0; w = 0.4
            x = r * np.cos(w * t)
            y = r * np.sin(w * t)
            z = 2.0 + 0.5 * np.sin(0.8 * t)
        elif target_path == "figure8":
            r = 4.5; w = 0.45
            x = r * np.sin(w * t)
            y = r * np.sin(w * t) * np.cos(w * t)
            z = 2.0 + 1.0 * np.sin(0.6 * t)
        else:
            raise ValueError("target_path must be 'lissajous', 'circle', or 'figure8'")
        return np.array([x, y, z], dtype=np.float32)

    # ===== Seed logs consistently at t=0 =====
    t0 = 0.0
    tgt0 = target_at(t0)
    gust0 = gust_field.get_force(state[:3], t0)

    trajectory = [state[:3].copy()]
    velocity = [state[3:6].copy()]
    times = [t0]
    gust_forces_log = [float(np.linalg.norm(gust0))]
    err_log = [float(np.linalg.norm(state[:3] - tgt0))]

    # ===== Figure (kept style) =====
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax_3d.set_xlim([-10, 10]); ax_3d.set_ylim([-10, 10]); ax_3d.set_zlim([-2, 8])
    ax_3d.set_xlabel('X (m)', fontsize=10); ax_3d.set_ylabel('Y (m)', fontsize=10); ax_3d.set_zlabel('Z (m)', fontsize=10)
    ax_3d.set_title(f'3D Trajectory & Gust Field{title_suffix}', fontsize=12, fontweight='bold')
    path_line, = ax_3d.plot([], [], [], 'b-', lw=2, alpha=0.7, label='Path')
    drone_pt = ax_3d.scatter([], [], [], c='lime', s=300, marker='o',
                             edgecolors='darkgreen', linewidths=2, label='Drone')
    tgt_pt = ax_3d.scatter([], [], [], c='red', s=400, marker='X',
                           edgecolors='darkred', linewidths=2, label='Target')
    burst_scatter = ax_3d.scatter([], [], [], c='orange', s=150, marker='*',
                                  alpha=0.6, edgecolors='darkorange', linewidths=1.5)
    ax_3d.legend(loc='upper right', fontsize=9); ax_3d.grid(True, alpha=0.3)

    duration = float(duration)
    n_steps = int(duration / dt)

    ax_pos = fig.add_subplot(gs[0, 1:])
    ax_pos.set_xlim([0, duration]); ax_pos.set_ylim([-8, 8])
    ax_pos.set_ylabel('Position (m)', fontsize=10)
    ax_pos.set_title('Position vs Time', fontsize=11, fontweight='bold'); ax_pos.grid(True, alpha=0.3)
    line_x, = ax_pos.plot([], [], 'r-', lw=2, label='X', alpha=0.8)
    line_y, = ax_pos.plot([], [], 'g-', lw=2, label='Y', alpha=0.8)
    line_z, = ax_pos.plot([], [], 'b-', lw=2, label='Z', alpha=0.8)
    ax_pos.axhline(y=0, color='k', linestyle='--', alpha=0.4, lw=1); ax_pos.legend(loc='upper right', fontsize=9, ncol=3)

    ax_vel = fig.add_subplot(gs[1, 1:])
    ax_vel.set_xlim([0, duration]); ax_vel.set_ylim([-5, 5])
    ax_vel.set_ylabel('Velocity (m/s)', fontsize=10)
    ax_vel.set_title('Velocity vs Time', fontsize=11, fontweight='bold'); ax_vel.grid(True, alpha=0.3)
    line_vx, = ax_vel.plot([], [], 'r-', lw=2, label='Vx', alpha=0.8)
    line_vy, = ax_vel.plot([], [], 'g-', lw=2, label='Vy', alpha=0.8)
    line_vz, = ax_vel.plot([], [], 'b-', lw=2, label='Vz', alpha=0.8)
    ax_vel.legend(loc='upper right', fontsize=9, ncol=3)

    ax_err = fig.add_subplot(gs[2, 1])
    ax_err.set_xlim([0, duration]); ax_err.set_ylim([0, 12])
    ax_err.set_xlabel('Time (s)', fontsize=10); ax_err.set_ylabel('Distance (m)', fontsize=10)
    ax_err.set_title('Tracking Error', fontsize=11, fontweight='bold'); ax_err.grid(True, alpha=0.3)
    line_err, = ax_err.plot([], [], 'purple', lw=2.5, alpha=0.8)

    ax_gust = fig.add_subplot(gs[2, 2])
    ax_gust.set_xlim([0, duration]); ax_gust.set_ylim([0, 120])
    ax_gust.set_xlabel('Time (s)', fontsize=10); ax_gust.set_ylabel('Force Magnitude (N)', fontsize=10)
    ax_gust.set_title('Gust / Wind Disturbance', fontsize=11, fontweight='bold'); ax_gust.grid(True, alpha=0.3)
    line_gust, = ax_gust.plot([], [], 'orange', lw=2.5, alpha=0.8)

    info = fig.text(0.02, 0.98, '', fontsize=10, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def init():
        path_line.set_data([], []); path_line.set_3d_properties([])
        return []

    def update(k):
        nonlocal state
        # advance to the *next* sample so lengths grow together:
        t = (k + 1) * dt
        tgt = target_at(t)

        # World update & true gust
        gust_field.update(dt, state[:3])
        true_gust = gust_field.get_force(state[:3], t)

        # PINN disturbance estimate
        gf_vec = encode_gust_features(state[:3], gust_field, t, max_bursts=2)
        trunk_vec = encode_trunk_vec(state, gust_field, t)
        d_hat = network.forward(gf_vec.reshape(1, -1), trunk_vec.reshape(1, -1))[0]

        # Control (plain FF cancellation)
        u_pid = controller.compute(state[:3], tgt, dt)
        u_total = u_pid - d_hat

        # Dynamics (world uses TRUE gust)
        dstate = quad.compute_dynamics(state, u_total, true_gust)
        state[:] = state + dstate * dt

        # Log (all arrays grow by exactly one)
        trajectory.append(state[:3].copy())
        velocity.append(state[3:6].copy())
        times.append(t)
        gust_forces_log.append(float(np.linalg.norm(true_gust)))
        err_log.append(float(np.linalg.norm(state[:3] - tgt)))

        # Draw
        arr = np.array(trajectory); arrv = np.array(velocity); T = np.array(times)

        path_line.set_data(arr[:,0], arr[:,1]); path_line.set_3d_properties(arr[:,2])
        drone_pt._offsets3d = ([state[0]],[state[1]],[state[2]])
        tgt_pt._offsets3d   = ([tgt[0]],[tgt[1]],[tgt[2]])

        if hasattr(gust_field, "bursts") and len(gust_field.bursts) > 0 and wind_mode == "storm":
            bp = np.array([b['position'] for b in gust_field.bursts], dtype=np.float32)
            burst_scatter._offsets3d = (bp[:,0], bp[:,1], bp[:,2])
        else:
            burst_scatter._offsets3d = ([], [], [])

        line_x.set_data(T, arr[:,0]); line_y.set_data(T, arr[:,1]); line_z.set_data(T, arr[:,2])
        line_vx.set_data(T, arrv[:,0]); line_vy.set_data(T, arrv[:,1]); line_vz.set_data(T, arrv[:,2])
        line_err.set_data(T, np.array(err_log)); line_gust.set_data(T, np.array(gust_forces_log))

        mode_indicator = "⚠️ STORM" if wind_mode == "storm" else "⚠️ STRONG CONST WIND"
        info.set_text(f'[{mode_indicator}] t={t:.1f}s | pos=[{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}] '
                      f'| gust={np.linalg.norm(true_gust):.1f} N | err={err_log[-1]:.2f} m')
        return []

    anim = FuncAnimation(fig, update, init_func=init, frames=n_steps, interval=50, blit=False, repeat=False)
    return anim, fig


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PHYSICS-INFORMED DEEPONET FOR DRONE GUST DISTURBANCE ESTIMATION")
    print("=" * 70)

    # Train model
    trained_network, training_losses = train_deeponet_physics(
        epochs=300, samples_per_epoch=200, batch_size=20, lr0=5e-3, lambda_dyn=0.3
    )

    # Plot training loss
    fig_loss = plt.figure(figsize=(10, 5))
    plt.plot(training_losses, linewidth=2.5, color='steelblue')
    plt.xlabel('Epoch', fontsize=12); plt.ylabel('Loss', fontsize=12)
    plt.title('DeepONet Training Loss (Physics-Informed)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3); plt.yscale('log'); plt.tight_layout(); plt.show()

    # Normal-mode comparison (plain FF)
    print("\n" + "=" * 70)
    print("Running comparison: PINN-Augmented vs Baseline PID (NORMAL MODE)...")
    print("=" * 70)
    times, traj_pinn, traj_baseline, gust_forces, gust_field = run_comparison_simulation(
        trained_network, duration=25.0, dt=0.02, storm_mode=False
    )
    fig_cmp = static_comparison_analysis(times, traj_pinn, traj_baseline, gust_forces)
    plt.show()

    # Storm-mode comparison (plain FF)
    print("\n" + "=" * 70)
    print("STORM MODE TEST - EXTREME ADVERSARIAL CONDITIONS")
    print("=" * 70)
    times_storm, tp_storm, tb_storm, gf_storm, gf_field_storm = run_comparison_simulation(
        trained_network, duration=30.0, dt=0.02, storm_mode=True
    )
    fig_storm = static_comparison_analysis(times_storm, tp_storm, tb_storm, gf_storm)
    plt.suptitle('STORM MODE: PINN-Augmented vs Baseline PID (Extreme Conditions)',
                 fontsize=16, fontweight='bold', y=0.995, color='darkred')
    plt.show()

    # Free-flight (no gusts) with moving target
    print("\n" + "=" * 70)
    print("Free-flight (NO GUSTS) with moving target — animation")
    print("=" * 70)
    anim_free, fig_free = create_animated_free_flight_moving_target_no_gusts(duration=20.0, dt=0.05)
    plt.show()

    # V-stack multi-drone under strong constant horizontal wind
    print("\n" + "=" * 70)
    print("V-Stack simulation under strong +X wind")
    print("=" * 70)
    t_vs, trajs_vs = run_vstack_simulation_constant_wind(trained_network, num_drones=5, spacing=1.0, duration=20.0, dt=0.02)
    fig_vs = plot_vstack_topdown(t_vs, trajs_vs)
    plt.show()

    # Moving target under extreme wind (STORM)
    print("\n" + "=" * 70)
    print("Moving target under EXTREME WIND — animation (STORM MODE)")
    print("=" * 70)
    anim_mt_storm, fig_mt_storm = create_animated_moving_target_extreme_wind(
        trained_network, duration=25.0, dt=0.05, wind_mode="storm", target_path="lissajous"
    )
    plt.show()

    # Moving target under extreme wind (STRONG CONSTANT)
    print("\n" + "=" * 70)
    print("Moving target under EXTREME WIND — animation (STRONG CONSTANT WIND)")
    print("=" * 70)
    anim_mt_const, fig_mt_const = create_animated_moving_target_extreme_wind(
        trained_network, duration=25.0, dt=0.05, wind_mode="constant",
        constant_wind_vec=np.array([22.0, -12.0, 4.0], dtype=np.float32),
        target_path="figure8"
    )
    plt.show()

    # Animated gust sim (plain FF)
    print("\n" + "=" * 70)
    print("Animated gust simulation (Normal Mode, plain FF)")
    print("=" * 70)
    anim, fig_anim = create_animated_simulation(trained_network, duration=25.0, dt=0.05, storm_mode=False)
    plt.show()

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE!")
    print("=" * 70)
