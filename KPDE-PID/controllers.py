#!/usr/bin/env python3
"""
controllers.py - Control algorithms
"""

import numpy as np


class PIDController:
    """Classic PID controller with anti-windup"""
    
    def __init__(self, kp=20.0, ki=0.5, kd=10.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral = np.zeros(3, dtype=np.float32)
        self.prev_error = np.zeros(3, dtype=np.float32)

    def compute(self, current_pos, target_pos, dt):
        """Compute PID control force"""
        error = (target_pos - current_pos).astype(np.float32)
        
        # Integral with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -5, 5)
        
        # Derivative
        derivative = (error - self.prev_error) / dt if dt > 0 else np.zeros(3, dtype=np.float32)
        self.prev_error = error.copy()
        
        # PID output
        force = self.kp * error + self.ki * self.integral + self.kd * derivative
        return np.clip(force, -50, 50).astype(np.float32)

    def reset(self):
        """Reset controller state"""
        self.integral[:] = 0.0
        self.prev_error[:] = 0.0


class PINNAugmentedController:
    """PID controller augmented with DeepONet disturbance prediction"""
    
    def __init__(self, network, pid_controller, use_ema=False, alpha_ema=0.7):
        self.network = network
        self.pid = pid_controller
        self.use_ema = use_ema
        self.alpha = alpha_ema
        self.d_hat_ema = np.zeros(3, dtype=np.float32)

    def compute(self, current_pos, current_vel, target_pos, gust_field, t, dt):
        """
        Compute control with disturbance cancellation
        
        Args:
            current_pos: 3D position
            current_vel: 3D velocity
            target_pos: 3D target position
            gust_field: GustField object
            t: current time
            dt: timestep
            
        Returns:
            control_force: 3D control force
        """
        # PID baseline
        u_pid = self.pid.compute(current_pos, target_pos, dt)
        
        # Encode features
        state = np.concatenate([current_pos, current_vel]).astype(np.float32)
        gust_features = encode_gust_features(current_pos, gust_field, t, max_bursts=2)
        trunk_vec = encode_trunk_vec(state, gust_field, t)
        
        # Predict disturbance
        d_hat = self.network.forward(
            gust_features.reshape(1, -1), 
            trunk_vec.reshape(1, -1)
        )[0]
        
        # Optional EMA smoothing
        if self.use_ema:
            self.d_hat_ema = self.alpha * self.d_hat_ema + (1 - self.alpha) * d_hat
            d_hat = self.d_hat_ema
        
        # Feedforward cancellation
        u_total = u_pid - d_hat
        return u_total


def encode_gust_features(drone_pos, gust_field, t, max_bursts=2):
    """
    Encode gust field into 10-dim feature vector
    Format: [rel_pos/10, dist/10, intensity/70] × 2 bursts
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
    Encode drone state into 10-dim trunk vector
    Format: [x, y, z, vx, vy, vz, t_norm, base_wind/10 (3)]
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