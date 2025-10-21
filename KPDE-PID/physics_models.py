#!/usr/bin/env python3
"""
physics_models.py - Core physics simulation components
"""

import numpy as np


class GustField:
    """Pulsed-jet burst gust field model with optional storm mode"""
    
    def __init__(self, max_bursts=5, spawn_radius=10.0, storm_mode=False):
        self.max_bursts = max_bursts
        self.spawn_radius = spawn_radius
        self.bursts = []
        self.storm_mode = storm_mode

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
        """Update burst lifetimes and spawn new bursts"""
        self.bursts = [b for b in self.bursts if b['lifetime'] > 0]
        for burst in self.bursts:
            burst['lifetime'] -= dt

        spawn_prob = 0.15 if self.storm_mode else 0.1
        if len(self.bursts) < self.max_bursts and np.random.rand() < spawn_prob:
            self._spawn_burst(drone_pos)

    def _spawn_burst(self, drone_pos):
        """Create a new burst directed at drone"""
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
        """Compute total gust force (base wind + active bursts)"""
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


class ConstantWindField:
    """Simple constant horizontal wind field"""
    
    def __init__(self, wind_vec=np.array([15.0, 0.0, 0.0], dtype=np.float32)):
        self.wind_vec = wind_vec.astype(np.float32)
        self.bursts = []
        self.storm_mode = True
        self.base_wind = self.wind_vec
        self.wind_turbulence_freq = 0.0
        self.wind_turbulence_amp = 0.0

    def update(self, dt, pos):
        pass

    def get_force(self, pos, t):
        return self.wind_vec


class QuadcopterDynamics:
    """6-DOF quadcopter dynamics model"""
    
    def __init__(self, mass=1.0, drag_coeff=0.1):
        self.mass = float(mass)
        self.drag = float(drag_coeff)
        self.g = 9.81

    def compute_dynamics(self, state, control_force, disturbance_force):
        """
        Compute state derivative
        State: [x, y, z, vx, vy, vz]
        Returns: [vx, vy, vz, ax, ay, az]
        """
        vel = state[3:6]
        gravity = np.array([0, 0, -self.mass * self.g], dtype=np.float32)
        drag_force = -self.drag * vel
        total_force = control_force + disturbance_force + gravity + drag_force
        accel = total_force / self.mass
        return np.concatenate([vel, accel]).astype(np.float32)