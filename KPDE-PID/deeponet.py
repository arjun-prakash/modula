#!/usr/bin/env python3
"""
deeponet.py - Deep Operator Network implementation
"""

import numpy as np


class DeepONet:
    """
    Deep Operator Network for disturbance prediction
    
    Architecture:
        Branch: encodes gust features (10-dim)
        Trunk: encodes state + time + base wind (10-dim)
        Output: predicted disturbance force (3-dim)
    """
    
    def __init__(self, branch_input_dim=10, trunk_input_dim=10, 
                 hidden_dim=64, output_dim=3, seed=0):
        rng = np.random.default_rng(seed)

        def xavier_init(din, dout):
            return (rng.standard_normal((din, dout)).astype(np.float32) 
                    * np.sqrt(2.0 / din))

        # Branch network
        self.W_branch1 = xavier_init(branch_input_dim, hidden_dim)
        self.b_branch1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W_branch2 = xavier_init(hidden_dim, hidden_dim)
        self.b_branch2 = np.zeros(hidden_dim, dtype=np.float32)

        # Trunk network
        self.W_trunk1 = xavier_init(trunk_input_dim, hidden_dim)
        self.b_trunk1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W_trunk2 = xavier_init(hidden_dim, hidden_dim)
        self.b_trunk2 = np.zeros(hidden_dim, dtype=np.float32)

        # Output layer
        self.W_out = xavier_init(hidden_dim, output_dim)
        self.b_out = np.zeros(output_dim, dtype=np.float32)

    @staticmethod
    def _relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_deriv_from_pre(z):
        return (z > 0.0).astype(np.float32)

    def forward(self, gust_features, trunk_vec, return_cache=False):
        """
        Forward pass
        
        Args:
            gust_features: (B, 10) or (10,)
            trunk_vec: (B, 10) or (10,)
            return_cache: bool, whether to cache for backprop
            
        Returns:
            output: (B, 3) predicted disturbance force
            cache: dict (if return_cache=True)
        """
        gf = gust_features
        tv = trunk_vec
        if gf.ndim == 1:
            gf = gf.reshape(1, -1)
        if tv.ndim == 1:
            tv = tv.reshape(1, -1)

        # Branch forward
        z_b1 = gf @ self.W_branch1 + self.b_branch1
        b1 = self._relu(z_b1)
        z_b2 = b1 @ self.W_branch2 + self.b_branch2
        b2 = self._relu(z_b2)

        # Trunk forward
        z_t1 = tv @ self.W_trunk1 + self.b_trunk1
        t1 = self._relu(z_t1)
        z_t2 = t1 @ self.W_trunk2 + self.b_trunk2
        t2 = self._relu(z_t2)

        # Combine and output
        combined = b2 * t2
        out = combined @ self.W_out + self.b_out

        if not return_cache:
            return out

        cache = dict(
            gust_features=gf, trunk_vec=tv,
            z_b1=z_b1, b1=b1, z_b2=z_b2, b2=b2,
            z_t1=z_t1, t1=t1, z_t2=z_t2, t2=t2,
            combined=combined
        )
        return out, cache

    def backward(self, cache, grad_out, learning_rate=0.001, weight_decay=0.0):
        """
        Backpropagation for mini-batch
        
        Args:
            cache: dict from forward pass
            grad_out: (B, 3) gradient of loss w.r.t. output
            learning_rate: float
            weight_decay: float, L2 regularization
        """
        # Output layer gradients
        grad_W_out = cache['combined'].T @ grad_out
        grad_b_out = np.sum(grad_out, axis=0)
        grad_comb = grad_out @ self.W_out.T

        # Split to branch & trunk
        grad_b2 = grad_comb * cache['t2']
        grad_t2 = grad_comb * cache['b2']

        # Branch backprop
        grad_b2_pre = grad_b2 * self._relu_deriv_from_pre(cache['z_b2'])
        grad_W_b2 = cache['b1'].T @ grad_b2_pre
        grad_b_b2 = np.sum(grad_b2_pre, axis=0)

        grad_b1 = grad_b2_pre @ self.W_branch2.T
        grad_b1_pre = grad_b1 * self._relu_deriv_from_pre(cache['z_b1'])
        grad_W_b1 = cache['gust_features'].T @ grad_b1_pre
        grad_b_b1 = np.sum(grad_b1_pre, axis=0)

        # Trunk backprop
        grad_t2_pre = grad_t2 * self._relu_deriv_from_pre(cache['z_t2'])
        grad_W_t2 = cache['t1'].T @ grad_t2_pre
        grad_b_t2 = np.sum(grad_t2_pre, axis=0)

        grad_t1 = grad_t2_pre @ self.W_trunk2.T
        grad_t1_pre = grad_t1 * self._relu_deriv_from_pre(cache['z_t1'])
        grad_W_t1 = cache['trunk_vec'].T @ grad_t1_pre
        grad_b_t1 = np.sum(grad_t1_pre, axis=0)

        # Apply weight decay
        def decay(W):
            return weight_decay * W if weight_decay > 0 else 0.0

        # Update weights
        self.W_out -= learning_rate * (grad_W_out + decay(self.W_out))
        self.b_out -= learning_rate * grad_b_out

        self.W_branch2 -= learning_rate * (grad_W_b2 + decay(self.W_branch2))
        self.b_branch2 -= learning_rate * grad_b_b2
        self.W_branch1 -= learning_rate * (grad_W_b1 + decay(self.W_branch1))
        self.b_branch1 -= learning_rate * grad_b_b1

        self.W_trunk2 -= learning_rate * (grad_W_t2 + decay(self.W_trunk2))
        self.b_trunk2 -= learning_rate * grad_b_t2
        self.W_trunk1 -= learning_rate * (grad_W_t1 + decay(self.W_trunk1))
        self.b_trunk1 -= learning_rate * grad_b_t1

    def save_weights(self, filepath):
        """Save network weights to file"""
        np.savez(
            filepath,
            W_branch1=self.W_branch1, b_branch1=self.b_branch1,
            W_branch2=self.W_branch2, b_branch2=self.b_branch2,
            W_trunk1=self.W_trunk1, b_trunk1=self.b_trunk1,
            W_trunk2=self.W_trunk2, b_trunk2=self.b_trunk2,
            W_out=self.W_out, b_out=self.b_out
        )

    def load_weights(self, filepath):
        """Load network weights from file"""
        data = np.load(filepath)
        self.W_branch1 = data['W_branch1']
        self.b_branch1 = data['b_branch1']
        self.W_branch2 = data['W_branch2']
        self.b_branch2 = data['b_branch2']
        self.W_trunk1 = data['W_trunk1']
        self.b_trunk1 = data['b_trunk1']
        self.W_trunk2 = data['W_trunk2']
        self.b_trunk2 = data['b_trunk2']
        self.W_out = data['W_out']
        self.b_out = data['b_out']