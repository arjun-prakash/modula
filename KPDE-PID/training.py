#!/usr/bin/env python3
"""
training.py - Physics-informed training for DeepONet
"""

import numpy as np
from physics_models import GustField, QuadcopterDynamics
from deeponet import DeepONet
from controllers import encode_gust_features, encode_trunk_vec


def train_deeponet_physics(epochs=300, samples_per_epoch=200, batch_size=20,
                           lr0=5e-3, weight_decay=1e-4, lambda_dyn=0.3,
                           save_path=None):
    """
    Train DeepONet with physics-informed loss
    
    Loss = MSE(d_hat, d_true) + λ * ||m*a_true - (u - m*g - c_d*v + d_hat)||²
    
    Args:
        epochs: number of training epochs
        samples_per_epoch: samples generated per epoch
        batch_size: mini-batch size
        lr0: initial learning rate
        weight_decay: L2 regularization strength
        lambda_dyn: physics loss weight
        save_path: path to save trained model (optional)
        
    Returns:
        network: trained DeepONet
        losses: list of training losses per epoch
    """
    print("="*70)
    print("PHYSICS-INFORMED DEEPONET TRAINING")
    print("="*70)
    print(f"Epochs: {epochs}")
    print(f"Samples/epoch: {samples_per_epoch}")
    print(f"Batch size: {batch_size}")
    print(f"Initial LR: {lr0}")
    print(f"Physics weight (λ): {lambda_dyn}")
    print("="*70)
    
    network = DeepONet(branch_input_dim=10, trunk_input_dim=10, 
                       hidden_dim=64, output_dim=3)
    quad = QuadcopterDynamics(mass=1.0, drag_coeff=0.1)
    g = quad.g

    losses = []
    best_loss = float('inf')
    patience = 50
    bad_epochs = 0

    for epoch in range(epochs):
        epoch_loss = 0.0

        # Collect samples
        GF_list, TRUNK_list, TRUE_list = [], [], []
        ATRUE_list, U_list, V_list = [], [], []

        for _ in range(samples_per_epoch):
            # Random scenario
            drone_pos = np.random.randn(3).astype(np.float32) * 4.0
            drone_vel = (np.random.randn(3) * 1.5).astype(np.float32)
            state = np.concatenate([drone_pos, drone_vel]).astype(np.float32)

            # Random gust field
            gust_field = GustField(max_bursts=4, 
                                   storm_mode=(np.random.rand() < 0.5))
            n_bursts = np.random.randint(0, 4)
            for _ in range(n_bursts):
                gust_field._spawn_burst(drone_pos)

            t = float(np.random.rand() * 10.0)
            d_true = gust_field.get_force(drone_pos, t).astype(np.float32)

            # Random control input
            u = (np.random.randn(3) * 6.0).astype(np.float32)
            
            # Compute true acceleration from physics
            gravity = np.array([0, 0, -quad.mass * g], dtype=np.float32)
            drag = -quad.drag * drone_vel
            a_true = (u + d_true + drag + gravity) / quad.mass

            # Encode features
            gf = encode_gust_features(drone_pos, gust_field, t, max_bursts=2)
            trunk = encode_trunk_vec(state, gust_field, t)

            GF_list.append(gf)
            TRUNK_list.append(trunk)
            TRUE_list.append(d_true)
            ATRUE_list.append(a_true)
            U_list.append(u)
            V_list.append(drone_vel)

        # Convert to arrays
        GF = np.array(GF_list)
        TR = np.array(TRUNK_list)
        TRUE = np.array(TRUE_list)
        ATRUE = np.array(ATRUE_list)
        UU = np.array(U_list)
        VV = np.array(V_list)

        # Mini-batch training
        num_batches = samples_per_epoch // batch_size
        for b in range(num_batches):
            s = b * batch_size
            e = s + batch_size
            
            gf = GF[s:e]
            tr = TR[s:e]
            d_true = TRUE[s:e]
            a_true = ATRUE[s:e]
            u = UU[s:e]
            v = VV[s:e]

            # Forward pass
            pred, cache = network.forward(gf, tr, return_cache=True)

            # Physics residual: c = m*a_true - (u - m*g*e_z - c_d*v)
            m = quad.mass
            cd = quad.drag
            gravity_vec = np.array([0, 0, m * g], dtype=np.float32)
            c = (m * a_true) - (u - gravity_vec - cd * v)

            # Combined loss
            err = pred - d_true
            mse = np.mean(err**2)
            resid = pred - c
            l_dyn = np.mean(resid**2)
            total = mse + lambda_dyn * l_dyn
            
            epoch_loss += total

            # Gradient
            grad_pred = (2.0 / (batch_size * 3)) * (err + lambda_dyn * (pred - c))
            grad_pred = np.clip(grad_pred, -1.0, 1.0)

            # Learning rate decay
            lr = lr0 * (0.98 ** (epoch // 10))
            
            # Backprop
            network.backward(cache, grad_pred, 
                           learning_rate=lr, weight_decay=weight_decay)

        # Epoch statistics
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            bad_epochs = 0
            if save_path:
                network.save_weights(save_path)
        else:
            bad_epochs += 1

        # Early stopping
        if bad_epochs >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

        # Progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{epochs} | Loss: {avg_loss:.6f} | "
                  f"Best: {best_loss:.6f} | LR: {lr:.6f}")

    print("="*70)
    print(f"Training complete!")
    print(f"Final loss: {avg_loss:.6f}")
    print(f"Best loss: {best_loss:.6f}")
    print("="*70)

    return network, losses


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Train network
    network, losses = train_deeponet_physics(
        epochs=300,
        samples_per_epoch=200,
        batch_size=20,
        lr0=5e-3,
        lambda_dyn=0.3,
        save_path="deeponet_weights.npz"
    )
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses, linewidth=2.5, color='steelblue')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Physics-Informed DeepONet Training', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150)
    plt.show()