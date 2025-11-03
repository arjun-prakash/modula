import math

import torch

from msign import msign


@torch.no_grad()
def manifold_muon(W, G, eta=0.1, alpha=0.01, steps=100, tol=1e-6):
    """
    Note that this actually implements GD on || G + W @ (L + L.mT) ||_*,
    whereas the blog discusses the parameterization with an extra factor of 2 on L
    It exploits the property that if L is initialized symmetric, it stays symmetric
    """
    # Ensure that W and G are both tall matrices
    should_tranpose = W.shape[0] < W.shape[1]
    if should_tranpose:
        W = W.T
        G = G.T
    # Initialize the dual variable
    Lambda = -0.25 * (W.T @ G + G.T @ W)
    # Ascend on the dual problem to find the update direction A
    for step in range(steps):
        # Update the candidate direction A
        A = msign(G + 2 * W @ Lambda)
        # Measure deviation of A from the tangent space:
        H = W.T @ A + A.T @ W
        # Check the stopping criterion
        if torch.norm(H) / math.sqrt(H.numel()) < tol:
            break
        # Update the dual variable
        Lambda -= alpha * (1 - step / steps) * H
    # Descend on the primal problem
    new_W = W - eta * A
    # Retract to the manifold
    new_W = msign(new_W)
    # Restore the shape of the solution and return
    return new_W.T if should_tranpose else new_W


@torch.no_grad()
def manifold_muon_admm(W, G, eta=0.1, steps=10, rho=4.0):
    """Implements GD on || G + W @ (L + L.mT) ||_* (c.f. the blog)"""
    # Ensure that W and G are both tall matrices
    should_tranpose = W.shape[0] < W.shape[1]
    if should_tranpose:
        W = W.T
        G = G.T
    # Initialize the lagrangian, slack, and dual variable
    Lambda = -0.25 * (W.T @ G + G.T @ W)
    X = G + 2 * W @ Lambda
    Omega = torch.zeros_like(X)
    # Solve the dual problem with ADMM to find the update direction A
    for step in range(steps):
        # Update for Lambda (orthonormal least-squares solve)
        P = W.mT @ (1 / rho * Omega + X - G)
        Lambda_upd = 0.25 * (P + P.mT)
        # Update for X (singular value thresholding)
        B = G + 2 * W @ Lambda_upd - 1 / rho * Omega
        eye = torch.eye(B.shape[1], device=B.device, dtype=B.dtype)
        P_pos = 0.5 * (eye + msign(B.mT @ B - 1 / rho**2 * eye))
        X_upd = (B - 1 / rho * msign(B)) @ P_pos
        # Update for Omega (dual ascent)
        Omega_upd = Omega + rho * (X_upd - 2 * W @ Lambda_upd - G)
        Lambda, X, Omega = Lambda_upd, X_upd, Omega_upd
    # Calculate A from final ADMM solution
    # (at convergence, G + 2 * W @ Lambda \approx X)
    A = msign(G + 2 * W @ Lambda)
    # Descend on the primal problem
    new_W = W - eta * A
    # Retract to the manifold
    new_W = msign(new_W)
    # Restore the shape of the solution and return
    return new_W.T if should_tranpose else new_W