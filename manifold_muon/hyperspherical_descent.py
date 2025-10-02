import torch

@torch.no_grad()
def hyperspherical_descent(W, G, eta=0.1):
    w = W.flatten()
    g = G.flatten()
    # Compute update direction
    a = g - w * torch.dot(w, g)
    a /= (a.norm() + 1e-12)
    # Apply update
    new_w = w - eta * a
    # Retract to the manifold
    new_w /= new_w.norm()
    # Restore the shape of the solution and return
    return new_w.reshape(W.shape)