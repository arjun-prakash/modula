"""Width/update scaling helpers shared by benchmark entrypoints."""

from typing import Any, List

import jax.numpy as jnp
import numpy as np

from modula.atom import Linear
from modula.manifold import matrix_sign

SCALING_CHOICES = ("fan_ratio", "fan_max", "none")


def iter_weighted_atoms(module) -> List[Any]:
    atoms = int(getattr(module, "atoms", 0) or 0)
    if atoms == 0:
        return []

    children = getattr(module, "children", ())
    if not children:
        return [module]

    ordered_atoms: List[Any] = []
    for child in children:
        ordered_atoms.extend(iter_weighted_atoms(child))
    return ordered_atoms


def manifold_update_scale(atom, *, scaling: str = "fan_ratio") -> float:
    if scaling == "none":
        return 1.0
    if scaling == "fan_ratio":
        if isinstance(atom, Linear):
            return float(np.sqrt(atom.fanout / atom.fanin))
        raise ValueError(f"Unsupported manifold benchmark atom type: {type(atom).__name__}")
    if scaling != "fan_max":
        raise ValueError(f"Unknown manifold scaling: {scaling}")

    if isinstance(atom, Linear):
        return float(np.sqrt(max(atom.fanin, atom.fanout)))
    raise ValueError(f"Unsupported manifold benchmark atom type: {type(atom).__name__}")


def muon_update_scale(atom, *, scaling: str) -> float:
    if scaling == "none":
        return 1.0
    if scaling == "fan_ratio":
        return manifold_update_scale(atom, scaling=scaling)
    if scaling != "fan_max":
        raise ValueError(f"Unknown Muon scaling: {scaling}")

    if isinstance(atom, Linear):
        return float(np.sqrt(max(atom.fanin, atom.fanout)))
    raise ValueError(f"Unsupported Muon benchmark atom type: {type(atom).__name__}")


def manifold_directions(module, tangents, *, scaling: str):
    atoms = iter_weighted_atoms(module)

    if len(atoms) != len(tangents):
        raise ValueError(f"Mismatch between atom metadata ({len(atoms)}) and tangents ({len(tangents)})")

    directions = []
    for atom, tangent in zip(atoms, tangents):
        scale = manifold_update_scale(atom, scaling=scaling)
        directions.append(jnp.asarray(scale, dtype=tangent.dtype) * tangent)
    return directions


def muon_direction(atom, grad):
    if isinstance(atom, Linear):
        return matrix_sign(grad)
    raise ValueError(f"Unsupported Muon benchmark atom type: {type(atom).__name__}")


def muon_directions(module, grads, *, scaling: str):
    atoms = iter_weighted_atoms(module)

    if len(atoms) != len(grads):
        raise ValueError(f"Mismatch between atom metadata ({len(atoms)}) and gradients ({len(grads)})")

    directions = []
    for atom, grad in zip(atoms, grads):
        direction = muon_direction(atom, grad)
        scale = muon_update_scale(atom, scaling=scaling)
        directions.append(jnp.asarray(scale, dtype=direction.dtype) * direction)
    return directions
