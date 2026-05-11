"""Fan-ratio scaling helpers shared by benchmark entrypoints."""

from typing import Any, List, Sequence

import jax.numpy as jnp
import numpy as np

from modula.atom import Linear


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


def manifold_update_scale(atom) -> float:
    if isinstance(atom, Linear):
        return float(np.sqrt(atom.fanout / atom.fanin))
    raise ValueError(f"Unsupported benchmark atom type: {type(atom).__name__}")


def manifold_directions(module, tangents, *, atoms: Sequence[Any] | None = None):
    weighted_atoms = list(atoms) if atoms is not None else iter_weighted_atoms(module)

    if len(weighted_atoms) != len(tangents):
        raise ValueError(f"Mismatch between atom metadata ({len(weighted_atoms)}) and tangents ({len(tangents)})")

    directions = []
    for atom, tangent in zip(weighted_atoms, tangents):
        scale = manifold_update_scale(atom)
        directions.append(jnp.asarray(scale, dtype=tangent.dtype) * tangent)
    return directions
