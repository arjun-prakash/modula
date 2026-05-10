"""CIFAR MLP builders and presets for muP/manifold benchmark runs."""

import math
from dataclasses import dataclass

from benchmark.scaling import iter_weighted_atoms
from modula import mup as mup_atoms
from modula.bond import Flatten, ReLU

MLP_FEATURE_DIM = 64
PARAMETERIZATION_CHOICES = ("unit_stiefel", "rms_radius")
LINEAR_NORMALIZATION_CHOICES = (
    "unit_stiefel",
    "unit_stiefel_none",
    "unit_stiefel_fan_ratio",
    "rms_radius",
)


@dataclass(frozen=True)
class MupPreset:
    parameterization: str
    manifold_scaling: str
    muon_scaling: str
    project_trunk_after_update: bool


def linear_atom_for_parameterization(parameterization: str):
    if parameterization == "unit_stiefel":
        return mup_atoms.Linear
    if parameterization == "rms_radius":
        return mup_atoms.RMSRadiusLinear
    raise ValueError(f"Unknown muP parameterization: {parameterization}")


def resolve_linear_normalization(linear_normalization: str, args) -> MupPreset:
    if linear_normalization == "unit_stiefel":
        return MupPreset("unit_stiefel", args.manifold_scaling, args.muon_scaling, False)
    if linear_normalization == "unit_stiefel_none":
        return MupPreset("unit_stiefel", "none", "none", False)
    if linear_normalization == "unit_stiefel_fan_ratio":
        return MupPreset("unit_stiefel", "fan_ratio", "fan_ratio", False)
    if linear_normalization == "rms_radius":
        return MupPreset("rms_radius", "fan_ratio", "fan_ratio", True)
    raise ValueError(f"Unknown muP linear normalization: {linear_normalization}")


def build_mlp_trunk(hidden_size: int = MLP_FEATURE_DIM, *, parameterization: str = "unit_stiefel"):
    linear_cls = linear_atom_for_parameterization(parameterization)
    trunk = ReLU() @ linear_cls(hidden_size, hidden_size)
    trunk @= ReLU() @ linear_cls(hidden_size, 32 * 32 * 3)
    trunk @= Flatten()
    trunk.jit()
    return trunk


def build_wide3_mlp_trunk(hidden_size: int = MLP_FEATURE_DIM, *, parameterization: str = "unit_stiefel"):
    linear_cls = linear_atom_for_parameterization(parameterization)
    trunk = ReLU() @ linear_cls(4 * hidden_size, hidden_size)
    trunk @= ReLU() @ linear_cls(hidden_size, 4 * hidden_size)
    trunk @= ReLU() @ linear_cls(4 * hidden_size, 32 * 32 * 3)
    trunk @= Flatten()
    trunk.jit()
    return trunk


def build_mlp_classifier_head(
    num_classes: int,
    feature_dim: int = MLP_FEATURE_DIM,
    *,
    parameterization: str = "unit_stiefel",
):
    linear_cls = linear_atom_for_parameterization(parameterization)
    head = linear_cls(num_classes, feature_dim)
    head.jit()
    return head


def _head_fanin(head) -> int:
    weighted_atoms = iter_weighted_atoms(head)
    if weighted_atoms:
        fanin = getattr(weighted_atoms[0], "fanin", None)
    else:
        fanin = getattr(head, "fanin", None)
    if fanin is None:
        raise ValueError(f"Cannot infer head fan-in for {type(head).__name__}")
    return int(fanin)


def head_adam_update_scale(head, *, base_width: int = 1) -> float:
    return float(base_width) / float(_head_fanin(head))


def head_init_scale(head, *, base_width: int = 1) -> float:
    return math.sqrt(float(base_width) / float(_head_fanin(head)))


def build_cifar_mlp_models(
    num_classes: int,
    hidden_size: int = MLP_FEATURE_DIM,
    trunk: str = "default",
    *,
    parameterization: str = "unit_stiefel",
):
    head_parameterization = "unit_stiefel"
    if trunk == "default":
        return build_mlp_trunk(
            hidden_size,
            parameterization=parameterization,
        ), build_mlp_classifier_head(
            num_classes,
            hidden_size,
            parameterization=head_parameterization,
        )
    if trunk == "wide3":
        return build_wide3_mlp_trunk(
            hidden_size,
            parameterization=parameterization,
        ), build_mlp_classifier_head(
            num_classes,
            4 * hidden_size,
            parameterization=head_parameterization,
        )
    raise ValueError(f"Unknown CIFAR MLP trunk: {trunk}")
