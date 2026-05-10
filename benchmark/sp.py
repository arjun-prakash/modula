"""CIFAR MLP builders for standard-parameterization benchmark runs."""

from modula import sp as sp_atoms
from modula.atom import Linear as ManifoldLinear
from modula.bond import Flatten, ReLU

MLP_FEATURE_DIM = 64
PARAMETERIZATION_CHOICES = ("sp", "unit_stiefel")


def linear_atom_for_parameterization(parameterization: str):
    if parameterization == "sp":
        return sp_atoms.Linear
    if parameterization == "unit_stiefel":
        return ManifoldLinear
    raise ValueError(f"Unknown SP benchmark parameterization: {parameterization}")


def build_mlp_trunk(hidden_size: int = MLP_FEATURE_DIM, *, parameterization: str = "sp"):
    linear_cls = linear_atom_for_parameterization(parameterization)
    trunk = ReLU() @ linear_cls(hidden_size, hidden_size)
    trunk @= ReLU() @ linear_cls(hidden_size, 32 * 32 * 3)
    trunk @= Flatten()
    trunk.jit()
    return trunk


def build_wide3_mlp_trunk(hidden_size: int = MLP_FEATURE_DIM, *, parameterization: str = "sp"):
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
    parameterization: str = "sp",
):
    linear_cls = linear_atom_for_parameterization(parameterization)
    head = linear_cls(num_classes, feature_dim)
    head.jit()
    return head


def build_cifar_mlp_models(
    num_classes: int,
    hidden_size: int = MLP_FEATURE_DIM,
    trunk: str = "default",
    *,
    parameterization: str = "sp",
):
    if trunk == "default":
        return build_mlp_trunk(
            hidden_size,
            parameterization=parameterization,
        ), build_mlp_classifier_head(
            num_classes,
            hidden_size,
            parameterization=parameterization,
        )
    if trunk == "wide3":
        return build_wide3_mlp_trunk(
            hidden_size,
            parameterization=parameterization,
        ), build_mlp_classifier_head(
            num_classes,
            4 * hidden_size,
            parameterization=parameterization,
        )
    raise ValueError(f"Unknown CIFAR MLP trunk: {trunk}")
