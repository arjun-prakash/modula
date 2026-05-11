"""CIFAR MLP builders for standard-parameterization runs."""

from modula import sp as sp_atoms
from modula.bond import Flatten, ReLU

MLP_FEATURE_DIM = 64


def build_mlp_trunk(hidden_size: int = MLP_FEATURE_DIM):
    trunk = ReLU() @ sp_atoms.Linear(hidden_size, hidden_size)
    trunk @= ReLU() @ sp_atoms.Linear(hidden_size, 32 * 32 * 3)
    trunk @= Flatten()
    trunk.jit()
    return trunk


def build_mlp_classifier_head(num_classes: int, feature_dim: int = MLP_FEATURE_DIM):
    head = sp_atoms.Linear(num_classes, feature_dim)
    head.jit()
    return head


def build_cifar_mlp_models(num_classes: int, hidden_size: int = MLP_FEATURE_DIM):
    return build_mlp_trunk(hidden_size), build_mlp_classifier_head(num_classes, hidden_size)
