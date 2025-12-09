"""LossX: A JAX loss function library with config-based construction."""

from . import loss
from .builder import build_loss, register_loss
from .loss import (
    contrastive,
    cross_entropy,
    gaussian_nll,
    mse,
    penex,
    q_loss,
    quantile_loss,
)

__all__ = [
    "build_loss",
    "register_loss",
    "loss",
    # Individual loss functions
    "contrastive",
    "cross_entropy",
    "gaussian_nll",
    "mse",
    "penex",
    "q_loss",
    "quantile_loss",
]
