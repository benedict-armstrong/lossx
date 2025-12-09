"""Loss function builder for creating losses from configuration dictionaries."""

from collections.abc import Callable
from functools import partial
from typing import NotRequired, TypedDict

import jax.numpy as jnp
from jax import Array
from jax.tree_util import tree_leaves, tree_map
from jaxtyping import PyTree, Scalar

from .loss import (
    contrastive,
    cross_entropy,
    gaussian_nll,
    mse,
    penex,
    q_loss,
    quantile_loss,
)
from .types import LossFn, Reduction

# Registry of available loss functions
LOSS_REGISTRY: dict[str, Callable] = {
    "contrastive": contrastive,
    "cross_entropy": cross_entropy,
    "gaussian_nll": gaussian_nll,
    "mse": mse,
    "penex": penex,
    "q_loss": q_loss,
    "quantile_loss": quantile_loss,
}


class LossArgsCfg(TypedDict):
    """Configuration for a single loss function."""

    target: str  # loss function name
    weight: NotRequired[float]
    # ... any loss-specific kwargs


def build_loss(  # noqa: C901
    config: PyTree[LossArgsCfg],
    reduction: Reduction[Scalar] = jnp.sum,
) -> LossFn[PyTree[Array]]:
    """Build loss from PyTree of configs.

    Handles all cases uniformly:
    - Single dict → single loss
    - List of dicts → composite loss
    - Dict of dicts → multi-task loss
    - Any nested structure → arbitrary composition
    """

    def _build_single_loss(cfg: LossArgsCfg) -> LossFn[PyTree[Array]]:
        loss_fn = LOSS_REGISTRY[cfg["target"]]
        weight = cfg.get("weight", 1.0)
        # Filter out 'target' and 'weight' from config before passing to loss
        loss_kwargs = {k: v for k, v in cfg.items() if k not in ["target", "weight"]}

        if loss_kwargs:
            # Partially apply the loss-specific kwargs
            weighted_loss = partial(loss_fn, **loss_kwargs)
        else:
            weighted_loss = loss_fn

        # Return a function that applies the weight
        def loss_with_weight(true: PyTree[Array], pred: PyTree[Array]) -> Scalar:
            return weight * weighted_loss(true, pred)

        return loss_with_weight

    def _is_loss_config(cfg):
        """Check if a config dict represents a single loss (has 'target' key)."""
        return isinstance(cfg, dict) and "target" in cfg

    def _build_tree(cfg):  # noqa: C901
        """Recursively build a tree of loss functions from config tree."""
        if _is_loss_config(cfg):
            # This is a loss config - build the loss function
            return _build_single_loss(cfg)
        elif isinstance(cfg, dict):
            # This is a dict of configs - recursively build each value
            return {k: _build_tree(v) for k, v in cfg.items()}
        elif isinstance(cfg, (list, tuple)):
            # This is a list/tuple of configs - recursively build each element
            return type(cfg)(_build_tree(item) for item in cfg)
        else:
            raise ValueError(f"Unexpected config type: {type(cfg)}")

    # Build the loss function tree
    loss_tree = _build_tree(config)

    # Check if this is a single loss (no PyTree structure)
    if _is_loss_config(config):
        # Single loss case - return directly (no reduction needed)
        return loss_tree

    # Multi-loss case - create a function that applies all losses and reduces
    def loss(true: PyTree[Array], pred: PyTree[Array]) -> Scalar:
        # Apply losses element-wise across the tree
        losses = tree_map(lambda fn, t, p: fn(t, p), loss_tree, true, pred)
        # Flatten and reduce
        return reduction(tree_leaves(losses))

    return loss


def register_loss(name: str, loss_fn: Callable[[PyTree[Array], PyTree[Array]], Scalar]) -> None:
    """Register a custom loss function to make it available in build_loss.

    Args:
        name: Name to register the loss function under
        loss_fn: The loss function to register. Should take two PyTrees and return a scalar.

    Example:
        >>> def my_custom_loss(true_y, pred_y, **kwargs):
        ...     return jnp.mean((true_y - pred_y) ** 2)
        >>> register_loss("custom_mse", my_custom_loss)
    """
    if name in LOSS_REGISTRY:
        raise ValueError(f"Loss '{name}' is already registered")
    LOSS_REGISTRY[name] = loss_fn
