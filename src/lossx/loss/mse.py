"""Mean squared error loss function."""

import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree, Scalar


def mse(
    true_y: PyTree[Array],
    pred_y: PyTree[Array],
    *,
    index: int | None = None,
    **kwargs,
) -> Scalar:
    """Compute mean squared error loss.

    Args:
        true_y: Ground truth values
        pred_y: Predicted values
        index: Optional index to select specific dimension
        **kwargs: Additional arguments

    Returns:
        Scalar MSE loss
    """
    if index is not None:
        true_y = true_y[..., index]
        pred_y = pred_y[..., index]
    return jnp.mean((true_y - pred_y) ** 2)
