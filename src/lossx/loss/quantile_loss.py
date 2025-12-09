"""Quantile loss function."""

import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree, Scalar


def quantile_loss(
    true_y: PyTree[Array],
    pred_y: PyTree[Array],
    *,
    quantiles: tuple[float, ...] = (0.1587, 0.8413),
    **kwargs,
) -> Scalar:
    """Quantile loss for uncertainty estimation.

    Expects pred_y to have 3 components along the last axis:
    - predicted mean
    - lower quantile
    - upper quantile

    Args:
        true_y: Ground truth values
        pred_y: Predicted values (3 components in last dimension)
        quantiles: Quantile values (default: ~1 std for Gaussian)
        **kwargs: Additional arguments

    Returns:
        Scalar loss value combining MSE and quantile losses
    """
    p_means, *qs = jnp.split(pred_y, 3, axis=-1)

    mse_loss = jnp.mean((true_y - p_means) ** 2)

    for i, quantile in enumerate(quantiles):
        q_pred = qs[i]
        z = true_y - q_pred
        q_loss = jnp.where(z > 0, z * quantile, (quantile - 1) * z)
        mse_loss += jnp.mean(q_loss)

    return mse_loss
