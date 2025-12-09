"""Gaussian negative log-likelihood loss function."""

import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree, Scalar


def gaussian_nll(
    true_y: PyTree[Array],
    pred_y: PyTree[Array],
    *,
    eps: float = 1e-6,
    **kwargs,
) -> Scalar:
    """Compute the negative log-likelihood of a Gaussian distribution.

    Expects pred_y to have mean and variance stacked along the last dimension,
    where the first half is mean and the second half is variance.

    Args:
        true_y: Ground truth values
        pred_y: Predicted mean and variance (stacked on last axis)
        eps: Small value to prevent numerical issues (default: 1e-6)
        **kwargs: Additional arguments

    Returns:
        Negative log-likelihood of the Gaussian distribution as a scalar
    """
    # pred_y last dimension first half is mean, second half is var
    pred_y_mean, pred_y_var = jnp.split(pred_y, 2, axis=-1)

    squared_diff = (true_y - pred_y_mean) ** 2
    var_term = jnp.maximum(pred_y_var, eps)

    return 0.5 * jnp.mean(jnp.log(var_term) + squared_diff / var_term)
