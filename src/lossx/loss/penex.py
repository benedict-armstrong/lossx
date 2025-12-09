"""Penalized exponential (Penex) loss function."""

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree, Scalar


def _exp_loss_component(
    y: Array,
    logits: Array,
    sensitivity: float = 1.0,
    transform_y_flag: bool = False,
) -> Array:
    """Helper function to compute exponential loss component."""
    if not transform_y_flag:
        # detransform y
        y = y.argmax(axis=1)

    # if dimension of y is 1, expand it to 2 dimensions
    if y.ndim == 1:
        y = jnp.expand_dims(y, axis=1)

    loss = jnp.exp(-sensitivity * jnp.take_along_axis(logits, y, axis=1))[:, 0]
    return loss


def penex(
    true_y: PyTree[Array],
    pred_y: PyTree[Array],
    *,
    ignore_index: int | None = None,
    sensitivity: float = 1.0,
    huberization: float | None = None,
    penalty: float = 0.1,
    reduction: str = "mean",
    **kwargs,
) -> Scalar:
    """Penalized Exponential (Penex) loss from https://arxiv.org/pdf/2510.02107.

    Combines exponential loss with a penalty term based on the sum of exponentials.

    Args:
        true_y: Ground truth labels
        pred_y: Predicted logits
        ignore_index: Index to ignore in loss computation
        sensitivity: Sensitivity parameter for exponential loss
        huberization: Huberization parameter (not implemented)
        penalty: Penalty weight for sumexp term
        reduction: Reduction method ("mean", "sum", or "none")
        **kwargs: Additional arguments

    Returns:
        Scalar loss value
    """
    if huberization is not None:
        raise NotImplementedError("Huberization is not implemented yet.")

    input_data = pred_y
    target = true_y

    # filter out the ignore_index (if any)
    if ignore_index is not None:
        mask = target != ignore_index
        input_data = input_data[mask]
        target = target[mask]

    sumexp = jnp.exp(jax.scipy.special.logsumexp(input_data, axis=-1))
    exp_loss_values = _exp_loss_component(
        target,
        input_data,
        sensitivity,
        transform_y_flag=True,
    )

    # Compute final loss: exp_loss + penalty * sumexp
    total_loss = exp_loss_values + penalty * sumexp

    if reduction == "mean":
        return jnp.mean(total_loss)
    elif reduction == "sum":
        return jnp.sum(total_loss)
    elif reduction == "none":
        return total_loss
    else:
        raise ValueError(
            f"Invalid reduction mode: {reduction}. Choose from 'mean', 'sum', or 'none'."
        )
