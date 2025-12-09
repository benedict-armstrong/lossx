"""Q-loss for classification with masking support."""

import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree, Scalar


def q_loss(
    true_y: PyTree[Array],
    pred_y: PyTree[Array],
    *,
    mask_index: int = -100,
    **kwargs,
) -> Scalar:
    """Q-loss for classification tasks with masking support.

    Args:
        true_y: Ground truth labels
        pred_y: Predicted logits
        mask_index: Index to mask out (default: -100)
        **kwargs: Additional arguments

    Returns:
        Scalar loss value
    """
    # if true_y is one-hot encoded, we need to convert it back
    if true_y.ndim > 1:
        indices = jnp.argmax(true_y, axis=-1)
    else:
        # only batch dimension
        indices = true_y

    # Create a mask to ignore tokens marked with mask_index
    mask = indices != mask_index

    pred_y_at_target_index = jnp.take_along_axis(
        pred_y,
        jnp.expand_dims(indices, -1).astype(jnp.int32),
        axis=-1,
    )

    # Apply the mask to exclude mask_index tokens from loss computation
    masked_pred_y = jnp.where(mask[..., None], pred_y_at_target_index, 0.0)

    # divide by the valid tokens to get the mean loss
    return -jnp.sum(masked_pred_y) / jnp.sum(mask)
