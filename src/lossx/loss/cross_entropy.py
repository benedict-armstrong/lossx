"""Cross-entropy loss functions for classification tasks."""

import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree, Scalar


def cross_entropy(
    true_y: PyTree[Array],
    pred_y: PyTree[Array],
    *,
    cls_weights: list[float] | None = None,
    mask_index: int = -100,
    for_each_timestep: bool = False,
    **kwargs,
) -> Scalar:
    """Compute cross-entropy loss between true and predicted labels.

    Args:
        true_y: Ground truth labels
        pred_y: Predicted logits
        cls_weights: Optional weights for each class
        mask_index: Index to mask out (default: -100)
        for_each_timestep: Whether to compute loss for each timestep
        **kwargs: Additional arguments

    Returns:
        Scalar cross-entropy loss
    """
    # if true_y is one-hot encoded, we need to convert it back
    if true_y.ndim > 1 and not for_each_timestep:
        indices = jnp.argmax(true_y, axis=-1)
    else:
        # only batch dimension
        indices = true_y

    # Create a mask to ignore tokens marked with mask_index
    mask = indices != mask_index

    weights = jnp.ones(pred_y.shape[-1])
    if cls_weights is not None:
        weights = jnp.array(cls_weights)

    pred_y_at_target_index = jnp.take_along_axis(
        pred_y * weights,
        jnp.expand_dims(indices, -1).astype(jnp.int32),
        axis=-1,
    )

    # Apply the mask to exclude mask_index tokens from loss computation
    masked_pred_y = jnp.where(mask[..., None], pred_y_at_target_index, 0.0)

    # divide by the valid tokens to get the mean loss
    return -jnp.sum(masked_pred_y) / jnp.sum(mask)
