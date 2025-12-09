"""Contrastive loss functions for self-supervised learning."""

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree, Scalar


def contrastive(
    true_y: PyTree[Array],
    pred_y: PyTree[Array],
    *,
    temperature: float = 0.1,
    num_views: int = 2,
    **kwargs,
) -> Scalar:
    """NT-Xent (InfoNCE) contrastive loss for self-supervised learning.

    This loss encourages representations of different views of the same sample
    to be similar (positive pairs) while pushing apart representations of
    different samples (negative pairs).

    Args:
        true_y: Not used for contrastive learning
        pred_y: Embeddings of shape (N*C, embed_dim) where N is batch size
               and C is number of views per sample
        temperature: Temperature parameter for scaling similarities
        num_views: Number of views per sample (default: 2)
        **kwargs: Additional arguments

    Returns:
        Scalar contrastive loss value
    """
    batch_size = pred_y.shape[0]
    if batch_size <= 1:
        return jnp.array(0.0)

    # Assume augmented views are grouped together for each sample
    if batch_size % num_views != 0:
        raise RuntimeError(f"Batch size {batch_size} must be divisible by num_views {num_views}")

    num_samples = batch_size // num_views

    norm_embeddings = pred_y / (jnp.linalg.norm(pred_y, axis=1, keepdims=True) + 1e-8)
    similarity_matrix = jnp.matmul(norm_embeddings, norm_embeddings.T)

    labels = jnp.arange(num_samples).repeat(num_views)[:batch_size]
    positive_mask = labels[:, None] == labels[None, :]
    self_mask = jnp.eye(batch_size, dtype=bool)
    positive_mask = positive_mask & ~self_mask

    # Scale similarities by the temperature parameter
    sim_div_temp = similarity_matrix / temperature

    # Numerator: logsumexp over positive pairs for each anchor
    log_numerator = jax.nn.logsumexp(jnp.where(positive_mask, sim_div_temp, -jnp.inf), axis=1)

    # Denominator: logsumexp over all other pairs for each anchor
    log_denominator = jax.nn.logsumexp(jnp.where(~self_mask, sim_div_temp, -jnp.inf), axis=1)

    # The final loss is the average over the batch
    loss = jnp.mean(log_denominator - log_numerator)
    return loss
