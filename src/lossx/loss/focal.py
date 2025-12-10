"""Focal loss for classification tasks."""

import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree, Scalar


def focal(
    true_y: PyTree[Array],
    pred_y: PyTree[Array],
    *,
    gamma: float = 2.0,
    epsilon: float = 0.0,
    cls_weights: list[float] | None = None,
) -> Scalar:
    """Focal loss for classification tasks."""
    # alpha_m = minority_ratio
    # alpha_M = majority_ratio
    if cls_weights is not None:
        alpha_m, alpha_M, *_ = cls_weights
    else:
        alpha_m = 1.0
        alpha_M = 1.0
    gamma_d = gamma + 1.0
    p = 1 / (1 + jnp.exp(-pred_y))
    q = 1 - p

    focal_pos = jnp.power(q, gamma) * jnp.log(p)
    poly1_pos = epsilon * jnp.power(q, gamma_d)
    pos_loss = jnp.add(focal_pos, poly1_pos) * alpha_m

    focal_neg = jnp.power(p, gamma) * jnp.log(q)
    poly1_neg = epsilon * jnp.power(p, gamma_d)
    neg_loss = jnp.add(focal_neg, poly1_neg) * alpha_M

    return true_y * pos_loss + (1 - true_y) * neg_loss
