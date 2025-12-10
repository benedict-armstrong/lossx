"""Tests for individual loss functions."""

import jax.numpy as jnp
import pytest
from jax import random

from lossx.loss import (
    contrastive,
    cross_entropy,
    focal,
    gaussian_nll,
    mse,
    penex,
    q_loss,
    quantile_loss,
)


@pytest.fixture
def rng_key():
    """Provide a JAX random key."""
    return random.PRNGKey(42)


class TestMSE:
    """Tests for MSE loss function."""

    def test_mse_basic(self):
        """Test basic MSE computation."""
        true_y = jnp.array([1.0, 2.0, 3.0])
        pred_y = jnp.array([1.5, 2.5, 3.5])
        loss = mse(true_y, pred_y)
        expected = jnp.mean((true_y - pred_y) ** 2)
        assert jnp.allclose(loss, expected)

    def test_mse_with_index(self):
        """Test MSE with specific index selection."""
        true_y = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        pred_y = jnp.array([[1.5, 2.5], [3.5, 4.5]])
        loss = mse(true_y, pred_y, index=0)
        expected = jnp.mean((true_y[:, 0] - pred_y[:, 0]) ** 2)
        assert jnp.allclose(loss, expected)

    def test_mse_perfect_prediction(self):
        """Test MSE with perfect prediction."""
        true_y = jnp.array([1.0, 2.0, 3.0])
        pred_y = jnp.array([1.0, 2.0, 3.0])
        loss = mse(true_y, pred_y)
        assert jnp.allclose(loss, 0.0)


class TestCrossEntropy:
    """Tests for cross-entropy loss function."""

    def test_cross_entropy_basic(self, rng_key):
        """Test basic cross-entropy computation."""
        true_y = jnp.array([0, 1, 2, 0])
        pred_y = random.normal(rng_key, (4, 3))
        loss = cross_entropy(true_y, pred_y)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_cross_entropy_with_masking(self):
        """Test cross-entropy with masked indices."""
        true_y = jnp.array([0, 1, -100, 0])  # -100 should be masked
        pred_y = jnp.ones((4, 3))
        loss = cross_entropy(true_y, pred_y, mask_index=-100)
        # Should only compute loss on 3 elements (not the masked one)
        assert jnp.isfinite(loss)

    def test_cross_entropy_with_class_weights(self):
        """Test cross-entropy with class weights."""
        true_y = jnp.array([0, 1, 2])
        pred_y = jnp.ones((3, 3))
        loss_weighted = cross_entropy(true_y, pred_y, cls_weights=[2.0, 1.0, 1.0])
        loss_unweighted = cross_entropy(true_y, pred_y)
        # Weighted loss should be different
        assert not jnp.allclose(loss_weighted, loss_unweighted)


class TestFocal:
    """Tests for focal loss function."""

    def test_focal_basic(self):
        """Test basic focal loss computation."""
        true_y = jnp.array([1.0, 0.0, 1.0, 0.0])
        pred_y = jnp.array([2.0, -1.0, 1.5, -2.0])
        loss = focal(true_y, pred_y)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_focal_with_gamma(self):
        """Test focal loss with different gamma values."""
        true_y = jnp.array([1.0, 0.0, 1.0])
        pred_y = jnp.array([1.0, -1.0, 2.0])

        loss_gamma2 = focal(true_y, pred_y, gamma=2.0)
        loss_gamma3 = focal(true_y, pred_y, gamma=3.0)

        # Different gamma values should give different losses
        assert not jnp.allclose(loss_gamma2, loss_gamma3)

    def test_focal_with_epsilon(self):
        """Test focal loss with epsilon parameter."""
        true_y = jnp.array([1.0, 0.0, 1.0])
        pred_y = jnp.array([1.0, -1.0, 2.0])

        loss_no_eps = focal(true_y, pred_y, epsilon=0.0)
        loss_with_eps = focal(true_y, pred_y, epsilon=0.5)

        # Epsilon should affect the loss
        assert not jnp.allclose(loss_no_eps, loss_with_eps)

    def test_focal_with_class_weights(self):
        """Test focal loss with class weights."""
        true_y = jnp.array([1.0, 0.0, 1.0])
        pred_y = jnp.array([1.0, -1.0, 2.0])

        loss_unweighted = focal(true_y, pred_y)
        loss_weighted = focal(true_y, pred_y, cls_weights=[2.0, 1.0])

        # Class weights should affect the loss
        assert not jnp.allclose(loss_unweighted, loss_weighted)


class TestQLoss:
    """Tests for Q-loss function."""

    def test_q_loss_basic(self, rng_key):
        """Test basic Q-loss computation."""
        true_y = jnp.array([0, 1, 2])
        pred_y = random.normal(rng_key, (3, 3))
        loss = q_loss(true_y, pred_y)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_q_loss_with_masking(self):
        """Test Q-loss with masked indices."""
        true_y = jnp.array([0, 1, -100])
        pred_y = jnp.ones((3, 3))
        loss = q_loss(true_y, pred_y, mask_index=-100)
        assert jnp.isfinite(loss)


class TestQuantileLoss:
    """Tests for quantile loss function."""

    def test_quantile_loss_basic(self):
        """Test basic quantile loss computation."""
        true_y = jnp.array([[1.0], [2.0], [3.0]])
        # pred_y should have 3 components: mean, lower quantile, upper quantile
        pred_y = jnp.array([[1.0, 0.8, 1.2], [2.0, 1.8, 2.2], [3.0, 2.8, 3.2]])
        loss = quantile_loss(true_y, pred_y)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_quantile_loss_custom_quantiles(self):
        """Test quantile loss with custom quantile values."""
        true_y = jnp.array([[1.0], [2.0]])
        pred_y = jnp.array([[1.0, 0.5, 1.5], [2.0, 1.5, 2.5]])
        loss = quantile_loss(true_y, pred_y, quantiles=(0.1, 0.9))
        assert jnp.isfinite(loss)


class TestGaussianNLL:
    """Tests for Gaussian NLL loss function."""

    def test_gaussian_nll_basic(self):
        """Test basic Gaussian NLL computation."""
        true_y = jnp.array([[1.0], [2.0], [3.0]])
        # pred_y should have mean and variance stacked
        pred_y = jnp.array([[1.0, 0.1], [2.0, 0.1], [3.0, 0.1]])
        loss = gaussian_nll(true_y, pred_y)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_gaussian_nll_with_eps(self):
        """Test Gaussian NLL with epsilon parameter."""
        true_y = jnp.array([[1.0], [2.0]])
        pred_y = jnp.array([[1.0, 1e-10], [2.0, 1e-10]])  # Very small variance
        loss = gaussian_nll(true_y, pred_y, eps=1e-6)
        assert jnp.isfinite(loss)

    def test_gaussian_nll_perfect_prediction(self):
        """Test Gaussian NLL with perfect mean prediction."""
        true_y = jnp.array([[1.0], [2.0]])
        pred_y = jnp.array([[1.0, 1.0], [2.0, 1.0]])
        loss = gaussian_nll(true_y, pred_y)
        # Loss should be related to log(variance) when mean is perfect
        assert jnp.isfinite(loss)


class TestPenex:
    """Tests for Penex loss function."""

    def test_penex_basic(self, rng_key):
        """Test basic Penex loss computation."""
        true_y = jnp.array([0, 1, 2])
        pred_y = random.normal(rng_key, (3, 3))
        loss = penex(true_y, pred_y, penalty=0.1)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_penex_with_ignore_index(self):
        """Test Penex with ignored indices."""
        true_y = jnp.array([0, 1, -100])
        pred_y = jnp.ones((3, 3))
        loss = penex(true_y, pred_y, ignore_index=-100, penalty=0.1)
        assert jnp.isfinite(loss)

    def test_penex_different_reductions(self, rng_key):
        """Test Penex with different reduction modes."""
        true_y = jnp.array([0, 1, 2])
        pred_y = random.normal(rng_key, (3, 3))

        loss_mean = penex(true_y, pred_y, penalty=0.1, reduction="mean")
        loss_sum = penex(true_y, pred_y, penalty=0.1, reduction="sum")

        # Sum should be larger than mean
        assert loss_sum > loss_mean


class TestContrastive:
    """Tests for contrastive loss function."""

    def test_contrastive_basic(self, rng_key):
        """Test basic contrastive loss computation."""
        # 4 samples with 2 views each = 8 embeddings
        embeddings = random.normal(rng_key, (8, 128))
        true_y = jnp.array([])  # Not used
        loss = contrastive(true_y, embeddings, temperature=0.1, num_views=2)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_contrastive_temperature_effect(self, rng_key):
        """Test that temperature affects the loss."""
        embeddings = random.normal(rng_key, (8, 128))
        true_y = jnp.array([])

        loss_low_temp = contrastive(true_y, embeddings, temperature=0.1, num_views=2)
        loss_high_temp = contrastive(true_y, embeddings, temperature=1.0, num_views=2)

        # Different temperatures should give different losses
        assert not jnp.allclose(loss_low_temp, loss_high_temp)

    def test_contrastive_invalid_batch_size(self, rng_key):
        """Test that invalid batch size raises error."""
        embeddings = random.normal(rng_key, (7, 128))  # Not divisible by num_views=2
        true_y = jnp.array([])

        with pytest.raises(RuntimeError):
            contrastive(true_y, embeddings, num_views=2)

    def test_contrastive_single_sample(self, rng_key):
        """Test contrastive with single sample returns zero."""
        embeddings = random.normal(rng_key, (1, 128))
        true_y = jnp.array([])
        loss = contrastive(true_y, embeddings, num_views=2)
        assert jnp.allclose(loss, 0.0)
