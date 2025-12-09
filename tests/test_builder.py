"""Tests for the loss builder with different PyTree structures."""

import jax.numpy as jnp
import pytest
from jax import random

from lossx import build_loss, register_loss


@pytest.fixture
def rng_key():
    """Provide a JAX random key."""
    return random.PRNGKey(42)


class TestSingleLoss:
    """Tests for building single loss functions."""

    def test_build_mse_loss(self):
        """Test building a single MSE loss."""
        config = {"target": "mse"}
        loss_fn = build_loss(config)

        true_y = jnp.array([1.0, 2.0, 3.0])
        pred_y = jnp.array([1.5, 2.5, 3.5])
        loss = loss_fn(true_y, pred_y)

        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_build_cross_entropy_with_kwargs(self):
        """Test building cross-entropy with keyword arguments."""
        config = {"target": "cross_entropy", "mask_index": -100}
        loss_fn = build_loss(config)

        true_y = jnp.array([0, 1, -100])
        pred_y = jnp.ones((3, 3))
        loss = loss_fn(true_y, pred_y)

        assert jnp.isfinite(loss)

    def test_build_loss_with_weight(self):
        """Test that weight parameter scales the loss."""
        config_no_weight = {"target": "mse"}
        config_with_weight = {"target": "mse", "weight": 2.0}

        loss_fn_1 = build_loss(config_no_weight)
        loss_fn_2 = build_loss(config_with_weight)

        true_y = jnp.array([1.0, 2.0])
        pred_y = jnp.array([1.5, 2.5])

        loss_1 = loss_fn_1(true_y, pred_y)
        loss_2 = loss_fn_2(true_y, pred_y)

        assert jnp.allclose(loss_2, 2.0 * loss_1)


class TestListOfLosses:
    """Tests for building composite losses from lists."""

    def test_list_of_losses_sum_reduction(self, rng_key):
        """Test building a list of losses with sum reduction."""
        configs = [
            {"target": "mse"},
            {"target": "mse"},
        ]
        loss_fn = build_loss(configs, reduction=lambda losses: jnp.sum(jnp.array(losses)))

        true_y = [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])]
        pred_y = [jnp.array([1.5, 2.5]), jnp.array([3.5, 4.5])]

        loss = loss_fn(true_y, pred_y)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_list_of_losses_mean_reduction(self):
        """Test building a list of losses with mean reduction."""
        configs = [
            {"target": "mse", "weight": 1.0},
            {"target": "mse", "weight": 2.0},
        ]
        loss_fn = build_loss(configs, reduction=lambda losses: jnp.mean(jnp.array(losses)))

        true_y = [jnp.array([1.0]), jnp.array([1.0])]
        pred_y = [jnp.array([2.0]), jnp.array([2.0])]

        loss = loss_fn(true_y, pred_y)
        assert jnp.isfinite(loss)

    def test_list_of_different_losses(self, rng_key):
        """Test building a list with different loss types."""
        configs = [
            {"target": "mse"},
            {"target": "cross_entropy"},
        ]
        loss_fn = build_loss(configs, reduction=lambda losses: jnp.sum(jnp.array(losses)))

        true_y = [
            jnp.array([1.0, 2.0]),  # For MSE
            jnp.array([0, 1, 2]),  # For cross-entropy
        ]
        pred_y = [
            jnp.array([1.5, 2.5]),
            random.normal(rng_key, (3, 3)),
        ]

        loss = loss_fn(true_y, pred_y)
        assert jnp.isfinite(loss)


class TestDictOfLosses:
    """Tests for building multi-task losses from dicts."""

    def test_dict_of_losses_basic(self):
        """Test building a dict of losses for multi-task learning."""
        config = {
            "task1": {"target": "mse"},
            "task2": {"target": "mse"},
        }
        loss_fn = build_loss(config, reduction=lambda losses: jnp.sum(jnp.array(losses)))

        true_y = {
            "task1": jnp.array([1.0, 2.0]),
            "task2": jnp.array([3.0, 4.0]),
        }
        pred_y = {
            "task1": jnp.array([1.5, 2.5]),
            "task2": jnp.array([3.5, 4.5]),
        }

        loss = loss_fn(true_y, pred_y)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_dict_with_different_losses(self, rng_key):
        """Test dict with different loss types for different tasks."""
        config = {
            "regression": {"target": "mse"},
            "classification": {"target": "cross_entropy"},
            "uncertainty": {"target": "gaussian_nll"},
        }
        loss_fn = build_loss(config, reduction=lambda losses: jnp.mean(jnp.array(losses)))

        true_y = {
            "regression": jnp.array([1.0, 2.0]),
            "classification": jnp.array([0, 1]),
            "uncertainty": jnp.array([[1.0], [2.0]]),
        }
        pred_y = {
            "regression": jnp.array([1.5, 2.5]),
            "classification": random.normal(rng_key, (2, 3)),
            "uncertainty": jnp.array([[1.0, 0.1], [2.0, 0.1]]),
        }

        loss = loss_fn(true_y, pred_y)
        assert jnp.isfinite(loss)

    def test_dict_with_weights(self):
        """Test dict of losses with different weights."""
        config = {
            "task1": {"target": "mse", "weight": 1.0},
            "task2": {"target": "mse", "weight": 2.0},
        }
        loss_fn = build_loss(config, reduction=lambda losses: jnp.sum(jnp.array(losses)))

        # Use same data for both tasks to test weighting
        data = jnp.array([1.0])
        pred = jnp.array([2.0])

        true_y = {"task1": data, "task2": data}
        pred_y = {"task1": pred, "task2": pred}

        loss = loss_fn(true_y, pred_y)

        # Loss should be 3x the single task loss (1.0 + 2.0 times base loss)
        single_loss = (2.0 - 1.0) ** 2
        expected = 1.0 * single_loss + 2.0 * single_loss
        assert jnp.allclose(loss, expected)


class TestNestedPyTrees:
    """Tests for building losses with nested PyTree structures."""

    def test_nested_dict_of_lists(self):
        """Test nested structure: dict containing lists."""
        config = {
            "main": [
                {"target": "mse"},
                {"target": "mse"},
            ],
            "aux": {"target": "mse"},
        }
        loss_fn = build_loss(config, reduction=lambda losses: jnp.sum(jnp.array(losses)))

        true_y = {
            "main": [jnp.array([1.0]), jnp.array([2.0])],
            "aux": jnp.array([3.0]),
        }
        pred_y = {
            "main": [jnp.array([1.5]), jnp.array([2.5])],
            "aux": jnp.array([3.5]),
        }

        loss = loss_fn(true_y, pred_y)
        assert jnp.isfinite(loss)

    def test_list_of_dicts(self):
        """Test list containing dicts."""
        config = [
            {"task_a": {"target": "mse"}},
            {"task_b": {"target": "mse"}},
        ]
        loss_fn = build_loss(config, reduction=lambda losses: jnp.sum(jnp.array(losses)))

        true_y = [
            {"task_a": jnp.array([1.0])},
            {"task_b": jnp.array([2.0])},
        ]
        pred_y = [
            {"task_a": jnp.array([1.5])},
            {"task_b": jnp.array([2.5])},
        ]

        loss = loss_fn(true_y, pred_y)
        assert jnp.isfinite(loss)

    def test_deeply_nested_structure(self):
        """Test deeply nested PyTree structure."""
        config = {
            "level1": {
                "level2": [
                    {"target": "mse"},
                    {"target": "mse"},
                ],
            }
        }
        loss_fn = build_loss(config, reduction=lambda losses: jnp.sum(jnp.array(losses)))

        true_y = {
            "level1": {
                "level2": [jnp.array([1.0]), jnp.array([2.0])],
            }
        }
        pred_y = {
            "level1": {
                "level2": [jnp.array([1.5]), jnp.array([2.5])],
            }
        }

        loss = loss_fn(true_y, pred_y)
        assert jnp.isfinite(loss)


class TestCustomReductions:
    """Tests for custom reduction functions."""

    def test_sum_reduction(self):
        """Test sum reduction."""
        configs = [
            {"target": "mse"},
            {"target": "mse"},
        ]

        def sum_reduction(losses):
            return jnp.sum(jnp.array(losses))

        loss_fn = build_loss(configs, reduction=sum_reduction)

        true_y = [jnp.array([1.0]), jnp.array([1.0])]
        pred_y = [jnp.array([2.0]), jnp.array([2.0])]

        loss = loss_fn(true_y, pred_y)
        # Each loss is (2-1)^2 = 1, sum should be 2
        assert jnp.allclose(loss, 2.0)

    def test_mean_reduction(self):
        """Test mean reduction."""
        configs = [
            {"target": "mse"},
            {"target": "mse"},
        ]

        def mean_reduction(losses):
            return jnp.mean(jnp.array(losses))

        loss_fn = build_loss(configs, reduction=mean_reduction)

        true_y = [jnp.array([1.0]), jnp.array([1.0])]
        pred_y = [jnp.array([2.0]), jnp.array([2.0])]

        loss = loss_fn(true_y, pred_y)
        # Each loss is (2-1)^2 = 1, mean should be 1
        assert jnp.allclose(loss, 1.0)

    def test_weighted_reduction(self):
        """Test weighted reduction."""
        configs = [
            {"target": "mse"},
            {"target": "mse"},
        ]

        weights = jnp.array([0.3, 0.7])

        def weighted_reduction(losses):
            return jnp.sum(jnp.array(losses) * weights)

        loss_fn = build_loss(configs, reduction=weighted_reduction)

        true_y = [jnp.array([1.0]), jnp.array([1.0])]
        pred_y = [jnp.array([2.0]), jnp.array([2.0])]

        loss = loss_fn(true_y, pred_y)
        # Each loss is 1.0, weighted sum should be 0.3 + 0.7 = 1.0
        assert jnp.allclose(loss, 1.0)

    def test_max_reduction(self):
        """Test max reduction (take worst loss)."""
        configs = [
            {"target": "mse"},
            {"target": "mse"},
        ]

        def max_reduction(losses):
            return jnp.max(jnp.array(losses))

        loss_fn = build_loss(configs, reduction=max_reduction)

        true_y = [jnp.array([1.0]), jnp.array([1.0])]
        pred_y = [jnp.array([2.0]), jnp.array([3.0])]  # Different predictions

        loss = loss_fn(true_y, pred_y)
        # Losses are 1.0 and 4.0, max should be 4.0
        assert jnp.allclose(loss, 4.0)


class TestRegisterLoss:
    """Tests for registering custom loss functions."""

    def test_register_custom_loss(self):
        """Test registering and using a custom loss function."""

        def custom_mae(true_y, pred_y, **kwargs):
            """Mean absolute error."""
            return jnp.mean(jnp.abs(true_y - pred_y))

        register_loss("mae", custom_mae)

        config = {"target": "mae"}
        loss_fn = build_loss(config)

        true_y = jnp.array([1.0, 2.0, 3.0])
        pred_y = jnp.array([1.5, 2.5, 3.5])
        loss = loss_fn(true_y, pred_y)

        expected = jnp.mean(jnp.abs(true_y - pred_y))
        assert jnp.allclose(loss, expected)

    def test_register_duplicate_name_raises_error(self):
        """Test that registering a duplicate name raises an error."""

        def dummy_loss(true_y, pred_y, **kwargs):
            return jnp.mean(true_y)

        # MSE is already registered
        with pytest.raises(ValueError, match="already registered"):
            register_loss("mse", dummy_loss)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_loss_name(self):
        """Test that invalid loss name raises error."""
        config = {"target": "nonexistent_loss"}

        with pytest.raises(KeyError):
            build_loss(config)

    def test_mismatched_pytree_structures(self):
        """Test that mismatched PyTree structures raise error."""
        config = {
            "task1": {"target": "mse"},
            "task2": {"target": "mse"},
        }
        loss_fn = build_loss(config, reduction=lambda losses: jnp.sum(jnp.array(losses)))

        # Provide mismatched structures
        true_y = {"task1": jnp.array([1.0])}  # Missing task2
        pred_y = {"task1": jnp.array([1.5]), "task2": jnp.array([2.5])}

        # JAX tree_map should raise an error for mismatched structures
        with pytest.raises(ValueError):
            loss_fn(true_y, pred_y)
