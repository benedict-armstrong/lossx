# LossX

A functional JAX loss function library with PyTree-based composition and Hydra integration.

## Features

- **Pure Functional API** - All loss functions are pure functions
- **PyTree-Based Composition** - Build complex multi-task losses from nested configurations
- **Hydra Integration** - Seamlessly integrate with Hydra configs
- **Flexible Reductions** - Custom reduction strategies for combining multiple losses
- **Extensible** - Easy to add custom loss functions

## Installation

### Using uv (recommended)

```bash
uv add lossx
```

### Using pip

```bash
pip install lossx
```

### Or just copy one of the source files into your project

## Quick Start

### Simple Loss Function Usage

Import and use loss functions directly:

```python
import jax.numpy as jnp
from lossx.loss import cross_entropy, mse, gaussian_nll

# Mean Squared Error
true_y = jnp.array([1.0, 2.0, 3.0])
pred_y = jnp.array([1.1, 2.1, 3.1])
loss = mse(true_y, pred_y)

# Cross-Entropy with masking
true_labels = jnp.array([0, 1, 2, -100])  # -100 is masked
pred_logits = jnp.ones((4, 3))
loss = cross_entropy(true_labels, pred_logits, mask_index=-100)

# Gaussian NLL (for uncertainty estimation)
true_y = jnp.array([[1.0], [2.0]])
pred_y = jnp.array([[1.0, 0.1], [2.0, 0.1]])  # mean and variance
loss = gaussian_nll(true_y, pred_y, eps=1e-6)
```

## Available Loss Functions

- `cross_entropy` - Cross-entropy loss with masking and class weights
- `mse` - Mean squared error
- `q_loss` - Q-loss for classification
- `quantile_loss` - Quantile loss for uncertainty estimation
- `gaussian_nll` - Gaussian negative log-likelihood
- `penex` - Penalized exponential loss
- `contrastive` - NT-Xent contrastive loss

## PyTree-Based Loss Building

The real power of LossX comes from building complex losses using PyTree structures:

### Single Loss

```python
from lossx import build_loss

config = {
    "target": "cross_entropy",
    "mask_index": -100,
    "weight": 1.0
}

loss_fn = build_loss(config)
loss = loss_fn(true_y, pred_y)
```

### Multi-Task Learning (Dict of Losses)

```python
import jax.numpy as jnp

config = {
    "classification": {
        "target": "cross_entropy",
        "mask_index": -100,
        "weight": 1.0
    },
    "regression": {
        "target": "mse",
        "weight": 0.5
    },
    "uncertainty": {
        "target": "gaussian_nll",
        "eps": 1e-6,
        "weight": 0.3
    }
}

# Default reduction is sum
loss_fn = build_loss(config, reduction=lambda losses: jnp.mean(jnp.array(losses)))

# Provide PyTree-structured inputs matching the config
losses = loss_fn(
    true={
        "classification": class_labels,
        "regression": regression_targets,
        "uncertainty": uncertainty_targets
    },
    pred={
        "classification": class_logits,
        "regression": regression_preds,
        "uncertainty": uncertainty_preds
    }
)
```

### Composite Loss (List of Losses)

```python
config = [
    {"target": "mse", "weight": 1.0},
    {"target": "cross_entropy", "mask_index": -100, "weight": 2.0}
]

loss_fn = build_loss(config, reduction=lambda losses: jnp.sum(jnp.array(losses)))

# Provide list-structured inputs
loss = loss_fn(
    true=[regression_targets, class_labels],
    pred=[regression_preds, class_logits]
)
```

### Nested PyTree Structures

```python
config = {
    "main": [
        {"target": "mse"},
        {"target": "cross_entropy"}
    ],
    "auxiliary": {
        "task_a": {"target": "mse", "weight": 0.5},
        "task_b": {"target": "gaussian_nll"}
    }
}

loss_fn = build_loss(config)

# Inputs must match the same PyTree structure
loss = loss_fn(
    true={
        "main": [main_target1, main_target2],
        "auxiliary": {
            "task_a": aux_a_target,
            "task_b": aux_b_target
        }
    },
    pred={
        "main": [main_pred1, main_pred2],
        "auxiliary": {
            "task_a": aux_a_pred,
            "task_b": aux_b_pred
        }
    }
)
```

## Custom Reductions

You can provide custom reduction functions to combine losses:

```python
import jax.numpy as jnp

# Weighted reduction
weights = jnp.array([0.7, 0.3])
reduction = lambda losses: jnp.sum(jnp.array(losses) * weights)

# Max reduction (worst-case loss)
reduction = lambda losses: jnp.max(jnp.array(losses))

# Mean reduction
reduction = lambda losses: jnp.mean(jnp.array(losses))

loss_fn = build_loss(configs, reduction=reduction)
```

## Hydra Integration

LossX is designed to work seamlessly with Hydra configs:

**config.yaml:**

```yaml
loss:
  target: cross_entropy
  mask_index: -100
  cls_weights: [1.0, 2.0, 1.0]
  weight: 1.0
```

**multi_task.yaml:**

```yaml
loss:
  classification:
    target: cross_entropy
    mask_index: -100
    weight: 1.0
  regression:
    target: mse
    weight: 0.5
  uncertainty:
    target: gaussian_nll
    eps: 1e-6
    weight: 0.3

reduction: mean  # or sum, weighted
```

**Python code:**

```python
import hydra
from lossx import build_loss
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Convert OmegaConf to dict
    loss_config = dict(cfg.loss)
    loss_fn = build_loss(loss_config)

    # Use in training
    loss = loss_fn(targets, predictions)
```

## Custom Loss Functions

Register your own loss functions:

```python
from lossx import register_loss
import jax.numpy as jnp

def custom_loss(true_y, pred_y, *, alpha=0.5, **kwargs):
    """Custom loss function."""
    return alpha * jnp.mean((true_y - pred_y) ** 2)

# Register it
register_loss("custom", custom_loss)

# Now use it in configs
config = {"target": "custom", "alpha": 0.7}
loss_fn = build_loss(config)
```

## API Reference

### Loss Functions

All loss functions follow the signature:

```python
def loss_fn(
    true_y: PyTree[Array],
    pred_y: PyTree[Array],
    **kwargs
) -> Scalar
```

### Builder Functions

#### `build_loss(config, reduction=jnp.sum)`

Build a loss function from a configuration PyTree.

**Args:**

- `config`: Either a dict with `"target"` key (single loss) or a PyTree of such dicts
- `reduction`: Function to reduce multiple losses to a scalar (default: `jnp.sum`)

**Returns:**

- A loss function that takes `(true, pred)` PyTrees and returns a scalar

#### `register_loss(name, loss_fn)`

Register a custom loss function.

**Args:**

- `name`: Name to register under
- `loss_fn`: Loss function to register

## Development

### Running Tests

```bash
uv run pytest tests/ -v
```

### Installing Development Dependencies

```bash
uv add --group dev
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
