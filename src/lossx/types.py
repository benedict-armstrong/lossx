"""Type definitions for loss functions and reductions."""

from typing import NotRequired, Protocol, TypedDict, TypeVar

from jaxtyping import Array, PyTree, Scalar

T = TypeVar("T", bound=PyTree[Array])


class LossFn(Protocol[T]):
    """Protocol for loss functions.

    Loss functions should take two PyTrees of the same structure and reduce them to a scalar loss.
    """

    def __call__(self, true: T, pred: T, **kwargs) -> Scalar:
        """Compute loss between two PyTrees of the same structure.

        Args:
            true: Ground truth PyTree
            pred: Predicted PyTree (must have same structure as true)
            **kwargs: Additional loss-specific parameters

        Returns:
            Scalar loss value
        """
        ...


R = TypeVar("R", bound=Scalar)


class Reduction(Protocol[R]):
    """Protocol for reduction functions.

    Reduction functions should take a list of scalars and reduce them to a single scalar.
    """

    def __call__(self, losses: PyTree[R]) -> R:
        """Reduce a PyTree of scalars to a single scalar."""
        ...


class LossArgsCfg(TypedDict):
    """Configuration for a single loss function."""

    target: str  # loss function name
    weight: NotRequired[float]
    # ... any loss-specific kwargs


LossCfg = PyTree[LossArgsCfg]
