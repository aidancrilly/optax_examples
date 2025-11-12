import jax.numpy as jnp

"""Simple loss utilities based on finite differences.

This module provides small building-block losses to encourage smoothness in
1D arrays by penalizing n-th order finite differences.
"""

def mean_squared_nth_order_diff(n):
    """Create a loss that penalizes the n-th order difference.

    Args:
        n: Order of the finite difference to apply (e.g., 1 for first-order).

    Returns:
        A function mapping an array ``y`` to the mean squared n-th order
        difference of ``y``.
    """
    def loss(y):
        diff_y = jnp.diff(y, n=n)
        return jnp.mean(diff_y ** 2)
    return loss
