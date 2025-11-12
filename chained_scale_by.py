import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from losses import mean_squared_nth_order_diff
from utils import optimize

"""Experiment: smoothing a spike using L-BFGS + learning-rate scaling.

This script minimizes a finite-difference smoothing loss starting from a
spiky 1D signal, comparing multiple learning rates when chaining
``scale_by_lbfgs`` with ``scale_by_learning_rate``.
"""

loss = mean_squared_nth_order_diff(n=1)

N = 100
initial_y = jnp.zeros(N)
# Introduce a sharp spike in the middle to be smoothed by the optimizer.
initial_y = initial_y.at[N//2].set(1.0)

lrs = jnp.array([1.0, 0.5, 0.1, 0.01])

# Plot intermediate trajectories for each learning rate setting.
fig = plt.figure(figsize=(10, 6),dpi=200)
axs = fig.subplots(2, 2)
for ax, lr in zip(axs.flatten(), lrs):
    # Chain an L-BFGS-style preconditioner with a plain learning rate.
    optimizer = optax.chain(
        optax.scale_by_lbfgs(),
        optax.scale_by_learning_rate(lr)
        )
    ys = optimize(initial_y, optimizer, loss, num_steps=200)
    for i in range(0, ys.shape[0], 20):
        ax.plot(ys[i,:], label=f'Step {i}')
    ax.set_title(f'Learning Rate: {lr}')
    ax.legend()

fig.tight_layout()

plt.show()
