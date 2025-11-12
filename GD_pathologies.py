import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from losses import mean_squared_nth_order_diff
from utils import optimize

"""Gradient-descent behaviors across learning rates with Adam.

This script optimizes a simple smoothing loss starting from a spiky signal
and visualizes trajectories for several learning rates using Adam. It helps
illustrate how step size affects convergence and smoothness.
"""

loss = mean_squared_nth_order_diff(n=1)

N = 100
initial_y = jnp.zeros(N)
# Introduce a sharp spike in the middle to stress-test optimization dynamics.
initial_y = initial_y.at[N//2].set(1.0)

lrs = jnp.array([1.0, 0.5, 0.1, 0.01])

# Plotting the results
fig = plt.figure(figsize=(10, 6),dpi=200)
axs = fig.subplots(2, 2)
for ax, lr in zip(axs.flatten(), lrs):
    # Adam optimizer at different learning rates.
    optimizer = optax.adam(learning_rate=lr)
    ys = optimize(initial_y, optimizer, loss, num_steps=200)
    for i in range(0, ys.shape[0], 20):
        ax.plot(ys[i,:], label=f'Step {i}')
    ax.set_title(f'Learning Rate: {lr}')
    ax.legend()

fig.tight_layout()

plt.show()
