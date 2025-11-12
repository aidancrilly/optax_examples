import jax
import jax.numpy as jnp
import optax

def optimize(initial_y, optimizer, loss, num_steps=100):
    """Run an optimization loop with Optax over a 1D variable.

    Args:
        initial_y: Initial 1D array to optimize.
        optimizer: An Optax ``GradientTransformation``.
        loss: Callable accepting ``y`` and returning a scalar loss.
        num_steps: Number of optimization iterations to run.

    Returns:
        A stacked array of intermediate ``y`` values across iterations
        with shape ``[num_steps, ...]``.
    """
    opt_state = optimizer.init(initial_y)

    def step(carry, _):
        opt_state, y = carry
        # Compute loss and gradient w.r.t. current parameters.
        l, grad = jax.value_and_grad(loss)(y)
        # Transform gradients and update parameters.
        updates, opt_state = optimizer.update(grad, opt_state, y)
        y = optax.apply_updates(y, updates)
        # Lightweight logging to observe optimization progress.
        jax.debug.print("Loss: {l}", l=l)
        return (opt_state, y), y

    carry = (opt_state, initial_y)

    # Use ``lax.scan`` to run the loop efficiently and collect all states.
    _, ys = jax.lax.scan(step, carry, jnp.zeros(num_steps))

    return ys
