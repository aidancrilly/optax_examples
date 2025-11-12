import jax
import jax.numpy as jnp
import optax

def optimize(initial_y, optimizer, loss, num_steps=100):
    opt_state = optimizer.init(initial_y)

    def step(carry, _):
        opt_state, y = carry
        l, grad = jax.value_and_grad(loss)(y)
        updates, opt_state = optimizer.update(grad, opt_state, y)
        y = optax.apply_updates(y, updates)
        jax.debug.print("Loss: {l}", l=l)
        return (opt_state, y), y

    carry = (opt_state, initial_y)

    _, ys = jax.lax.scan(step, carry, jnp.zeros(num_steps))

    return ys
