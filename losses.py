import jax.numpy as jnp

def mean_squared_nth_order_diff(n):
    def loss(y):
        diff_y = jnp.diff(y, n=n)
        return jnp.mean(diff_y ** 2)
    return loss

# def eigenvalue_nth_order_diff(n, N):
#     D = jnp.zeros((N - n, N))
