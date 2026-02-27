import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from jaxtyping import Array, Float, Key


class Persistence(eqx.Module):
    """
        Naive baseline that repeats the last observation.
    """
    def __init__(self):
        pass
        
    @eqx.filter_jit
    def __call__(self, ts, ys, control_until, saveat, train_until):

        last_y = ys[control_until]

        extrapolation_length = train_until - control_until
        extrapolated_ys = jnp.full((extrapolation_length,), last_y)
        return jnp.concatenate([ys, extrapolated_ys], axis=0)