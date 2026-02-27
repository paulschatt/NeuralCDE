import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import diffrax
from jaxtyping import Array, Float, Key


class LinearForecaster(eqx.Module):
    """
        Naive Baseline that linearly extrapolates
    """
    def __init__(self):
        pass
        
    @eqx.filter_jit
    def __call__(self, ts, ys, control_until, saveat, train_until):

        interp_ys = diffrax.linear_interpolation(ts[:control_until], ys)

        control = diffrax.LinearInterpolation(ts[:control_until], interp_ys)

        extrapolated_ys = jax.vmap(control.evaluate)(ts) 

        return extrapolated_ys[:train_until]