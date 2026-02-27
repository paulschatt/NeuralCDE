import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array, Float, Key, PyTree
import diffrax
import jax.random as jr
import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from .Func import Func


class SingleSolveCDE(eqx.Module): 
    func: Func
    
    def __init__(self,
                 data_size : int,
                 hidden_size : int,
                 width_size : int, 
                 depth : int,
                 *,
                 key : Key,
                 **kwargs):
        super().__init__(**kwargs)

        initial_key, func_key, final_key = jr.split(key, 3)
        self.func = Func(data_size, hidden_size, width_size, depth, key=func_key)
      
    @eqx.filter_jit
    def __call__(self,
                ts: Array,
                ys: Array,
                control_until,
                saveat: Array,
                train_until):

        control = generate_control_without_flag(ys[:control_until])

        solution = diffrax.diffeqsolve(
            terms=diffrax.ControlTerm(self.func, control).to_ode(),
            solver=diffrax.Tsit5(),
            max_steps=32000,
            t0=ts[0],
            t1=ts[train_until-1],
            dt0=(ts[1]-ts[0])/5,
            y0=control.evaluate(ts[0]),
            saveat=diffrax.SaveAt(ts=saveat[:train_until])
        )
        
        return solution.ys[: , 0]

def generate_control_without_flag(ys):
    ts = jnp.linspace(0, 1, 100)
        
    control = diffrax.LinearInterpolation(ts=ts[:len(ys)], ys=ys)
    extrapolated_ys = jax.vmap(control.evaluate)(ts)

    t_times_ys = extrapolated_ys * ts

    final_ys = jnp.concatenate([extrapolated_ys[:, None], ts[:, None], t_times_ys[:, None]], axis=-1)

    return diffrax.LinearInterpolation(ts=ts, ys=final_ys)





"""
    Code from older experiments.
"""

def generate_control(ys):
    ts = jnp.linspace(0, 1, 100)
    
    flag = generate_increasing_flag(len(ys))
    
    extrapolated_ys = extrapolate(ys)

    t_times_ys = extrapolated_ys * ts

    final_ys = jnp.concatenate([extrapolated_ys[:, None], ts[:, None], t_times_ys[:, None], flag[:, None]], axis=-1)

    return diffrax.LinearInterpolation(ts=ts, ys=final_ys)


def generate_increasing_flag(control_length):
    """
    Generates a flag to indicate whether the control signal is extrapolated or not.
    0 if control signal is valid.
    Starts linearly increasing when the control signal ends.
    Divide by 100 to keep it in the same order of magnitude as time.
    """
    extrapolation_length = 100 - control_length

    zeros = jnp.zeros(control_length)
    increasing_flag = jnp.array([i / 100 for i in range(1, extrapolation_length+1)])
    return jnp.concatenate([zeros, increasing_flag], axis=0)


def extrapolate(ys):
    """
        Extrapolates an array of ys to length 100
        The last element is duplicated as many times as needed to have 100 values
    """
    ts = jnp.linspace(0, 1, 100)
    interp = diffrax.LinearInterpolation(ts=ts[:len(ys)], ys=ys)
    return jax.vmap(interp.evaluate)(ts)