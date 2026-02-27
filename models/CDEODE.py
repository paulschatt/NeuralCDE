from itertools import starmap
from typing import final
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array, Float, Key, PyTree
import diffrax
import jax.random as jr

from .Func import Func

class ODEField(eqx.Module):
    mlp: eqx.nn.MLP
    data_size: int
    hidden_size: int


    def __init__(self, data_size, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size+1,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            final_activation=jnn.tanh, 
            key=key)

    @eqx.filter_jit
    def __call__(self, t, y, args):
        y = jnp.concatenate([y, jnp.array([t])])
        out = self.mlp(y)
        return out



class CDEODE(eqx.Module):
    func: Func
    ode_term: ODEField

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
        self.func = Func(data_size, data_size, width_size, depth, key=func_key)
        self.ode_term = ODEField(data_size, data_size, width_size, depth, key=final_key)
        

    @eqx.filter_jit
    def __call__(self,
                ts: Array,
                ys: Array,
                control_until,
                saveat: Array,
                train_until):
        

        control = generate_control(ys)

        controlled_solution = diffrax.diffeqsolve(
            terms=diffrax.ControlTerm(self.func, control).to_ode(),
            solver=diffrax.Tsit5(),
            max_steps=32000,
            t0=ts[0],
            t1=ts[control_until-1],
            dt0=(ts[1]-ts[0])/5,
            y0=control.evaluate(ts[0]),
            saveat=diffrax.SaveAt(ts=ts[:control_until]),
        )

        final_ys = controlled_solution.ys

        extrapolated_solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self.ode_term),
            solver=diffrax.Tsit5(),
            max_steps=32000,
            t0=ts[control_until-1],
            t1=ts[train_until-1],
            dt0=(ts[1]-ts[0])/5,
            y0=final_ys[-1],
            saveat=diffrax.SaveAt(ts=ts[control_until:train_until]),
            )
        final_ys = jnp.concatenate((final_ys, extrapolated_solution.ys))

        return final_ys[:, 1]
    
def generate_control(ys):
    ts = jnp.linspace(0, 1, 100)[:len(ys)]
    t_times_y = ts*ys
    ys_aug = jnp.stack([ts, ys, t_times_y], axis=-1)
    return diffrax.LinearInterpolation(ts=ts[:len(ys)], ys=ys_aug)