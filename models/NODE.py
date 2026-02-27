from itertools import starmap
from typing import final
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array, Float, Key, PyTree
import diffrax
import jax.random as jr


"""
    I used this to get a general understanding of Neural Differential Equations
    before I moved on to CDEs
"""

class Func(eqx.Module):
    mlp: eqx.nn.MLP
    data_size: int
    hidden_size: int
    norm: eqx.nn.LayerNorm


    def __init__(self, data_size, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size+1,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            final_activation=jnn.tanh,
            key=key,
        )

        self.norm = eqx.nn.LayerNorm(shape=((hidden_size,)))

    @eqx.filter_jit
    def __call__(self, t, y, args):
        y = jnp.concatenate([y, jnp.array([t])])
        out = self.mlp(y)
        if out.ndim == 1:
            out = self.norm(out)
        else:
            out = jax.vmap(self.norm)(out)
        return out

class NODE(eqx.Module):
    encoder: eqx.nn.MLP
    func: Func
    decoder: eqx.nn.MLP


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

        self.encoder = eqx.nn.MLP(
            in_size='scalar', 
            out_size=hidden_size, 
            width_size=width_size, 
            depth=depth, 
            final_activation=lambda x: x, 
            key=initial_key)
        
        self.func = Func(data_size, hidden_size, width_size, depth, key=func_key)

        self.decoder = eqx.nn.MLP(
            in_size=hidden_size, 
            out_size=1,
            width_size=width_size,
            depth=depth,
            final_activation=jnn.softplus,
            key=final_key
        )

    @eqx.filter_jit
    def __call__(self,
                ts: Array,
                ys: Array,
                control_until,
                saveat: Array,
                train_until):
        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self.func),
            solver=diffrax.Tsit5(),
            max_steps=32000,
            t0=ts[0],
            t1=ts[train_until-1],
            dt0=(ts[1]-ts[0])/5,
            y0=self.encoder(ys[0]),
            saveat=diffrax.SaveAt(ts=ts[:train_until]),
            )

        return jax.vmap(self.decoder)(solution.ys).squeeze()