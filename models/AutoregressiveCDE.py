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
from .control_utils import generate_increasing_flag


class AutoregressiveCDE(eqx.Module):
    """
        Iteratively extrapolates the control signal step by step
    """

    func: Func
    hidden_size: int
    data_size: int

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

        self.data_size = data_size
        self.hidden_size = hidden_size

        self.func = Func(data_size, hidden_size, width_size, depth, key=func_key)

    @eqx.filter_jit
    def __call__(self,
                 ts,
                 ys,
                 control_until,
                 saveat,
                 train_until):
        
        t_times_y = ts[:len(ys)]*ys
        ys_aug  = jnp.stack([ts[:len(ys)], ys, t_times_y], axis=-1)
        control = generate_control(ys_aug[:control_until])

        controlled_solution = diffrax.diffeqsolve(
            terms=diffrax.ControlTerm(self.func, control).to_ode(),
            solver=diffrax.Tsit5(),
            max_steps=32000,
            t0=ts[0],
            t1=ts[control_until],
            dt0=(ts[1] - ts[0]) / 5,
            y0=control.evaluate(ts[0]),
            saveat=diffrax.SaveAt(ts=ts[:control_until])
        )
        
        all_ys = jnp.zeros((100, self.hidden_size))
        all_ys = all_ys.at[:control_until].set(controlled_solution.ys)
        
        chunk_size = 1
        total_remaining = train_until - control_until
        num_chunks = (total_remaining + chunk_size - 1) // chunk_size
        chunk_starts = control_until + jnp.arange(num_chunks) * chunk_size


        """
            I know this code is not very readable,
            but it is a necessary evil, since it achieves
            a 10-20x speedup compared to a for loop.

            The rewrite (and debugging) from using a for loop 
            to jax.lax.scan was assisted by Claude Sonnet 4.5 
            and ChatGPT-5.1 on 20/11/2025

            Initial Prompt to GPT-5.1: 

            '''
                Rewrite this for loop using jax.lax.scan:

                def __call__(self,
                    ts: Array,
                    ys: Array,
                    control_until, 
                    saveat: Array,
                    train_until):
        
                    control = generate_control(ys[:control_until])

                    controlled_solution = diffrax.diffeqsolve(
                        terms=diffrax.ControlTerm(self.func, control).to_ode(),
                        solver=diffrax.Tsit5(),
                        max_steps=32000,
                        t0=ts[0],
                        t1=ts[control_until],
                        dt0=(ts[1]-ts[0])/5,
                        y0=self.encoder(control.evaluate(ts[0])),
                        saveat=diffrax.SaveAt(ts=ts[:(control_until)])
                    )

                    final_hidden_ys = controlled_solution.ys
                    decoded = jax.vmap(self.decoder)(final_hidden_ys).squeeze()
                    
                    for current_control in range(control_until, train_until, 7):
                        last_t_index = min(train_until, current_control+7)

                        
                        intermediate_solution = diffrax.diffeqsolve(
                            terms=diffrax.ControlTerm(self.func, control).to_ode(),
                            solver=diffrax.Tsit5(),
                            max_steps=32000,
                            t0=ts[current_control],
                            t1=ts[current_control+6],
                            dt0=(ts[1]-ts[0])/5,
                            y0=final_hidden_ys[-1],
                            saveat=diffrax.SaveAt(ts=ts[current_control:last_t_index])
                        )
                        final_hidden_ys = jnp.concatenate([final_hidden_ys, intermediate_solution.ys], axis=0) 
                        decoded = jax.vmap(self.decoder)(final_hidden_ys).squeeze()
                        control = generate_control(decoded)

                    return decoded
            '''
            
        """
        def scan_step(carry, start_idx):
            all_current_ys = carry
            
            ts_scan = jnp.linspace(0, 1, 100)
            
            #TODO: clean up these index shifts
            #necessary because we need the last state of the previous iteration as an intial state
            ts_chunk = jax.lax.dynamic_slice(ts_scan, (start_idx-1,), (chunk_size+1,))

            extrapolated_ys = masked_linear_extrapolation(all_current_ys, start_idx)
            extrapolated_ys = extrapolated_ys.at[:, 0].set(ts)
            extrapolated_ys = extrapolated_ys.at[:, 2].set(extrapolated_ys[:, 1]*ts_scan)
            control = generate_control(extrapolated_ys)
            
    
            solution = diffrax.diffeqsolve(
                terms=diffrax.ControlTerm(self.func, control).to_ode(),
                solver=diffrax.Tsit5(),
                max_steps=32000,
                t0=ts_chunk[0],
                t1=ts_chunk[-1],
                dt0=(ts_scan[1] - ts_scan[0]) / 5,
                y0=all_current_ys[start_idx - 1],
                saveat=diffrax.SaveAt(ts=ts_chunk[1:])
            )
            
            all_ys_new = jax.lax.dynamic_update_slice(
                all_current_ys,
                solution.ys,
                (start_idx, 0)
            )
            
            return all_ys_new, None

        all_ys, _ = jax.lax.scan(
            f=scan_step,
            init=(all_ys),
            xs=chunk_starts
        )
        
        return all_ys[:, 1]
        

def masked_linear_extrapolation(xs, valid_length):
    """
    This function was also AI-generated in the same conversation
    where I let ChatGPT generate the AutoregressiveCDE loop using jax.lax.scan


    xs: (T, H) array of hidden states.
    valid_length: Python int, number of valid entries (0 ≤ L ≤ T)
    After valid_length, fill linearly based on slope at last valid point.
    """

    T, H = xs.shape
    L = valid_length                    # must be Python int outside scan!

    # last real value
    last = xs[L-1]                      # shape (H,)

    # estimate slope using last two real values
    # safe even for L=1: use zero slope
    slope = jnp.where(
        L > 1,
        xs[L-1] - xs[L-2],
        jnp.zeros_like(last)
    )                                    # shape (H,)

    # indices 0..T-1
    idx = jnp.arange(T)

    # mask for valid region
    mask = idx < L                      # shape (T,)

    # t distance from last known point (0,1,2,...)
    dt = idx - (L - 1)                  # shape (T,)

    # linear extrapolation: last + dt * slope
    extrap = last + dt[:, None] * slope[None, :]

    # combine
    xs_filled = jnp.where(mask[:, None], xs, extrap)

    return xs_filled


def generate_control(ys):
    ts = jnp.linspace(0, 1, 100)[:len(ys)]
    return diffrax.LinearInterpolation(ts=ts, ys=ys)


def extrapolate(ys):
    """
        Extrapolates an array of ys to length 100
        The last element is duplicated as many times as needed to have 100 values
    """
    ts = jnp.linspace(0, 1, 100)
    interp = diffrax.LinearInterpolation(ts=ts[:len(ys)], ys=ys)
    return jax.vmap(interp.evaluate)(ts)
    