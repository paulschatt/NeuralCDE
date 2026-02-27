import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from jaxtyping import Array, Float, Key


class RNNForecaster(eqx.Module):
    """
        Initial Version was AI-generated using ChatGPT-5.1 on 18/11/2025
        
        Initial Prompt: Implement a GRU for timeseries forecasting using Equinox


        This is pretty much equivalent to the example in the docs: https://docs.kidger.site/equinox/api/nn/rnn/
        except that we're using a readout.
    """
    cell: eqx.nn.GRUCell
    readout: eqx.nn.Linear
    h0: Array

    def __init__(self, input_size, hidden_size, output_size, *, key):
        k1, k2, k3 = jr.split(key, 3) 
        
   
        
        self.cell = eqx.nn.GRUCell(input_size, hidden_size, key=k1) # Note: input_size in GRUCell is now initializer_input_size
        self.readout = eqx.nn.Linear(hidden_size, 'scalar', key=k2)
        self.h0 = jnp.zeros(hidden_size)
        

    @eqx.filter_jit
    def __call__(self, ts, ys, control_until, saveat, train_until):
        # Warm-up with [time, value]
        xs_warm = jnp.stack([ts[:control_until], ys[:control_until]], axis=-1)
        
        def warm_step(h, x):
            h = self.cell(x, h)
            y = self.readout(h)
            return h, y
        
        h_after, warm_preds = jax.lax.scan(warm_step, self.h0, xs_warm)
        
        last_input = xs_warm[-1]  # Keep full [time, value] format
        rollout_steps = train_until - control_until
        
        # Pass the future times as the scanned sequence
        future_times = ts[control_until:train_until]
        
        def ar_step(carry, t):  # t comes from future_times
            h, x_prev = carry
            # x_prev is [time_prev, value_prev]
            
            # Feed previous prediction to cell
            h = self.cell(x_prev, h)
            y = self.readout(h)  # Predict next value
            
            # Construct next input as [t_next, y_next]
            x_next = jnp.array([t, y])
            
            return (h, x_next), x_next
        
        (h_final, _), ar_preds = jax.lax.scan(
            ar_step,
            (h_after, last_input),
            future_times  # Scan over future times
        )
        
        # Concatenate and extract values (index 1 if output has [time, value])
        all_preds = jnp.concatenate([warm_preds, ar_preds[:, 1]], axis=0)
        return all_preds.squeeze()
