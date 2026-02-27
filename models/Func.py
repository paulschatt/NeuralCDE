import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp


class Func(eqx.Module):
    """
        Taken from https://docs.kidger.site/diffrax/examples/neural_cde/

        Hidden size is actually not needed, since our models don't have hidden states
        But I didn't have time to change all signatures
    """
    mlp: eqx.nn.MLP
    data_size: int
    hidden_size: int

    def __init__(self, data_size, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size+1,
            out_size=(hidden_size)*(data_size),
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            final_activation=jnn.tanh,
            key=key,
        )
    
    @eqx.filter_jit
    def __call__(self, t, y, args):
        y = jnp.concatenate([y, jnp.array([t])]) #Since we're building non-autonomous CDEs
        return self.mlp(y).reshape(self.hidden_size, self.data_size)