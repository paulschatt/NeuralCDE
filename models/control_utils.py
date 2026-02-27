import jax.numpy as jnp






def generate_increasing_flag(control_length):
     """
        Generates a flag to indicate whether the control signal is extrapolated or not.
        0 if control signal is valid.
        Starts linearly increasing when the control signal ends.
        Divide by 100 to keep it in the same order of magnitude as time.
    """
     full_ramp = jnp.arange(100) / 100
     full_ramp = full_ramp - control_length / 100
     mask = (jnp.arange(100) >= control_length)
     return full_ramp * mask 
