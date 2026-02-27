import jax
import jax.numpy as jnp

from .train_utils import standardize, destandardize, inverse_log_transform


def integrated_gradients_all_outputs(model: callable,
                                     control_until, 
                                     predict_until,
                                     ys: jnp.ndarray,
                                     baseline: jnp.ndarray,
                                     steps: int,
                                     standardize_baseline: bool):
    """
        See a check of the Completeness Axiom of this in the notebook
        verify_ig_works_with_standardization.
    """

    ts = jnp.linspace(0, 1, 100)

    ys, mean, std = standardize(ys)

    if(standardize_baseline):
        baseline = (baseline - mean)/std

    alphas = jnp.linspace(0.0, 1.0, steps)
    interpolated_inputs = baseline + alphas[:, None] * (ys - baseline)

    def wrapper(input):
        """
            To preserve Completeness we must treat the inverse log transforms as part of the function being differentiated
        """
        return inverse_log_transform(model(ts, ys=input[:control_until], control_until=control_until, saveat=ts, train_until=predict_until) * std + mean)
    jac_fn = jax.jacrev(wrapper)     
    jacobian_path = jax.vmap(jac_fn)(interpolated_inputs)   
    avg_jacobian = jacobian_path.mean(axis=0)              


    ig = avg_jacobian * (ys - baseline)[None, :]
    
    return ig