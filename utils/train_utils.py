from socketserver import ForkingUDPServer
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import equinox as eqx
import jax.random
import optax

from enum import Enum

class PredictionMode(Enum):
    FOURTEEN_DAYS_AHEAD = 1
    UNTIL_DAY_100 = 2

"""
    HELPER FUNCTIONS
    ---------------------------------------
"""
def huber_loss(y, y_hat):
    """
        Huber loss (delta = 50) for ground truth and prediction of a single training example.
    """
    delta = 50
    a = jnp.abs(y - y_hat)
    loss = jnp.where(a <= delta, 0.5 * a**2, delta * (a - 0.5 * delta))
    return jnp.mean(loss)

def rmse(y, y_hat):
    """
        RMSE loss for ground truth and prediction of a single training example.
    """
    return jnp.sqrt(jnp.mean((y-y_hat)**2))

def inverse_log_transform(transformed_ys):
    """
        Inverse transform of the log transform y_{transformed} = log_{10}(y + 1)

        We only need to call the inverse transform during training, since the dataloader
        handles the original transform.
    """
    return ((10**transformed_ys)-1)

def generate_random_cutoffs(length, minval, maxval):
    key = jax.random.PRNGKey(0)
    control_untils = jax.random.randint(key, (length,), minval=minval, maxval=maxval)
    return control_untils

def standardize(ys):
    """
        Floating point std won't be exactly 0.0, hence the 'jnp.isclose'
    """
    mean = jnp.mean(ys)
    std = jnp.std(ys)
    return jax.lax.cond(
        jnp.isclose(std, 0.0),
        lambda _: (ys, mean, std),    
        lambda _: ((ys-mean)/std, mean, std),    
        operand=None
    )

def destandardize(ys, mean, std):
    """
        Floating point std won't be exactly 0.0, hence the 'jnp.isclose'
    """
    return jax.lax.cond(
        jnp.isclose(std, 0.0),
        lambda _: ys,    
        lambda _: ys*std + mean,    
        operand=None
    )
"""
    ---------------------------------------
"""

@eqx.filter_jit
def RMSE_loss(model, control_ys, all_ys, control_until, train_until):
    """
        Loss function for a single training example.
        Performs the model call and returns the loss.

        Params:
            model: The neural CDE model

            control_until: The day until which the control signal is given

            control_ys: ys that are used to generate the control signal

            all_ys: Used to calculate the loss

            train_until: The day until which the solution is saved and the loss is evaluated

        We have to pass control_ys and all_ys since control_until is a traced value, so we
        can't dynamically take a slice
    """
    ts = jnp.linspace(0, 1, 100)

    control_ys, mean, std = standardize(control_ys)

    y_pred = model(ts, ys=control_ys, control_until=control_until, saveat=ts, train_until=train_until)

    y_pred = destandardize(y_pred, mean, std)

    return rmse(inverse_log_transform(all_ys)[control_until:train_until], inverse_log_transform(y_pred)[control_until:train_until])

@eqx.filter_jit
def batch_loss(model, control_ys, all_ys, control_until, train_until):
    """
        Calculates batch loss for a given loss

        Params: 
            model:  The model which is being called

            control_until: The day until which the control signal is given

            control_ys: ys that are used to generate the control signal

            all_ys: Used to calculate the loss (shape: (B,100,))

            train_until: The day until which the solution is saved and the loss is evaluated
    """
    def single_loss_wrapper(control_ys, all_ys):
        return RMSE_loss(model, control_ys, all_ys, control_until, train_until)

    losses = jax.vmap(single_loss_wrapper, in_axes=(0, 0))(control_ys, all_ys)
    return jnp.mean(losses)

@eqx.filter_value_and_grad
def batch_loss_with_gradients(model, control_ys, all_ys, control_until, train_until):
    """ 
        This is just a call to the the batch loss function that also
        returns the gradients (see 'eqx.filter_value_and_grad' annotation).

        Params: 
            model:  The model which is being called

            control_until: The day until which the control signal is given

            control_ys: ys that are used to generate the control signal

            all_ys: Used to calculate the loss (shape: (B,100,))

            train_until: The day until which the solution is saved and the loss is evaluated
    """
    return batch_loss(model, control_ys, all_ys, control_until, train_until)

def validation_loss(model, ys):
    """ 
        Validation loss
        
        Test ability of the model to forecast with an input length between 10 and 86 using RMSE.


        Params: 
            model:  The model which is being called
            ys: Used to calculate the loss (shape: (B,100,))
            prediction_mode: Decides whether the forecast will be evaluated 14 days ahead or until day 100
    """
    size = ys.shape[0]
    slice_size = size // 9

    losses = []

    for i, control_until in enumerate([10, 20, 30, 40, 50, 60, 70, 80, 86]):
        start = i*slice_size
        end = start + slice_size

        ys_slice = ys[start:end]
        control_ys = ys_slice[:, :control_until]

        train_until = control_until + 14

        loss = batch_loss(model, control_ys, ys_slice, control_until, train_until)
        losses.append(loss)
    return jnp.mean(jnp.array(losses))


def validation_loss_until_100(model, ys):
    """ 
        Validation loss
        
        Test ability of the model to forecast with an input length between 10 and 86 using RMSE.


        Params: 
            model:  The model which is being called
            ys: Used to calculate the loss (shape: (B,100,))
            prediction_mode: Decides whether the forecast will be evaluated 14 days ahead or until day 100
    """
    size = ys.shape[0]
    slice_size = size // 9

    losses = []

    for i, control_until in enumerate([10, 20, 30, 40, 50, 60, 70, 80, 86]):
        start = i*slice_size
        end = start + slice_size

        ys_slice = ys[start:end]
        control_ys = ys_slice[:, :control_until]

        loss = batch_loss(model, control_ys, ys_slice, control_until, 100)
        losses.append(loss)
    return jnp.mean(jnp.array(losses))


def test_loss(model, ys):
    """ 
        Validation loss
        
        Test ability of the model to forecast with an input length between 10 and 86 using RMSE.


        Params: 
            model:  The model which is being called
            ys: Used to calculate the loss (shape: (B,100,))
            prediction_mode: Decides whether the forecast will be evaluated 14 days ahead or until day 100
    """
    size = ys.shape[0]
    slice_size = size // 9

    losses = []

    for control_until in [10, 20, 30, 40, 50, 60, 70, 80, 86]:
        control_ys = ys[:, :control_until]

        train_until = control_until + 14

        loss = batch_loss(model, control_ys, ys, control_until, train_until)
        losses.append(loss)

        print(f"Loss for input length of {control_until}: {loss}")
    total_mean_test_loss =  jnp.mean(jnp.array(losses))
    print(f"Mean Test Loss: {total_mean_test_loss}")
    return total_mean_test_loss


def test_loss_until_100(model, ys):
    """ 
        Validation loss
        
        Test ability of the model to forecast with an input length between 10 and 86 using RMSE.


        Params: 
            model:  The model which is being called
            ys: Used to calculate the loss (shape: (B,100,))
            prediction_mode: Decides whether the forecast will be evaluated 14 days ahead or until day 100
    """
    size = ys.shape[0]
    slice_size = size // 9

    losses = []

    for control_until in [10, 20, 30, 40, 50, 60, 70, 80, 86]:
        control_ys = ys[:, :control_until]

        loss = batch_loss(model, control_ys, ys, control_until, 100)
        losses.append(loss)

        print(f"Loss for input length of {control_until}: {loss}")
    total_mean_test_loss =  jnp.mean(jnp.array(losses))
    print(f"Mean Test Loss: {total_mean_test_loss}")
    return total_mean_test_loss


def train_step(model, control_ys, all_ys, optimizer, loss_fn, opt_state, control_until, train_until):
    """
        Performs model calls, loss calculation and weight updates for a single batch.

        Params: 
            model: The model which is being called
            optimizer: optimizer
            loss_fn: A batch loss function
            opt_state: Current state of the optimizer (learning rate etc.)
            control_until: Day until which the control is given in this batch
            train_until: Day until which the solutions are evaluated in this batch
    """
   
    loss, grads = loss_fn(model, control_ys, all_ys, control_until, train_until)
    params = eqx.filter(model, eqx.is_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

   

def plot_loss_history(train_losses, validation_losses):
    """
        Plots training and validation loss for a training run.

        This function is not in the plots file to avoid circular imports.
    """
    train_epochs = [i for i in range(0, len(train_losses))]
    validation_epochs = [i for i in train_epochs if i % 10 == 0]

    fig, ax = plt.subplots(1, 1,figsize=(5,5))
    ax.semilogy(train_epochs, train_losses, label='Train Loss')
    ax.plot(validation_epochs, validation_losses, label='Validation Loss')
    ax.set(xlabel=r'$Training Steps$', ylabel=r'$MSE Loss$', yscale='log')
    ax.grid(True)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def train_14_days_ahead(model, dataloader):
    train_loss_history = []
    validation_loss_history = []

    num_steps = 100
    batch_size = 32

    learning_rate_schedule = optax.exponential_decay(
            init_value=1e-3,       
            transition_steps=300,  
            decay_rate=0.95,      
            end_value=1e-4     
        )

    optimizer = optax.chain(
        optax.adamw( 
                    learning_rate=learning_rate_schedule,
                    weight_decay=1e-5,
                    b1=0.9,
                    b2=0.999
                    )
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    for min_control in [70, 50, 40, 35, 30, 25, 20, 15, 10]:
        control_untils = generate_random_cutoffs(num_steps, min_control, 71)
        for step, control_until in enumerate(control_untils):
            
            control_until = int(control_until)

            batch_ys = dataloader.sample_batch(batch_size)
            control_ys = batch_ys[:, :control_until]

            model, opt_state, loss = train_step(model=model, 
                                                            control_ys=control_ys,
                                                            all_ys=batch_ys,
                                                            optimizer=optimizer,
                                                            loss_fn=batch_loss_with_gradients,
                                                            opt_state=opt_state,
                                                            control_until=control_until,
                                                            train_until=control_until+14)

            if (step%10 == 0):
                valid_loss = validation_loss(model, ys=dataloader.validation_data)
                validation_loss_history.append(valid_loss)
                print(f"Control until day {control_until}, Step {step}: Loss = {(loss):.4f}, Validation Loss = {(valid_loss):.4f}")

            else:
                print(f"Control until day {control_until}, Step {step}: Loss = {(loss):.4f}")
            train_loss_history.append(loss)

    plot_loss_history(train_losses=train_loss_history, validation_losses=validation_loss_history)
    return model


def train_until_day_100(model, dataloader):
    train_loss_history = []
    validation_loss_history = []

    num_epochs = 100
    batch_size = 32

    learning_rate_schedule = optax.exponential_decay(
            init_value=1e-3,       
            transition_steps=300,  
            decay_rate=0.95,      
            end_value=1e-4     
        )

    optimizer = optax.chain(
        optax.adamw( 
                    learning_rate=learning_rate_schedule,
                    weight_decay=1e-5,
                    b1=0.9,
                    b2=0.999
                    )
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    for min_control in [70, 50, 40, 35, 30, 25, 20, 15, 10]:
        control_untils = generate_random_cutoffs(num_epochs, min_control, 71)
        for step, control_until in enumerate(control_untils):

            control_until = int(control_until)
            
            batch_ys = dataloader.sample_batch(batch_size)
            control_ys = batch_ys[:, :control_until]

            model, opt_state, loss = train_step(model=model, 
                                                            control_ys=control_ys,
                                                            all_ys=batch_ys,
                                                            optimizer=optimizer,
                                                            loss_fn=batch_loss_with_gradients,
                                                            opt_state=opt_state,
                                                            control_until=control_until, 
                                                            train_until=100)

            if (step%10 == 0):
                valid_loss = validation_loss_until_100(model, ys=dataloader.validation_data)
                validation_loss_history.append(valid_loss)
                print(f"Control until day {control_until}, Step {step}: Loss = {(loss):.4f}, Validation Loss = {(valid_loss):.4f}")

            else:
                print(f"Control until day {control_until}, Step {step}: Loss = {(loss):.4f}")
            train_loss_history.append(loss)

    plot_loss_history(train_losses=train_loss_history, validation_losses=validation_loss_history)
    return model


