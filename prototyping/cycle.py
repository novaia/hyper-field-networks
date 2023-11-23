from functools import partial
import jax.numpy as jnp
import time

def cyclical_linear_schedule(initial_value, final_value, transition_steps, cycle_steps):
    slope = final_value / transition_steps
    linear_fn = partial(
        lambda m, b, max_y, x: jnp.clip(m * x + b, 0, max_y), 
        slope, initial_value, final_value
    )
    cyclical_fn = partial(lambda period, x: linear_fn(x % period), cycle_steps)
    return cyclical_fn

steps_per_epoch = 100
schedule = cyclical_linear_schedule(
    initial_value=0.0, 
    final_value=1.0, 
    transition_steps=steps_per_epoch//2,
    cycle_steps=steps_per_epoch
)
for i in range(steps_per_epoch*3):
    print(i, schedule(i))
    time.sleep(0.1)