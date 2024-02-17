import jax
import jax.numpy as jnp

def diffusion_schedule(diffusion_times, min_signal_rate, max_signal_rate):
    start_angle = jnp.arccos(max_signal_rate)
    end_angle = jnp.arccos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = jnp.cos(diffusion_angles)
    noise_rates = jnp.sin(diffusion_angles)
    return noise_rates, signal_rates

def ddim_sample(
    state, 
    num_samples:int, 
    diffusion_steps:int, 
    diffusion_schedule,
    token_dim:int, 
    context_length:int,
    min_signal_rate:float,
    max_signal_rate:float,
    noise_clip:float,
    seed:int, 
):
    @jax.jit
    def inference_fn(state, noisy_batch, diffusion_times):
        return jax.lax.stop_gradient(
            state.apply_fn({'params': state.params}, noisy_batch, diffusion_times)
        )
    
    initial_noise = jax.random.normal(
        jax.random.PRNGKey(seed), 
        shape=(num_samples, context_length, token_dim)
    )
    initial_noise = jnp.clip(initial_noise, -noise_clip, noise_clip)
    step_size = 1.0 / diffusion_steps
    
    next_noisy_batch = initial_noise
    for step in range(diffusion_steps):
        noisy_batch = next_noisy_batch
        
        diffusion_times = jnp.ones((num_samples, 1, 1)) - step * step_size
        noise_rates, signal_rates = diffusion_schedule(
            diffusion_times, min_signal_rate, max_signal_rate
        )
        pred_noises = inference_fn(state, noisy_batch, noise_rates**2)
        pred_batch = (noisy_batch - noise_rates * pred_noises) / signal_rates
        
        next_diffusion_times = diffusion_times - step_size
        next_noise_rates, next_signal_rates = diffusion_schedule(
            next_diffusion_times, min_signal_rate, max_signal_rate
        )
        next_noisy_batch = (next_signal_rates * pred_batch + next_noise_rates * pred_noises)
    return pred_batch
