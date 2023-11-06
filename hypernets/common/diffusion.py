import jax
import jax.numpy as jnp

def diffusion_schedule(diffusion_times, min_signal_rate, max_signal_rate):
    start_angle = jnp.arccos(max_signal_rate)
    end_angle = jnp.arccos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = jnp.cos(diffusion_angles)
    noise_rates = jnp.sin(diffusion_angles)
    return noise_rates, signal_rates

def reverse_diffusion(
    apply_fn, 
    params,
    num_images, 
    diffusion_steps, 
    context_length,
    token_dim, 
    diffusion_schedule_fn,
    min_signal_rate,
    max_signal_rate,
    seed, 
    initial_noise = None,
):
    if initial_noise == None:
        initial_noise = jax.random.normal(
            jax.random.PRNGKey(seed), 
            shape=(num_images, context_length, token_dim)
        )
    step_size = 1.0 / diffusion_steps
    
    next_noisy_images = initial_noise
    for step in range(diffusion_steps):
        noisy_images = next_noisy_images
        
        diffusion_times = jnp.ones((num_images, 1, 1)) - step * step_size
        noise_rates, signal_rates = diffusion_schedule_fn(
            diffusion_times, min_signal_rate, max_signal_rate
        )
        pred_noises = jax.lax.stop_gradient(
            apply_fn(
                {'params': params}, 
                [noisy_images, noise_rates**2], 
            )
        )
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        
        next_diffusion_times = diffusion_times - step_size
        next_noise_rates, next_signal_rates = diffusion_schedule_fn(
            next_diffusion_times, min_signal_rate, max_signal_rate
        )
        next_noisy_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)
    return pred_images