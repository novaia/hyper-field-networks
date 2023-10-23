import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from functools import partial
import os
import matplotlib.pyplot as plt

def sinusoidal_embedding(x, embedding_max_frequency, embedding_dims):
    embedding_min_frequency = 1.0
    frequencies = jnp.exp(
        jnp.linspace(
            jnp.log(embedding_min_frequency),
            jnp.log(embedding_max_frequency),
            embedding_dims // 2
        )
    )
    angular_speeds = 2.0 * jnp.pi * frequencies
    embeddings = jnp.concatenate(
        [jnp.sin(angular_speeds * x), jnp.cos(angular_speeds * x)],
        axis = -1
    )
    return embeddings

class DenseResidualBlock(nn.Module):
    width: int

    @nn.compact
    def __call__(self, x):
        input_width = x.shape[-1]
        if input_width == self.width:
            residual = x
        else:
            residual = nn.Dense(self.width)(x)
        x = nn.Dense(self.width)(x)
        x = nn.activation.swish(x)
        x = nn.Dense(self.width)(x)
        x = x + residual
        x = nn.LayerNorm()(x)
        return x

class DenseDownBlock(nn.Module):
    width: int
    block_depth: int

    @nn.compact
    def __call__(self, x):
        x, skips = x
        for _ in range(self.block_depth):
            x = DenseResidualBlock(self.width)(x)
            skips.append(x)
        return x

class DenseUpBlock(nn.Module):
    width: int
    block_depth: int

    @nn.compact
    def __call__(self, x):
        x, skips = x
        for _ in range(self.block_depth):
            x = DenseResidualBlock(self.width)(x)
            skips.append(x)
        return x
    
class DenseDDIM(nn.Module):
    embedding_max_frequency: float
    embedding_dims: int
    widths: list
    block_depth: int
    feature_dim: int

    @nn.compact
    def __call__(self, x):
        x, noise_variances = x

        e = sinusoidal_embedding(
            noise_variances, 
            self.embedding_max_frequency, 
            self.embedding_dims
        )
        e = jax.image.resize(e, shape=(x.shape), method='nearest')
        x = nn.LayerNorm()(x + e)

        skips = []
        for width in self.widths[:-1]:
            x = nn.Dense(width)(x)
            x = DenseDownBlock(width, self.block_depth)([x, skips])

        for _ in range(self.block_depth):
            x = DenseResidualBlock(self.widths[-1])(x)

        for width in reversed(self.widths[:-1]):
            x = DenseUpBlock(width, self.block_depth)([x, skips])

        x = nn.Dense(self.feature_dim, kernel_init=nn.initializers.zeros_init())(x)
        return x
    
def diffusion_schedule(diffusion_times, min_signal_rate, max_signal_rate):
    start_angle = jnp.arccos(max_signal_rate)
    end_angle = jnp.arccos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = jnp.cos(diffusion_angles)
    noise_rates = jnp.sin(diffusion_angles)
    return noise_rates, signal_rates

def create_train_state(model, rng, learning_rate, feature_dim):
    x = (jnp.ones([1, feature_dim]), jnp.ones([1, 1]))
    variables = model.init(rng, x)
    params = variables['params']
    tx = optax.adam(learning_rate)
    ts = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return ts

def train_loop(dataset, epochs, batch_size, min_signal_rate, max_signal_rate, state):
    steps_per_epoch = dataset.shape[0] // batch_size
    losses = []
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            losses_this_epoch = []
            step_key = jax.random.PRNGKey(epoch * steps_per_epoch + step)
            batch_indices_key, step_key = jax.random.split(step_key, num=2)
            batch_indices = jax.random.randint(
                batch_indices_key, shape=(batch_size,), minval=0, maxval=dataset.shape[0]
            )
            batch = dataset[batch_indices]
            loss, state = train_step(
                batch=batch, 
                min_signal_rate=min_signal_rate,
                max_signal_rate=max_signal_rate,
                state=state,
                parent_key=step_key
            )
            #print('Step', step, 'Loss:', loss)
            losses_this_epoch.append(loss)
        losses.append(sum(losses_this_epoch) / len(losses_this_epoch))
        print('Epoch', epoch, 'Loss:', losses[-1])
    return state

@jax.jit
def train_step(batch, min_signal_rate, max_signal_rate, state, parent_key):
    noise_key, diffusion_time_key = jax.random.split(parent_key, 2)
    
    def loss_fn(params):
        noises = jax.random.normal(noise_key, batch.shape)
        diffusion_times = jax.random.uniform(diffusion_time_key, (batch.shape[0], 1))
        noise_rates, signal_rates = diffusion_schedule(
            diffusion_times, min_signal_rate, max_signal_rate
        )
        noisy_batch = signal_rates * batch + noise_rates * noises

        pred_noises = state.apply_fn({'params': params}, [noisy_batch, noise_rates**2])

        loss = jnp.mean((pred_noises - noises)**2)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    grads = jax.tree_util.tree_map(jnp.nan_to_num, grads)
    state = state.apply_gradients(grads=grads)
    return loss, state

def reverse_diffusion(
    apply_fn, 
    params,
    num_images, 
    diffusion_steps, 
    feature_dim, 
    diffusion_schedule_fn,
    min_signal_rate,
    max_signal_rate,
    seed, 
    initial_noise = None,
):
    if initial_noise == None:
        initial_noise = jax.random.normal(
            jax.random.PRNGKey(seed), 
            shape=(num_images, feature_dim)
        )
    step_size = 1.0 / diffusion_steps
    
    next_noisy_images = initial_noise
    for step in range(diffusion_steps):
        noisy_images = next_noisy_images
        
        diffusion_times = jnp.ones((num_images, 1)) - step * step_size
        noise_rates, signal_rates = diffusion_schedule_fn(
            diffusion_times, min_signal_rate, max_signal_rate
        )
        pred_noises = lax.stop_gradient(
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

def load_dataset(path:str):
    dataset_directory_list = os.listdir(path)
    dataset = []
    for file in dataset_directory_list:
        if file.endswith('.npy'):
            dataset.append(jnp.ravel(jnp.load(os.path.join(path, file))))
    dataset = jnp.array(dataset, dtype=jnp.float32)
    #dataset = dataset - jnp.min(dataset)
    #dataset = dataset / jnp.max(dataset)
    #dataset = dataset * 2.0 - 1.0
    return dataset

if __name__ == '__main__':
    print('GPU:', jax.devices('gpu'))
    #jax.config.update("jax_enable_x64", True)

    epochs = 1000
    batch_size = 32
    min_signal_rate = 0.02
    max_signal_rate = 0.95
    embedding_max_frequency = 1000.0
    embedding_dims = 32
    widths = [512, 256, 128]
    block_depth = 2
    learning_rate = 5e-5

    dataset = load_dataset('data/approximation_field_small')
    feature_dim = dataset.shape[-1]
    print('Dataset shape:', dataset.shape)
    print('Dataset min:', jnp.min(dataset))
    print('Dataset max:', jnp.max(dataset))

    model = DenseDDIM(
        embedding_max_frequency=embedding_max_frequency,
        embedding_dims=embedding_dims,
        widths=widths, 
        block_depth=block_depth, 
        feature_dim=feature_dim
    )
    rng = jax.random.PRNGKey(0)
    state = create_train_state(model, rng, learning_rate, feature_dim)
    del rng
    state = train_loop(dataset, epochs, batch_size, min_signal_rate, max_signal_rate, state)
    generated_weights = reverse_diffusion(
        apply_fn=state.apply_fn, 
        params=state.params, 
        num_images=4, 
        diffusion_steps=20,
        feature_dim=feature_dim,
        diffusion_schedule_fn=diffusion_schedule,
        min_signal_rate=min_signal_rate,
        max_signal_rate=max_signal_rate,
        seed=0
    )
    generated_weights = jnp.squeeze(generated_weights)
    generated_weights = jnp.reshape(generated_weights, (generated_weights.shape[0], 48, 16))
    for i in range(generated_weights.shape[0]):
        jnp.save(f'data/generated_weights/{i}_weights.npy', generated_weights[i])
        weights_image = generated_weights[i] - jnp.min(generated_weights[i])
        weights_image = weights_image / jnp.max(weights_image)
        plt.imsave(f'data/generated_weights/{i}_weights.png', weights_image, cmap='magma')