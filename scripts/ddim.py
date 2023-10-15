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

class ResidualBlock(nn.Module):
    width: int

    @nn.compact
    def __call__(self, x):
        input_width = x.shape[-1]
        if input_width == self.width:
            residual = x
        else:
            residual = nn.Conv(self.width, kernel_size=(1, 1))(x)
        x = nn.Conv(self.width, kernel_size=(3, 3))(x)
        x = nn.GroupNorm()(x)
        x = nn.activation.swish(x)
        x = nn.Conv(self.width, kernel_size=(3, 3))(x)
        x = nn.GroupNorm()(x)
        x = x + residual
        return x

class DownBlock(nn.Module):
    width: int
    block_depth: int

    @nn.compact
    def __call__(self, x):
        x, skips = x

        for _ in range(self.block_depth):
            x = ResidualBlock(self.width)(x)
            skips.append(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x

class UpBlock(nn.Module):
    width: int
    block_depth: int

    @nn.compact
    def __call__(self, x):
        x, skips = x

        upsample_shape = (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3])
        x = jax.image.resize(x, upsample_shape, method='bilinear')

        for _ in range(self.block_depth):
            x = jnp.concatenate([x, skips.pop()], axis=-1)
            x = ResidualBlock(self.width)(x)
        return x

class DDIM(nn.Module):
    embedding_max_frequency: float
    embedding_dims: int
    widths: list
    block_depth: int
    output_channels: int

    @nn.compact
    def __call__(self, x):
        x, noise_variances = x

        e = sinusoidal_embedding(
            noise_variances, 
            self.embedding_max_frequency, 
            self.embedding_dims
        )
        e = jax.image.resize(
            e, shape=(x.shape[0], x.shape[1], x.shape[2], 1), method='nearest'
        )
        
        x = nn.Conv(self.widths[0], kernel_size=(1, 1))(x)
        x = jnp.concatenate([x, e], axis=-1)

        skips = []
        for width in self.widths[:-1]:
            x = DownBlock(width, self.block_depth)([x, skips])

        for _ in range(self.block_depth):
            x = ResidualBlock(self.widths[-1])(x)

        for width in reversed(self.widths[:-1]):
            x = UpBlock(width, self.block_depth)([x, skips])

        x = nn.Conv(
            self.output_channels, 
            kernel_size=(1, 1), 
            kernel_init=nn.initializers.zeros_init()
        )(x)
        return x
    
def diffusion_schedule(diffusion_times, min_signal_rate, max_signal_rate):
    start_angle = jnp.arccos(max_signal_rate)
    end_angle = jnp.arccos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = jnp.cos(diffusion_angles)
    noise_rates = jnp.sin(diffusion_angles)
    return noise_rates, signal_rates

def create_train_state(model, rng, learning_rate, input_height, input_width):
    x = (jnp.ones([1, input_height, input_width, 1]), jnp.ones([1, 1, 1, 1]))
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
        diffusion_times = jax.random.uniform(diffusion_time_key, (batch.shape[0], 1, 1, 1))
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
    image_width, 
    image_height, 
    channels,
    diffusion_schedule_fn,
    min_signal_rate,
    max_signal_rate,
    seed, 
    initial_noise = None,
):
    if initial_noise == None:
        initial_noise = jax.random.normal(
            jax.random.PRNGKey(seed), 
            shape=(num_images, image_height, image_width, channels)
        )
    step_size = 1.0 / diffusion_steps
    
    next_noisy_images = initial_noise
    for step in range(diffusion_steps):
        noisy_images = next_noisy_images
        
        diffusion_times = jnp.ones((num_images, 1, 1, 1)) - step * step_size
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
            dataset.append(jnp.load(os.path.join(path, file)))
    dataset = jnp.array(dataset, dtype=jnp.float32)
    dataset = dataset - jnp.min(dataset)
    dataset = dataset / jnp.max(dataset)
    dataset = dataset * 2.0 - 1.0
    dataset = jnp.expand_dims(dataset, axis=-1)
    padding = jnp.zeros((dataset.shape[0], 3, dataset.shape[2], dataset.shape[3]))
    dataset = jnp.concatenate([dataset, padding], axis=1)
    return dataset

if __name__ == '__main__':
    print('GPU:', jax.devices('gpu'))
    #jax.config.update("jax_enable_x64", True)

    epochs = 10
    batch_size = 32
    min_signal_rate = 0.02
    max_signal_rate = 0.95
    embedding_max_frequency = 1000.0
    embedding_dims = 32
    widths = [32, 32]
    block_depth = 2
    output_channels = 1
    learning_rate = 1e-5

    dataset = load_dataset('data/approximation_field')
    input_height = dataset.shape[1]
    input_width = dataset.shape[2]
    print('Dataset shape:', dataset.shape)
    print('Dataset min:', jnp.min(dataset))
    print('Dataset max:', jnp.max(dataset))

    model = DDIM(
        embedding_max_frequency=embedding_max_frequency,
        embedding_dims=embedding_dims,
        widths=widths, 
        block_depth=block_depth, 
        output_channels=output_channels
    )
    rng = jax.random.PRNGKey(0)
    state = create_train_state(model, rng, learning_rate, input_height, input_width)
    del rng
    state = train_loop(dataset, epochs, batch_size, min_signal_rate, max_signal_rate, state)
    generated_weights = reverse_diffusion(
        apply_fn=state.apply_fn, 
        params=state.params, 
        num_images=4, 
        diffusion_steps=20,
        image_width=input_width, 
        image_height=input_height, 
        channels=output_channels, 
        diffusion_schedule_fn=diffusion_schedule,
        min_signal_rate=min_signal_rate,
        max_signal_rate=max_signal_rate,
        seed=0
    )
    generated_weights = jnp.squeeze(generated_weights, axis=-1)
    for i in range(generated_weights.shape[0]):
        jnp.save(f'data/generated_weights/{i}_weights.npy', generated_weights[i])
        plt.imsave(f'data/generated_weights/{i}_weights.png', generated_weights[i], cmap='magma')