import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import List, Callable
import optax
from nvidia.dali import pipeline_def, fn
from nvidia.dali.plugin.jax import DALIGenericIterator
from flax.training.train_state import TrainState
from functools import partial
import matplotlib.pyplot as plt
import os

class ResidualBlock(nn.Module):
    width: int
    activation: Callable

    @nn.compact
    def __call__(self, x):
        input_width = x.shape[-1]
        if input_width == self.width:
            residual = x
        else:
            residual = nn.Conv(self.width, kernel_size=(1, 1))(x)
        x = nn.Conv(self.width, kernel_size=(3, 3))(x)
        x = nn.GroupNorm()(x)
        x = self.activation(x)
        x = nn.Conv(self.width, kernel_size=(3, 3))(x)
        x = nn.GroupNorm()(x)
        x = self.activation(x)
        x = x + residual
        return x

class Generator(nn.Module):
    widths: List[int]
    block_depth: int

    @nn.compact
    def __call__(self, x):
        for width in self.widths:
            upsample_shape = (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3])
            x = jax.image.resize(x, upsample_shape, method='bilinear')
            for _ in range(self.block_depth):
                x = ResidualBlock(width, activation=nn.relu)(x)
        x = nn.Conv(3, kernel_size=(1, 1))(x)
        x = nn.sigmoid(x)
        return x
    
class Discriminator(nn.Module):
    widths: List[int]
    block_depth: int

    @nn.compact
    def __call__(self, x):
        for width in self.widths:
            for _ in range(self.block_depth):
                x = ResidualBlock(width, activation=nn.leaky_relu)(x)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(1, kernel_size=(1, 1))(x)
        x = jnp.squeeze(x, axis=-1)
        x = jnp.mean(x, axis=(-1, -2))
        return x

@partial(jax.jit, static_argnames=['latent_width', 'latent_height'])
def train_step(
    generator_state, discriminator_state, real_data, latent_width, latent_height, key
):
    latent_noise = jax.random.normal(key, (real_data.shape[0], latent_height, latent_width, 1))
    real_labels = jnp.ones((real_data.shape[0]))
    fake_labels = jnp.zeros((real_data.shape[0]))
    labels = jnp.concatenate([real_labels, fake_labels], axis=0)

    def generator_loss_fn(params):
        fake_data = generator_state.apply_fn(
            {'params': params}, latent_noise
        )
        real_and_fake_data = jnp.concatenate([real_data, fake_data], axis=0)
        logits = discriminator_state.apply_fn(
            {'params': discriminator_state.params}, real_and_fake_data
        )
        loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))
        return loss
    generator_grad_fn = jax.value_and_grad(generator_loss_fn)
    generator_loss, generator_grads = generator_grad_fn(generator_state.params)
    generator_state = generator_state.apply_gradients(grads=generator_grads)

    def discriminator_loss_fn(params):
        fake_data = generator_state.apply_fn(
            {'params': generator_state.params}, latent_noise
        )
        real_and_fake_data = jnp.concatenate([real_data, fake_data], axis=0)
        logits = discriminator_state.apply_fn(
            {'params': params}, real_and_fake_data
        )
        loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))
        return loss
    discriminator_grad_fn = jax.value_and_grad(discriminator_loss_fn)
    discriminator_loss, discriminator_grads = discriminator_grad_fn(discriminator_state.params)
    discriminator_state = discriminator_state.apply_gradients(grads=discriminator_grads)

    return (generator_loss, discriminator_loss), generator_state, discriminator_state

def get_data_iterator(dataset_path, batch_size, num_threads=3):
    @pipeline_def
    def my_pipeline_def():
        data = fn.readers.numpy(
            device='cpu', 
            file_root=dataset_path, 
            file_filter='*.npy', 
            shuffle_after_epoch=True,
            name='r'
        )
        return data
    my_pipeline = my_pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=0)
    iterator = DALIGenericIterator(pipelines=[my_pipeline], output_map=['x'], reader_name='r')
    return iterator

def main():
    latent_height = 8
    latent_width = 8
    batch_size = 32
    num_epochs = 10
    dataset_path = 'data/cifar10_numpy'
    save_path = 'data/gan_generations'
    generator = Generator(widths=[32, 32], block_depth=2)
    discriminator = Discriminator(widths=[32, 32], block_depth=2)

    learning_rate = 3e-4
    optimizer = optax.adam(learning_rate)

    key = jax.random.PRNGKey(0)
    latent_input = jnp.ones((1, latent_height, latent_width, 1))
    generator_output, generator_variables = generator.init_with_output(key, latent_input)
    generator_params = generator_variables['params']
    generator_state = TrainState.create(
        apply_fn=generator.apply, params=generator_params, tx=optimizer
    )
    print('Generator Output:', generator_output.shape)

    discriminator_output, discriminator_variables = discriminator.init_with_output(
        key, generator_output
    )
    discriminator_params = discriminator_variables['params']
    discriminator_state = TrainState.create(
        apply_fn=discriminator.apply, params=discriminator_params, tx=optimizer
    )
    print('Discriminator Output:', discriminator_output.shape)

    data_iterator = get_data_iterator(dataset_path, batch_size)
    for epoch in range(num_epochs):
        for batch in data_iterator:
            batch = batch['x']
            losses, generator_state, discriminator_state = train_step(
                discriminator_state=discriminator_state,
                generator_state=generator_state,
                real_data=batch,
                latent_width=latent_width,
                latent_height=latent_height,
                key=jax.random.PRNGKey(generator_state.step)
            )
            generator_loss, disciminator_loss = losses
            print('Gen Loss:', generator_loss, 'Disc Loss:', disciminator_loss)
        print(f'Finished epoch {epoch}')
        latent_shape = (4, latent_height, latent_width, 1)
        latent_key = jax.random.PRNGKey(generator_state.step)
        latent_input = jax.random.normal(latent_key, latent_shape)
        generated_images = generator_state.apply_fn(
            {'params': generator_state.params}, latent_input
        )
        for i in range(generated_images.shape[0]):
            plt.imsave(os.path.join(save_path, f'epoch{epoch}_image{i}.png'), generated_images[i])


if __name__ == '__main__':
    main()
    
