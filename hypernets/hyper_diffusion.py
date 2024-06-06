import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import os
import matplotlib.pyplot as plt
from hypernets.common.nn import SinusoidalEmbedding, VanillaTransformer
from hypernets.common.diffusion import diffusion_schedule, reverse_diffusion

class HyperDiffusion(nn.Module):
    num_blocks: int
    feed_forward_dim: int
    attention_dim: int
    attention_heads: int
    token_dim: int
    embedded_token_dim: int
    embedding_max_frequency: float
    context_length: int

    @nn.compact
    def __call__(self, x):
        x, noise_variances = x
        e = SinusoidalEmbedding(self.embedding_max_frequency)(noise_variances)
        x = jnp.concatenate([x, e], axis=-2)
        x = nn.Dense(features=self.embedded_token_dim)(x)
        positions = jnp.arange(self.context_length+1)
        e = nn.Embed(
            num_embeddings=self.context_length+1, 
            features=self.embedded_token_dim
        )(positions)
        x = x + e

        x = VanillaTransformer(
            num_heads=self.attention_heads,
            num_blocks=self.num_blocks,
            attention_dim=self.attention_dim,
            residual_dim=self.embedded_token_dim,
            feed_forward_dim=self.feed_forward_dim
        )(x)
        x = nn.Dense(features=self.token_dim)(x)
        x = x[:, :-1, :] # Remove embedded noise variances token.
        return x

def create_train_state(model, rng, learning_rate, context_length, token_dim, steps_per_epoch):
    x = (jnp.ones([1, context_length, token_dim]), jnp.ones([1, 1, 1]))
    variables = model.init(rng, x)
    params = variables['params']
    learning_rate_schedule = optax.exponential_decay(
        learning_rate, transition_begin=80*steps_per_epoch, 
        transition_steps=100*steps_per_epoch, decay_rate=0.8
    )
    tx = optax.adam(learning_rate_schedule)
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
        diffusion_times = jax.random.uniform(diffusion_time_key, (batch.shape[0], 1, 1))
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

def load_dataset(path:str):
    dataset_directory_list = os.listdir(path)
    dataset = []
    for file in dataset_directory_list:
        if file.endswith('.npy'):
            dataset.append(jnp.load(os.path.join(path, file)))
    dataset = jnp.array(dataset, dtype=jnp.float32)
    return dataset

def main():
    print('GPU:', jax.devices('gpu'))

    epochs = 1000
    batch_size = 32
    min_signal_rate = 0.02
    max_signal_rate = 0.95
    embedding_max_frequency = 1000.0
    num_blocks = 4
    feed_forward_dim = 512
    attention_dim = 512
    embedded_token_dim = 512
    attention_heads = 8
    learning_rate = 5e-5

    dataset = load_dataset('data/approximation_field_small')
    token_dim = dataset.shape[-1]
    context_length = dataset.shape[-2]
    steps_per_epoch = dataset.shape[0] // batch_size
    print('Dataset shape:', dataset.shape)
    print('Dataset min:', jnp.min(dataset))
    print('Dataset max:', jnp.max(dataset))

    model = HyperDiffusion(
        num_blocks=num_blocks,
        feed_forward_dim=feed_forward_dim,
        attention_dim=attention_dim,
        attention_heads=attention_heads,
        token_dim=token_dim,
        embedded_token_dim=embedded_token_dim,
        embedding_max_frequency=embedding_max_frequency,
        context_length=context_length
    )
    rng = jax.random.PRNGKey(0)
    state = create_train_state(
        model, rng, learning_rate, context_length, token_dim, steps_per_epoch
    )
    del rng
    state = train_loop(dataset, epochs, batch_size, min_signal_rate, max_signal_rate, state)
    generated_weights = reverse_diffusion(
        apply_fn=state.apply_fn, 
        params=state.params, 
        num_images=4, 
        diffusion_steps=20,
        context_length=context_length,
        token_dim=token_dim,
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

if __name__ == '__main__':
    main()
