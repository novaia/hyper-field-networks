import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.99'

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import matplotlib.pyplot as plt
from hypernets.common.nn import SinusoidalEmbedding, VanillaTransformer
from hypernets.common.diffusion import diffusion_schedule, reverse_diffusion
from dataclasses import dataclass

def load_dataset(dataset_path, test_size, split_seed):
    field_config = None
    with open(os.path.join(dataset_path, 'field_config.json'), 'r') as f:
        field_config = json.load(f)
    assert field_config is not None
    param_map = None
    with open(os.path.join(dataset_path, 'param_map.json'), 'r') as f:
        param_map = json.load(f)
    assert param_map is not None
    
    parquet_dir = os.path.join(dataset_path, 'data')
    parquet_paths = [
        os.path.join(parquet_dir, p) 
        for p in os.listdir(parquet_dir) if p.endswith('.parquet')
    ]
    num_parquet_files = len(parquet_paths)
    assert num_parquet_files > 0
    print(f'Found {num_parquet_files} parquet file(s) in dataset directory')

    dataset = datasets.load_dataset('parquet', data_files=parquet_paths)
    train, test = dataset['train'].train_test_split(test_size=test_size, seed=split_seed).values()
    device = str(jax.devices('gpu')[0])
    train = train.with_format('jax', device=device)
    test = test.with_format('jax', device=device)
    context_length = train[0]['mlp_latent'].shape[0] + train[0]['hash_latent'].shape[0]
    return train, test, field_config, param_map, context_length

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

@dataclass
class SplitLatentDiffusionConfig:
    min_signal_rate: float
    max_signal_rate: float
    num_blocks: int
    feed_forward_dim: int
    attention_dim: int
    attention_heads: int
    token_dim: int
    embedded_token_dim: int
    embedding_max_frequency: float
    context_length: int
    num_epochs: int
    batch_size: int
    test_split_size: float
    split_seed: int
    model_seed: int

    def __init__(self, config_dict) -> None:
        raise NotImplementedError()

def main():
    config_path = '...'
    dataset_path = '...'

    with open(config_path, 'r') as f:
        main_config_dict = json.load(f)
        main_config = SplitLatentDiffusionConfig(main_config_dict)

    train_set, test_set, field_config, param_map, context_length = load_dataset(
        dataset_path=dataset_path, 
        test_size=main_config.test_split_size, 
        split_seed=main_config.split_seed
    )
    main_config.context_length = context_length

    model = HyperDiffusion(
        num_blocks=main_config.num_blocks,
        feed_forward_dim=main_config.feed_forward_dim,
        attention_dim=main_config.attention_dim,
        attention_heads=main_config.attention_heads,
        token_dim=main_config.token_dim,
        embedded_token_dim=main_config.embedded_token_dim,
        embedding_max_frequency=main_config.embedding_max_frequency,
        context_length=main_config.context_length
    )
    x = jnp.ones((main_config.batch_size, main_config.context_length), dtype=jnp.float32)
    params_key = jax.random.PRNGKey(main_config.model_seed)
    params = model.init(params_key, x=x)['params']
    opt = optax.adamw(
        learning_rate=main_config.learning_rate, weight_decay=main_config.weight_decay
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)

    num_batches = len(train_set) // main_config.batch_size
    for epoch in range(main_config.num_epochs):
        train_set = train_set.shuffle(seed=epoch)
        train_iterator = train_set.iter(main_config.batch_size)
        loss_this_train = []
        for _ in range(num_batches):
            batch = next(train_iterator)
            batch = jnp.concatenate((batch['mlp_latents'], batch['hash_latents']), axis=-1)
            loss, state = train_step(
                batch=batch, 
                min_signal_rate=main_config.min_signal_rate, 
                max_signal_rate=main_config.max_signal_rate,
                state=state,
                parent_key=jax.random.PRNGKey(state.step)
            )
            losses_this_train.append(loss)
        average_loss = sum(losses_this_epoch) / len(losses_this_epoch)
        print(f'epoch {epoch}, loss {average_loss}')
        wandb.log({'loss': average_loss}, step=state.step)
