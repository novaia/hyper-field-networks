# 1D convolutional VAE.
import jax
from jax import numpy as jnp
from flax import linen as nn
import optax
from orbax import checkpoint as ocp
from flax.training.train_state import TrainState
import datasets
import os, json
from typing import Any
from functools import partial
from hypernets.common.nn import SinusoidalEmbedding, kl_divergence
from float_tokenization import detokenize
from fields.common.flattening import unflatten_params
from fields import ngp_image
import matplotlib.pyplot as plt
import math
import wandb

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
    context_length = train[0]['params'].shape[0]
    return train, test, field_config, param_map, context_length

class ConvVae(nn.Module):
    latent_dim: int
    hidden_dims: list
    block_depth: int
    kernel_dim: int
    valid_block_dim: int
    num_valid_blocks: int
    dtype: Any

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_valid_blocks):
            x = nn.remat(nn.Conv)(
                features=self.valid_block_dim, kernel_size=(self.kernel_dim,), 
                strides=(1,), padding='SAME', dtype=self.dtype
            )(x)
            x = nn.gelu(x)
            x = nn.remat(nn.Conv)(
                features=self.valid_block_dim, kernel_size=(self.kernel_dim,), 
                strides=(1,), padding='VALID', dtype=self.dtype
            )(x)
            #print('pre', x.shape)
            x = nn.gelu(x)
        # Encoder
        for hidden_dim in self.hidden_dims:
            for _ in range(self.block_depth):
                x = nn.remat(nn.Conv)(
                    features=hidden_dim, kernel_size=(self.kernel_dim,), 
                    strides=(1,), padding='SAME', dtype=self.dtype
                )(x)
                x = nn.gelu(x)
                x = nn.remat(nn.Conv)(
                    features=hidden_dim, kernel_size=(self.kernel_dim,), 
                    strides=(1,), padding='SAME', dtype=self.dtype
                )(x)
                x = nn.gelu(x)
                # Instance normalization.
                x = nn.remat(nn.LayerNorm)(reduction_axes=[1,], feature_axes=-1)(x)
            #x = nn.avg_pool(x, window_shape=(2,), strides=(2,))
            x = nn.remat(nn.Conv)(
                features=hidden_dim, kernel_size=(self.kernel_dim,),
                strides=(2,), padding='VALID', dtype=self.dtype
            )(x)
            #print(x.shape)
        x = nn.remat(nn.Conv)(
            features=1, kernel_size=(1,), strides=(1,), 
            padding='SAME', dtype=self.dtype
        )(x)
        #print(x.shape)
        z = nn.remat(nn.DenseGeneral)(features=self.latent_dim, axis=(-1, -2))(x)

        # Decoder
        x = nn.remat(nn.DenseGeneral)(features=(x.shape[1], self.hidden_dims[-1]), axis=-1)(z)
        for hidden_dim in reversed(self.hidden_dims):
            #x = jax.image.resize(x, shape=(x.shape[0], x.shape[1]*2, x.shape[2]), method='nearest')
            x = nn.remat(nn.ConvTranspose)(
                features=hidden_dim, kernel_size=(self.kernel_dim,),
                strides=(2,), padding='VALID', dtype=self.dtype
            )(x)
            #print(x.shape)
            for _ in range(self.block_depth):
                x = nn.remat(nn.Conv)(
                    features=hidden_dim, kernel_size=(self.kernel_dim,), 
                    strides=(1,), padding='SAME', dtype=self.dtype
                )(x)
                x = nn.gelu(x)
                x = nn.remat(nn.Conv)(
                    features=hidden_dim, kernel_size=(self.kernel_dim,), 
                    strides=(1,), padding='SAME', dtype=self.dtype
                )(x)
                x = nn.gelu(x)
                # Instance normalization.
                x = nn.remat(nn.LayerNorm)(reduction_axes=[1,], feature_axes=-1)(x)
        
        for _ in range(self.num_valid_blocks+1):
            x = nn.remat(nn.ConvTranspose)(
                features=self.valid_block_dim, kernel_size=(self.kernel_dim,), 
                strides=(1,), padding='VALID', dtype=self.dtype
            )(x)
            #print('post', x.shape)
            x = nn.gelu(x)
            x = nn.remat(nn.Conv)(
                features=self.valid_block_dim, kernel_size=(self.kernel_dim,), 
                strides=(1,), padding='SAME', dtype=self.dtype
            )(x)
            x = nn.gelu(x)

        x = x[:, :-1, :]
        x = nn.Conv(
            features=1, kernel_size=(self.kernel_dim,), 
            strides=(1,), padding='SAME', dtype=jnp.float32
        )(x)
        return x

def preprocess(x):
    return jnp.expand_dims(x, axis=-1)

@jax.jit
def train_step(state, x):
    x = preprocess(x)
    
    def loss_fn(params):
        x_hat = state.apply_fn({'params': params}, x)
        return jnp.mean((x_hat - x)**2)
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state

@jax.jit
def test_step(state, x):
    x = preprocess(x)
    x_hat = state.apply_fn({'params': state.params}, x)
    return jnp.mean((x_hat - x)**2)

@jax.jit
def reconstruct(state, x):
    x = preprocess(x)
    x_hat = state.apply_fn({'params': state.params}, x)
    return x_hat

def main():
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.99'
    
    checkpoint_path = None
    experiment_number = 1
    output_path = f'data/conv_vae_output/{experiment_number}/images'
    checkpoint_output_path = f'data/conv_vae_output/{experiment_number}/checkpoints'
    dataset_path = 'data/colored-monsters-ngp-image-18k'
    split_size = 0.1
    split_seed = 0
    train_set, test_set, field_config, param_map, context_length = \
        load_dataset(dataset_path, split_size, split_seed)
    print('context_length', context_length)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(checkpoint_output_path):
        os.makedirs(checkpoint_output_path)

    num_epochs = 100
    batch_size = 6
    latent_dim = 1024
    hidden_dims = [32, 64, 128, 128]
    block_depth = 8
    kernel_dim = 4
    learning_rate = 3e-4
    weight_decay = 1e-4
    valid_block_dim = 32
    # Once we get to a context length of 17728 we can safely halve/double it.
    num_valid_blocks = 1

    wandb_config = {
        'batch_size': batch_size,
        'latent_dim': latent_dim,
        'hidden_dims': hidden_dims,
        'block_depth': block_depth,
        'kernel_dim': kernel_dim,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    }

    model = ConvVae(
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        block_depth=block_depth,
        kernel_dim=kernel_dim,
        num_valid_blocks=num_valid_blocks,
        valid_block_dim=valid_block_dim,
        dtype=jnp.bfloat16
    )
    x = jnp.ones((batch_size, context_length, 1), dtype=jnp.float32)
    params_key = jax.random.PRNGKey(91)
    params = jax.jit(model.init)(params_key, x=x)['params']
    opt = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)

    num_train_samples = len(train_set)
    train_steps = num_train_samples // batch_size
    print('num_train_samples', num_train_samples)
    print('train_steps', train_steps)
    num_test_samples = len(test_set)
    test_steps = num_test_samples // batch_size
    print('num_test_samples', num_test_samples)
    print('test_steps', test_steps)
    
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=True))
    if checkpoint_path is not None:
        print(f'loading checkpoint {checkpoint_path}')
        state = checkpointer.restore(checkpoint_path, item=state)

    field_model = ngp_image.create_model_from_config(field_config)
    field_state = ngp_image.create_train_state(field_model, 3e-4, jax.random.PRNGKey(0))

    wandb.init(project='conv-vae', config=wandb_config)
    test_sample = jnp.expand_dims(test_set[0]['params'], axis=0)
    print('test sample shape', test_sample.shape)
    for epoch in range(num_epochs):
        train_set = train_set.shuffle(seed=epoch)
        train_iterator = train_set.iter(batch_size)
        losses_this_epoch = []
        for step in range(train_steps):
            batch = next(train_iterator)['params']
            loss, state = train_step(state, batch)
            losses_this_epoch.append(loss)
        average_loss = sum(losses_this_epoch) / len(losses_this_epoch)
        print(f'epoch {epoch}, loss {average_loss}')
        wandb.log({'loss': average_loss}, step=state.step)
        
        test_set = test_set.shuffle(seed=epoch)
        test_iterator = test_set.iter(batch_size)
        losses_this_test = []
        for step in range(test_steps):
            batch = next(test_iterator)['params']
            loss = test_step(state, batch)
            losses_this_test.append(loss)
        average_test_loss = sum(losses_this_test) / len(losses_this_test)
        print(f'epoch {epoch}, test_loss {average_test_loss}')
        wandb.log({'test_loss': average_test_loss}, step=state.step)
        current_checkpoint_path = os.path.join(
            os.path.abspath(checkpoint_output_path), f'step{state.step}'
        )
        checkpointer.save(current_checkpoint_path, state, force=True)
        print(f'saved checkpoint {current_checkpoint_path}')
        flat_params = jnp.squeeze(reconstruct(state, test_sample)[0], axis=-1)
        params = unflatten_params(jnp.array(flat_params, dtype=jnp.float32), param_map)
        field_state = field_state.replace(params=params)
        field_render = ngp_image.render_image(
            field_state, field_config['image_height'], field_config['image_width'], field_config['channels']
        )
        field_render = jax.device_put(field_render, jax.devices('cpu')[0])
        plt.imsave(os.path.join(output_path, f'{epoch}.png'), field_render)

if __name__ == '__main__':
    main()
