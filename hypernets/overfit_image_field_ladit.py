# LADiT = Linear Attention Diffusion Transformer.
# The purpose of this script is to train a LADiT model on vanilla image fields.
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
import matplotlib.pyplot as plt
import os
from hypernets.common.nn import Ladit
from hypernets.common.diffusion import diffusion_schedule, reverse_diffusion
from hypernets.common.rendering import unpack_and_render_image_field
import argparse
import orbax.checkpoint as ocp
from nvidia.dali import pipeline_def, fn
from nvidia.dali.plugin.jax import DALIGenericIterator
from functools import partial
import json
import wandb
    
@jax.jit
def train_step(state:TrainState, key:int, batch:jax.Array):
    noise_key, diffusion_time_key = jax.random.split(key)

    def loss_fn(params):
        diffusion_times = jax.random.uniform(diffusion_time_key, (batch.shape[0], 1, 1))
        noise_rates, signal_rates = diffusion_schedule(diffusion_times, 0.02, 0.95)
        noise = jax.random.normal(noise_key, batch.shape)
        noisy_batch = batch * signal_rates + noise * noise_rates
        x = (noisy_batch, noise_rates**2)
        pred_noise = state.apply_fn({'params': params}, x)
        return jnp.mean((pred_noise - noise)**2)
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    grad = jax.tree_util.tree_map(jnp.nan_to_num, grad)
    state = state.apply_gradients(grads=grad)
    return loss, state

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

def tokenize_batch(token_dim, batch):
    context_length = int(jnp.ceil((batch.shape[1] * batch.shape[2]) / token_dim))
    batch = jnp.resize(batch, (batch.shape[0], context_length, token_dim))
    return batch

def detokenize_batch(original_height, original_width, batch):
    batch = jnp.resize(batch, (batch.shape[0], original_height, original_width))
    return batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_epoch', type=int, default=-1)
    parser.add_argument('--render_only', type=bool, default=False)
    parser.add_argument('--train_epochs', type=int, default=1000)
    args = parser.parse_args()

    with open('configs/image_field_ladit.json', 'r') as f:
        config = json.load(f)

    config['dataset'] = 'cifar10_single_batch'
    experiment_id = 0
    experiment_name = f'experiment_{experiment_id}'
    project_name = 'overfit_image_field_latid_cifar10'
    project_path = os.path.join('data', project_name)
    image_path = os.path.join(project_path, 'images', experiment_name)
    dataset_path = 'data/ngp_images/cifar10_single_batch'
    config_path = 'configs/image_field.json'
    weight_map_path = os.path.join(dataset_path, 'weight_map.json')

    if not os.path.exists(project_path): os.makedirs(project_path)
    if not os.path.exists(image_path): os.makedirs(image_path)

    with open(os.path.join(project_path, f'{experiment_name}.json'), 'w') as f:
        json.dump(config, f, indent=4)
    wandb.init(project="overfit-image-field-ladit", config=config)

    image_width = 32
    image_height = 32
    num_render_only_images = 6
    num_train_preview_images = 5
    epochs_between_generations = 5

    data_iterator = get_data_iterator(dataset_path, config['batch_size'])
    dummy_batch = data_iterator.next()['x']
    original_batch_width = dummy_batch.shape[-1]
    original_batch_height = dummy_batch.shape[-2]
    tokenize_batch_fn = partial(tokenize_batch, token_dim=config['token_dim'])
    detokenize_batch_fn = partial(
        detokenize_batch, 
        original_height=original_batch_height, 
        original_width=original_batch_width
    )
    dummy_batch = tokenize_batch_fn(batch=dummy_batch)
    context_length = dummy_batch.shape[-2]
    print('Batch shape', dummy_batch.shape)

    model = Ladit(
        attention_dim=config['attention_dim'],
        num_attention_heads=config['num_attention_heads'],
        token_dim=config['token_dim'],
        embedding_dim=config['embedding_dim'],
        num_bocks=config['num_blocks'],
        feed_forward_dim=config['feed_forward_dim'],
        embedding_max_frequency=config['embedding_max_frequency'],
        context_length=context_length,
        normal_dtype=jnp.dtype(config['normal_dtype']),
        quantized_dtype=jnp.dtype(config['quantized_dtype']),
        remat=config['remat']
    )

    tx = optax.adam(config['learning_rate'])
    rng = jax.random.PRNGKey(0)
    x = (jnp.ones((1, context_length, config['token_dim'])), jnp.ones((1, 1, 1)))
    params = model.init(rng, x)['params']
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    if args.render_only:
        print('Render only mode')
        generated_weights = reverse_diffusion(
            state.apply_fn, 
            state.params, 
            num_images=num_render_only_images, 
            diffusion_steps=config['diffusion_steps'], 
            context_length=context_length,
            token_dim=config['token_dim'],
            diffusion_schedule_fn=diffusion_schedule,
            min_signal_rate=config['min_signal_rate'],
            max_signal_rate=config['max_signal_rate'],
            seed=2
        )
        generated_weights = detokenize_batch_fn(batch=generated_weights)
        print('Generated weights max:', jnp.max(generated_weights))
        print('Generated weights min:', jnp.min(generated_weights))
        for i in range(num_render_only_images):
            rendered_image = unpack_and_render_image_field(
                config_path=config_path,
                weight_map_path=weight_map_path,
                packed_weights=generated_weights[i],
                image_width=image_width,
                image_height=image_height
            )
            plt.imsave(os.path.join(image_path, f'render_only_{i}.png'), rendered_image)
        exit(0)

    gpu = jax.devices('gpu')[0]
    for epoch in range(args.checkpoint_epoch+1, args.checkpoint_epoch+args.train_epochs+1):
        losses_this_epoch = []
        for i in range(1000):
            for step, batch in enumerate(data_iterator):
                step_key = jax.random.PRNGKey(state.step)
                batch = jax.device_put(batch['x'], gpu)
                batch = tokenize_batch_fn(batch=batch)
                loss, state = train_step(state, step_key, batch)
                wandb.log({'loss': loss}, step=state.step)
                losses_this_epoch.append(loss)

        average_loss = sum(losses_this_epoch) / len(losses_this_epoch)
        print('Epoch:', epoch, 'Loss:', average_loss)
        
        if epoch % epochs_between_generations != 0 or epoch == 0:
            continue

        generated_weights = reverse_diffusion(
            state.apply_fn, 
            state.params, 
            num_images=num_train_preview_images, 
            diffusion_steps=config['diffusion_steps'], 
            context_length=context_length,
            token_dim=config['token_dim'],
            diffusion_schedule_fn=diffusion_schedule,
            min_signal_rate=config['min_signal_rate'],
            max_signal_rate=config['max_signal_rate'],
            seed=0
        )
        generated_weights = detokenize_batch_fn(batch=generated_weights)
        print('Generated weights max:', jnp.max(generated_weights))
        print('Generated weights min:', jnp.min(generated_weights))
        
        for i in range(num_train_preview_images):
            rendered_image = unpack_and_render_image_field(
                config_path=config_path,
                weight_map_path=weight_map_path,
                packed_weights=generated_weights[i],
                image_width=image_width,
                image_height=image_height
            )
            image_save_path = os.path.join(image_path, f'image_{i}_epoch_{epoch}.png')
            plt.imsave(image_save_path, rendered_image)
    print('Finished training')

if __name__ == '__main__':
    main()