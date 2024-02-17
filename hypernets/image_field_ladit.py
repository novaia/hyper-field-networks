# LADiT = Linear Attention Diffusion Transformer.
# The purpose of this script is to train a LADiT model on neural image fields.
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
    
@partial(jax.jit, static_argnames=['noise_clip', 'min_signal_rate', 'max_signal_rate'])
def train_step(
    state:TrainState, seed:int, batch:jax.Array, 
    noise_clip:float, min_signal_rate:float, max_signal_rate:float
):
    # Add channel dim to batch.
    batch = jnp.reshape(batch, (*batch.shape, 1))
    noise_key, diffusion_time_key = jax.random.split(jax.random.PRNGKey(seed))
    diffusion_times = jax.random.uniform(diffusion_time_key, (batch.shape[0], 1, 1))
    noise_rates, signal_rates = diffusion_schedule(diffusion_times, min_signal_rate, max_signal_rate)
    noise = jax.random.normal(noise_key, batch.shape)
    # Clipping the noise prevents NaN gradients.
    noise = jnp.clip(noises, -noise_clip, noise_clip)
    noisy_batch = batch * signal_rates + noise * noise_rates
    x = (noisy_batch, noise_rates**2)

    def loss_fn(params):
        pred_noise = state.apply_fn({'params': params}, x)
        return jnp.mean((pred_noise - noise)**2)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return loss, state

def get_data_iterator(file_paths, batch_size, num_threads=3):
    @pipeline_def
    def my_pipeline_def():
        data = fn.readers.numpy(
            device='gpu', 
            files=file_paths,
            shuffle_after_epoch=True,
            name='r'
        )
        return data
    my_pipeline = my_pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=0)
    iterator = DALIGenericIterator(pipelines=[my_pipeline], output_map=['x'], reader_name='r')
    return iterator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_epoch', type=int, default=-1)
    parser.add_argument('--render_only', action='store_true')
    parser.add_argument('--train_epochs', type=int, default=1000)
    args = parser.parse_args()

    with open('configs/image_field_ladit.json', 'r') as f:
        config = json.load(f)

    config['dataset'] = 'danbooru_ngp_images'
    experiment_id = 12
    experiment_name = f'experiment_{experiment_id}'
    project_name = 'image_field_latid_danbooru'
    project_path = os.path.join('data', project_name)
    checkpoint_path = os.path.join(project_path, 'checkpoints', experiment_name)
    image_path = os.path.join(project_path, 'images', experiment_name)
    dataset_path = 'data/danbooru_ngp_images_flattened'
    config_path = 'configs/image_field.json'
    weight_map_path = os.path.join(dataset_path, 'weight_map.json')
    dataset_file_paths = glob.glob(f'{dataset_path}/*/*.npy')

    if not os.path.exists(project_path): os.makedirs(project_path)
    if not os.path.exists(checkpoint_path): os.makedirs(checkpoint_path)
    if not os.path.exists(image_path): os.makedirs(image_path)

    with open(os.path.join(project_path, f'{experiment_name}.json'), 'w') as f:
        json.dump(config, f, indent=4)
    wandb.init(project="image-field-ladit", config=config)

    image_width = 128
    image_height = 128
    num_render_only_images = 5
    num_train_preview_images = 5
    epochs_between_checkpoints = 5
    # TODO: shape index will need to be -2 if DALI loads samples as (n, l, c)
    # instead of (n, l), where n is batch size, l is context length, and c is channel dim.
    # For now I'm assuming it's the latter.
    context_length = jnp.load(dataset_file_paths[0]).shape[-1]
    # Token dim is 1 because each parameter of the image field is 1 token.
    token_dim = 1
    min_signal_rate = config['min_signal_rate']
    max_signal_rate = config['max_signal_rate']
    noise_clip = config['noise_clip']

    model = Ladit(
        attention_dim=config['attention_dim'],
        num_attention_heads=config['num_attention_heads'],
        embedding_dim=config['embedding_dim'],
        token_dim=token_dim,
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
    x = (jnp.ones((1, context_length, 1)), jnp.ones((1, 1, 1)))
    params = model.init(rng, x)['params']
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=True))
    if args.checkpoint_epoch > -1:
        checkpoint_name = f'epoch_{args.checkpoint_epoch}'
        state = checkpointer.restore(os.path.join(checkpoint_path, checkpoint_name), item=state)
        print(f'Loaded checkpoint {checkpoint_name}')
    if args.render_only:
        print('Render only mode')
        generated_weights = reverse_diffusion(
            state.apply_fn, 
            state.params, 
            num_images=num_render_only_images, 
            diffusion_steps=config['diffusion_steps'], 
            context_length=context_length,
            token_dim=token_dim,
            diffusion_schedule_fn=diffusion_schedule,
            min_signal_rate=min_signal_rate,
            max_signal_rate=max_signal_rate,
            seed=2
        )
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

    for epoch in range(args.checkpoint_epoch+1, args.checkpoint_epoch+args.train_epochs+1):
        losses_this_epoch = []
        for step, batch in enumerate(data_iterator):
            loss, state = train_step(state, step, batch, noise_clip, min_signal_rate, max_signal_rate)
            wandb.log({'loss': loss}, step=state.step)
            losses_this_epoch.append(loss)

        average_loss = sum(losses_this_epoch) / len(losses_this_epoch)
        print('Epoch:', epoch, 'Loss:', average_loss)
        
        if epoch % epochs_between_checkpoints != 0 or epoch == 0:
            continue

        checkpointer.save(os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch}'), state)
        generated_weights = reverse_diffusion(
            state.apply_fn, 
            state.params, 
            num_images=num_train_preview_images, 
            diffusion_steps=config['diffusion_steps'], 
            context_length=context_length,
            token_dim=token_dim,
            diffusion_schedule_fn=diffusion_schedule,
            min_signal_rate=config['min_signal_rate'],
            max_signal_rate=config['max_signal_rate'],
            seed=0
        )
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
