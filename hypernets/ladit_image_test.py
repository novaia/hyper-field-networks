# The purpose of this script is to test the LADiT architecture on regular image generation.
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
import matplotlib.pyplot as plt
import os
import glob
from hypernets.common.nn import Ladit
from hypernets.common.diffusion import diffusion_schedule, ddim_sample
import argparse
from nvidia.dali import pipeline_def, fn
from nvidia.dali.plugin.jax import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali import types as dali_types
from functools import partial
import json
import wandb

def get_data_iterator(
    file_paths, batch_size, image_width, image_height, 
    context_length, token_dim, num_threads=3
):
    @pipeline_def
    def my_pipeline_def(
        files, image_width, image_height, context_length, token_dim
    ):
        image_files = fn.readers.file(
            files=files,
            read_ahead=True, 
            shuffle_after_epoch=True, 
            device='cpu'
        )[0]
        images = fn.decoders.image(
            image_files, 
            device='mixed', 
            output_type=dali_types.RGB, 
            preallocate_height_hint=image_height,
            preallocate_width_hint=image_width
        )
        scale_constant = dali_types.Constant(127.5).float32()
        shift_constant = dali_types.Constant(1.0).float32()
        normalized = (images / scale_constant) - shift_constant
        tokens = fn.reshape(
            normalized, 
            shape=[context_length, token_dim], 
            device='gpu'
        )
        return tokens

    data_pipeline = my_pipeline_def(
        files=file_paths,
        image_height=image_height,
        image_width=image_width,
        batch_size=batch_size,
        context_length=context_length,
        token_dim=token_dim,
        num_threads=num_threads, 
        device_id=0
    )
    data_iterator = DALIGenericIterator(
        pipelines=[data_pipeline], output_map=['x'], last_batch_policy=LastBatchPolicy.DROP
    )
    return data_iterator

@partial(jax.jit, static_argnames=['min_signal_rate', 'max_signal_rate', 'noise_clip'])
def train_step(state, batch, min_signal_rate, max_signal_rate, noise_clip, seed):
    noise_key, diffusion_time_key = jax.random.split(jax.random.PRNGKey(seed), 2)
    noises = jax.random.normal(noise_key, batch.shape, dtype=jnp.float32)
    noises = jnp.clip(noises, -noise_clip, noise_clip)
    diffusion_times = jax.random.uniform(diffusion_time_key, (batch.shape[0], 1, 1))
    noise_rates, signal_rates = diffusion_schedule(
        diffusion_times, min_signal_rate, max_signal_rate
    )
    noisy_batch = signal_rates * batch + noise_rates * noises

    def loss_fn(params):
        pred_noises = state.apply_fn({'params': params}, noisy_batch, noise_rates**2)
        return jnp.mean((pred_noises - noises)**2)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state

def main():
    output_directory = 'data/ladit_image_test/6'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    file_root = 'data/flattened-cifar-10'
    file_paths = glob.glob(f'{file_root}/*.jpg')
    
    batch_size = 32
    steps_per_epoch = len(file_paths) // batch_size
    num_epochs = 1000

    image_width = 32
    image_height = 32
    token_dim = 3
    context_length = image_width * image_height

    min_signal_rate = 0.02
    max_signal_rate = 0.95
    noise_clip = 3.0
    init_learning_rate = 1e-8
    learning_rate = 1e-4
    # Cifar-10 has 60k samples.
    lr_warmup_steps = (60_000//batch_size)*15
    lr_transition_steps = (60_0000//batch_size)*3
    lr_decay_rate = 0.9
    adam_b1 = 0.9
    adam_b2 = 0.9
    adam_eps = 1e-6
    weight_decay = 1e-3

    attention_dim = 2048
    num_attention_heads = 16
    embedding_dim = 128
    num_blocks = 12
    feed_forward_dim = 128
    embedding_max_frequency = 1000.0
    
    model = Ladit(
        attention_dim=attention_dim,
        num_attention_heads=num_attention_heads,
        embedding_dim=embedding_dim,
        num_bocks=num_blocks,
        feed_forward_dim=feed_forward_dim,
        token_dim=token_dim,
        embedding_max_frequency=embedding_max_frequency,
        context_length=context_length,
        normal_dtype=jnp.float32,
        quantized_dtype=jnp.bfloat16,
        remat=True
    )
    
    t = jnp.ones((batch_size, 1, 1))
    x = jnp.ones((batch_size, context_length, token_dim))
    lr_schedule = optax.warmup_exponential_decay_schedule(
        init_value=init_learning_rate, peak_value=learning_rate, warmup_steps=lr_warmup_steps, 
        transition_steps=lr_transition_steps, decay_rate=lr_decay_rate
    )
    tx = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay, b1=adam_b1, b2=adam_b2, eps=adam_eps)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, x, t)['params']
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print('Param count:', param_count)
    
    data_iterator = get_data_iterator(
        file_paths, batch_size, image_width, image_height, 
        context_length, token_dim, 3
    )
    
    wandb_config = {
        'batch_size': batch_size,
        'token_dim': token_dim,
        'image_width': image_width,
        'image_height': image_height,
        'context_length': context_length,
        'min_signal_rate': min_signal_rate,
        'max_signal_rate': max_signal_rate,
        'noise_clip': noise_clip,
        'learning_rate': learning_rate,
        'init_learning_rate': init_learning_rate,
        'lr_warmup_steps': lr_warmup_steps,
        'lr_decay_rate': lr_decay_rate,
        'lr_transition_steps': lr_transition_steps,
        'adam_b1': adam_b1,
        'adam_b2': adam_b2,
        'adam_eps': adam_eps,
        'weight_decay': weight_decay,
        'param_count': param_count,
        'attention_dim': attention_dim,
        'num_attention_heads': num_attention_heads,
        'embedding_dim': embedding_dim,
        'num_blocks': num_blocks,
        'feed_forward_dim': feed_forward_dim,
        'embedding_max_frequency': embedding_max_frequency
    }
    wandb.init(project='ladit-image-test', config=wandb_config)
    wandb_loss_accumulation_steps = 300
    steps_since_loss_report = 0
    wandb_accumulated_loss = []
    for epoch in range(num_epochs):
        losses_this_epoch = []
        for _ in range(steps_per_epoch):
            batch = next(data_iterator)['x']
            loss, state = train_step(
                state, batch, min_signal_rate, max_signal_rate, noise_clip, state.step
            )
            losses_this_epoch.append(loss)
            wandb_accumulated_loss.append(loss)
            steps_since_loss_report  += 1
            if(steps_since_loss_report == wandb_loss_accumulation_steps):
                steps_since_loss_report = 0
                wandb_average_loss = sum(wandb_accumulated_loss) / len(wandb_accumulated_loss)
                wandb_accumulated_loss = []
                wandb.log({'loss': wandb_average_loss}, step=state.step)
        
        average_loss = sum(losses_this_epoch) / len(losses_this_epoch)
        wandb.log({'average_epoch_loss': average_loss, 'current_lr': lr_schedule(state.step)}, step = state.step)

        num_samples = 10
        samples = ddim_sample(
            state=state, 
            num_samples=num_samples, 
            diffusion_steps=20, 
            diffusion_schedule=diffusion_schedule, 
            token_dim=token_dim,
            context_length=context_length,
            min_signal_rate=min_signal_rate,
            max_signal_rate=max_signal_rate,
            noise_clip=noise_clip,
            seed=epoch
        )
        samples = jnp.reshape(samples, (10, image_width, image_height, 3))
        samples = (samples + 1.0) / 2.0
        samples = jnp.clip(samples, 0.0, 1.0)
        for i in range(num_samples):
            plt.imsave(os.path.join(output_directory, f'epoch{epoch}_image{i}.png'), samples[i])

if __name__ == '__main__':
    main()
