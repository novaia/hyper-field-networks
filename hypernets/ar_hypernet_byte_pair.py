import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.99'

import jax
from jax import numpy as jnp
from flax import linen as nn
import optax
from flax.training.train_state import TrainState
import datasets
import json
from typing import Any
from functools import partial
import fp_tokenization as fpt
from fields.common.flattening import unflatten_params
from fields import ngp_image
import matplotlib.pyplot as plt
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
    context_length = train[0]['tokens'].shape[0] // 2 # 1 mantissa token and 1 exponent token per timestep.
    return train, test, field_config, param_map, context_length

class ArHypernet(nn.Module):
    vocab_size: int
    context_length: int
    hidden_dim: int
    ff_dim: int
    num_attention_heads: int
    num_blocks: int
    dtype: Any
    dropout_rate: float

    @nn.compact
    def __call__(self, mantissa_tokens, exponent_tokens, training):
        attention_mask = nn.make_causal_mask(tokens)
        # Add 1 to vocab size to allow for start token.
        # Start token should never be predicted so it is not added to the base vocab.
        mantissa_emb = nn.Embed(num_embeddings=self.vocab_size+1, features=self.hidden_dim//2)(mantissa_tokens)
        exponent_emb = nn.Embed(num_embeddings=self.vocab_size+1, features=self.hidden_dim//2)(exponent_tokens)
        token_emb = jnp.concatenate([mantissa_emb, exponent_emb], axis=-1)
        positions = jnp.arange(self.context_length, dtype=jnp.uint32)
        pos_emb = nn.Embed(num_embeddings=self.context_length, features=self.hidden_dim)(positions)
        x = token_emb + pos_emb
        
        for _ in range(self.num_blocks):
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.remat(nn.MultiHeadDotProductAttention)(
                num_heads=self.num_attention_heads,
                qkv_features=self.hidden_dim,
                out_features=self.hidden_dim,
                dtype=self.dtype,
                use_bias=False,
                normalize_qk=False,
            )(inputs_q=x, mask=attention_mask)
            x = x + residual
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.Dense(features=self.ff_dim)(x)
            x = nn.gelu(x)
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
            x = x + residual
        
        mantissa_logits = nn.Dense(features=self.vocab_size)(x)
        exponent_logits = nn.Dense(features=self.vocab_size)(x)
        return mantissa_logits, exponent_logits 

@jax.jit
def train_step(state, mantissa_tokens, start_tokens):
    input_tokens = jnp.concatenate([start_tokens, tokens[..., :-1]], axis=-1)
    target_tokens = tokens
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params}, 
            tokens=input_tokens, 
            training=True, 
            rngs={'dropout': jax.random.PRNGKey(state.step)}
        )
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, target_tokens))
        return loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return loss, state

@jax.jit
def test_step(state, tokens, start_tokens):
    input_tokens = jnp.concatenate([start_tokens, tokens[..., :-1]], axis=-1)
    target_tokens = tokens
    logits = state.apply_fn({'params': state.params}, tokens=input_tokens, training=False)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, target_tokens))
    return loss

def sample_context(state, prompt_tokens, vocab_size, context_length, temperature=1.0):
    tokens = jnp.zeros((1, context_length), dtype=jnp.uint32)
    for i in range(len(prompt_tokens)):
        tokens = tokens.at[0, i].set(prompt_tokens[i])
    vocab = jnp.arange(vocab_size, dtype=jnp.uint32)

    @jax.jit
    def get_logits(state, tokens):
        return state.apply_fn({'params': state.params}, tokens=tokens, training=False)

    for i in range(len(prompt_tokens)-1, context_length):
        logits = get_logits(state, tokens)
        logits = logits[0, i, :] / temperature
        probs = nn.softmax(logits)
        next_token = jax.random.choice(jax.random.PRNGKey(state.step+i), a=vocab, p=probs)
        if i == context_length - 1:
            # Remove the start token and add last token to the end.
            tokens = tokens[:, 1:]
            tokens = jnp.concatenate(
                [tokens, jnp.ones((1, 1), dtype=jnp.uint32) * next_token], 
                axis=-1
            )
        else:
            tokens = tokens.at[0, i+1].set(next_token)
    
    return tokens

def preprocess_tokens(tokens: jax.Array):
    tokens_reshape = jnp.reshape(tokens, (tokens.shape[0], tokens.shape[1]//2, 2))
    mantissa_tokens = tokens_reshape[:, :, 0]
    exponent_tokens = tokens_reshape[:, :, 1]
    return mantissa_tokens, exponent_tokens

def main():
    output_path = 'data/ar_hypernet_output/17'
    #dataset_path = 'data/colored-primitives-ngp-image-2291-byte-pair'
    dataset_path = 'data/colored-primitives-ngp-image-2291-8bit'
    split_size = 0.2
    split_seed = 0
    train_set, test_set, field_config, param_map, context_length = \
        load_dataset(dataset_path, split_size, split_seed)
    print('Image width', field_config['image_width'])
    print('Image height', field_config['image_height'])
    print('Context length', context_length)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    vocab_size = 2**8
    #tokenize_fn = fpt.byte_pair_tokenize
    #detokenize_fn = fpt.byte_pair_detokenize
    tokenize_fn = fpt.u8_tokenize
    detokenize_fn = fpt.u8_detokenize

    field_model = ngp_image.create_model_from_config(field_config)
    field_state = ngp_image.create_train_state(field_model, 3e-4, jax.random.PRNGKey(0))
    base_tokens = train_set[200]['tokens']
    tokens = jnp.array(jnp.expand_dims(train_set[200]['tokens'], axis=0), dtype=jnp.uint8)
    flat_params = detokenize_fn(jnp.ravel(tokens))
    print(flat_params)
    params = unflatten_params(jnp.array(flat_params, dtype=jnp.float32), param_map)
    field_state = field_state.replace(params=params)
    field_render = ngp_image.render_image(
        field_state, field_config['image_height'], field_config['image_width'], field_config['channels']
    )
    field_render = jax.device_put(field_render, jax.devices('cpu')[0])
    plt.imsave('data/fp8_quality_check.png', field_render)
    exit()

    '''
    field_model = ngp_image.create_model_from_config(field_config)
    field_state = ngp_image.create_train_state(field_model, 3e-4, jax.random.PRNGKey(0))
    
    # For some reason HF Datasets doesn't store the tokens as uint8, so they need to be converted.
    tokens = jnp.array(jnp.expand_dims(train_set[300]['tokens'], axis=0), dtype=jnp.uint8)
    #print(tokens[0, :32])
    #mantissa_tokens, exponent_tokens = preprocess_tokens(tokens)
    #print(mantissa_tokens[0, :16])
    #print(exponent_tokens[0, :16])
    flat_params = detokenize_fn(jnp.ravel(tokens))
    #print(flat_params)
    #print(flat_params.shape)
    #exit()
    params = unflatten_params(jnp.array(flat_params, dtype=jnp.float32), param_map)
    #print(params)
    #print(field_config)
    field_state = field_state.replace(params=params)
    field_render = ngp_image.render_image(
        field_state, field_config['image_height'], field_config['image_width'], field_config['channels']
    )
    field_render = jax.device_put(field_render, jax.devices('cpu')[0])
    plt.imsave('data/byte_pair_quality_check.png', field_render)
    exit()
    '''

    num_epochs = 200
    batch_size = 4
    hidden_dim = 256
    ff_dim = 256
    num_attention_heads = 32
    num_blocks = 24
    learning_rate = 1e-4
    weight_decay = 1e-6
    sample_temperature = 1.0
    dropout_rate = 0.2
    start_token = vocab_size
    batched_start_tokens = jnp.full((batch_size, 1), fill_value=start_token, dtype=jnp.uint8)

    wandb_config = {
        'batch_size': batch_size,
        'context_length': context_length,
        'hidden_dim': hidden_dim,
        'num_attention_heads': num_attention_heads,
        'num_blocks': num_blocks,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'sample_temperature': sample_temperature,
        'dropout_rate': dropout_rate,
        'ff_dim': ff_dim,
    }
        
    model = ArHypernet(
        vocab_size=vocab_size,
        context_length=context_length,
        hidden_dim=hidden_dim,
        num_attention_heads=num_attention_heads,
        num_blocks=num_blocks,
        dtype=jnp.bfloat16,
        dropout_rate=dropout_rate,
        ff_dim=ff_dim
    )
    x = jnp.ones((batch_size, context_length), dtype=jnp.uint32)
    params_key = jax.random.PRNGKey(91)
    params = model.init(params_key, tokens=x, training=False)['params']
    opt = optax.chain(
        optax.zero_nans(),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)
    
    num_train_samples = len(train_set)
    train_steps = num_train_samples // batch_size
    print('Train set size:', num_train_samples)
    num_test_samples = len(test_set)
    test_steps = num_test_samples // batch_size
    print('Test set size:', num_test_samples)
    
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print('Param count', param_count)
    wandb_config['param_count'] = param_count
    wandb.init(project='ar-hypernet', config=wandb_config)
    wandb_loss_accumulation_steps = 300
    steps_since_loss_report = 0
    for epoch in range(num_epochs):
        train_set = train_set.shuffle(seed=epoch)
        train_iterator = train_set.iter(batch_size)
        accumulated_losses = []
        for step in range(train_steps):
            tokens = next(train_iterator)['tokens']
            loss, state = train_step(state, tokens=tokens, start_tokens=batched_start_tokens)
            accumulated_losses.append(loss)
            steps_since_loss_report += 1
            if steps_since_loss_report >= wandb_loss_accumulation_steps:
                average_loss = sum(accumulated_losses) / len(accumulated_losses)
                wandb.log({'loss': average_loss}, step=state.step)
                accumulated_losses = []
                steps_since_loss_report = 0
                #print(f'step {step}, loss {loss}')
        
        test_set = test_set.shuffle(seed=epoch)
        test_iterator = test_set.iter(batch_size)
        losses_this_test = []
        for step in range(test_steps):
            tokens = next(test_iterator)['tokens']
            loss = test_step(state, tokens=tokens, start_tokens=batched_start_tokens)
            losses_this_test.append(loss)
        average_test_loss = sum(losses_this_test) / len(losses_this_test)
        wandb.log({'test_loss': average_test_loss}, step=state.step)
        #print(f'epoch {epoch}, test_loss {average_test_loss}')
        
        tokens = sample_context(
            state, 
            prompt_tokens=[start_token], 
            vocab_size=vocab_size,
            context_length=context_length, 
            temperature=sample_temperature
        )[0]
        print(tokens)
        flat_params = detokenize_fn(tokens)
        flat_params = jnp.nan_to_num(flat_params)
        params = unflatten_params(jnp.array(flat_params, dtype=jnp.float32), param_map)
        field_state = field_state.replace(params=params)
        field_render = ngp_image.render_image(
            field_state, field_config['image_height'], field_config['image_width'], field_config['channels']
        )
        field_render = jax.device_put(field_render, jax.devices('cpu')[0])
        plt.imsave(os.path.join(output_path, f'{epoch}.png'), field_render)

if __name__ == '__main__':
    main()
