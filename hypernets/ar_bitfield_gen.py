import jax
from jax import numpy as jnp
from flax import linen as nn
import optax
from flax.training.train_state import TrainState
import datasets
import os, json
from typing import Any
from functools import partial
from fp_tokenization import bitfield16_to_fp32 
import matplotlib.pyplot as plt
import wandb

def load_dataset(dataset_path, test_size, split_seed):
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
    context_length = int(train[0]['bitfields'].shape[0]//16)
    return train, test, context_length

class ArHypernet(nn.Module):
    vocab_size: int
    context_length: int
    tok_emb_dim: int
    hidden_dim: int
    ff_dim: int
    num_attention_heads: int
    num_blocks: int
    dtype: Any
    dropout_rate: float

    @nn.compact
    def __call__(self, tokens, training):
        # Add 1 to vocab size to allow for start token.
        # Start token should never be predicted so it is not added to the base vocab.
        emb_table = nn.Embed(num_embeddings=self.vocab_size+1, features=self.tok_emb_dim)
        def make_embedding(table, id):
            return table(id)
        #print(tokens.shape)
        tok_emb = jax.vmap(
            jax.vmap(
                make_embedding, in_axes=(None, 0)
            ), in_axes=(None, 0),
        )(emb_table, tokens)
        #print(tok_emb.shape)
        tok_emb = nn.DenseGeneral(features=self.hidden_dim, axis=(-1, -2))(tok_emb)
        #print(tok_emb.shape)
        positions = jnp.arange(self.context_length, dtype=jnp.uint32)
        pos_emb = nn.Embed(num_embeddings=self.context_length, features=self.hidden_dim)(positions)
        x = tok_emb + pos_emb
        #print(x.shape)
        attention_mask = nn.make_causal_mask(
            jnp.zeros(shape=(tokens.shape[0], self.context_length), dtype=self.dtype)
        )
        
        for _ in range(self.num_blocks):
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.MultiHeadDotProductAttention(
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
        
        logits = nn.DenseGeneral(features=(16, self.vocab_size), axis=-1)(x)
        return logits

@partial(jax.jit, static_argnames=('context_length',))
def train_step(state, tokens, start_tokens, context_length):
    tokens = jnp.reshape(tokens, (tokens.shape[0], context_length, 16))
    input_tokens = jnp.concatenate([start_tokens, tokens[..., :-1, :]], axis=1)
    target_tokens = tokens
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params}, 
            tokens=input_tokens, 
            training=True, 
            rngs={'dropout': jax.random.PRNGKey(state.step)}
        )
        #print(logits.shape)
        sce_loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_tokens)
        #print(sce_loss.shape)
        loss = jnp.mean(sce_loss)
        return loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return loss, state

@partial(jax.jit, static_argnames=('context_length',))
def test_step(state, tokens, start_tokens, context_length):
    tokens = jnp.reshape(tokens, (tokens.shape[0], context_length, 16))
    input_tokens = jnp.concatenate([start_tokens, tokens[..., :-1, :]], axis=1)
    target_tokens = tokens
    logits = state.apply_fn({'params': state.params}, tokens=input_tokens, training=False)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, target_tokens))
    return loss

def sample_context(state, prompt_tokens, vocab_size, context_length, temperature=1.0):
    tokens = jnp.zeros((1, context_length, 16), dtype=jnp.uint32)
    for i in range(len(prompt_tokens)):
        tokens = tokens.at[0, i, :].set(prompt_tokens[i])
    vocab = jnp.arange(vocab_size, dtype=jnp.uint32)

    @jax.jit
    def get_logits(state, tokens):
        return state.apply_fn({'params': state.params}, tokens=tokens, training=False)

    @jax.jit
    def sample_bitfield(logits):
        def sample_bit(bit_logit):
            return jnp.argmax(bit_logit)
        # Map across the bitfield dim.
        # Each set of logits gives the distribution for 1 bit so each sample gives us 1 bit in the bitfield.
        bitfield = jax.vmap(sample_bit, in_axes=0)(logits)
        return bitfield

    for i in range(len(prompt_tokens)-1, context_length):
        logits = get_logits(state, tokens)[0, i, :, :]
        next_token = jnp.array(sample_bitfield(logits), dtype=jnp.uint32)
        if i == context_length - 1:
            # Remove the start token and add last token to the end.
            tokens = tokens[:, 1:, :]
            tokens = jnp.concatenate(
                [tokens, jnp.reshape(next_token, (1, 1, next_token.shape[-1]))], 
                axis=1
            )
        else:
            tokens = tokens.at[0, i+1, :].set(next_token)
    
    return tokens

@jax.jit
def bitfield_kernel_test(tokens):
    bitfields = jnp.array(tokens[0], dtype=jnp.uint32)
    fp_re = bitfield16_to_fp32(bitfields)
    return fp_re

def main():
    output_path = 'data/ar_bitfiled_gen_output/0'
    dataset_path = 'data/mnist-bitfield16'
    split_size = 0.2
    split_seed = 0
    train_set, test_set, context_length = \
        load_dataset(dataset_path, split_size, split_seed)
    image_width = 28
    image_height = 28
    print('Image width', image_width)
    print('Image height', image_height)
    print('Context length', context_length)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    vocab_size = 2 # 0 and 1
    print('Vocab size', vocab_size)

    num_epochs = 200
    batch_size = 32
    tok_emb_dim = 2
    hidden_dim = 32
    ff_dim = 32
    num_attention_heads = 8
    num_blocks = 12
    learning_rate = 1e-4
    weight_decay = 1e-6
    sample_temperature = 1.0
    dropout_rate = 0.2
    start_token = vocab_size
    batched_start_tokens = jnp.full((batch_size, 1, 16), fill_value=start_token, dtype=jnp.uint32)

    wandb_config = {
        'batch_size': batch_size,
        'context_length': context_length,
        'tok_emb_dim': tok_emb_dim,
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
        tok_emb_dim=tok_emb_dim,
        hidden_dim=hidden_dim,
        num_attention_heads=num_attention_heads,
        num_blocks=num_blocks,
        dtype=jnp.bfloat16,
        dropout_rate=dropout_rate,
        ff_dim=ff_dim
    )
    x = jnp.ones((batch_size, context_length, 16), dtype=jnp.uint32)
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
    print('Train steps:', train_steps)
    num_test_samples = len(test_set)
    test_steps = num_test_samples // batch_size
    print('Test set size:', num_test_samples)
    print('Test steps:', test_steps)
    
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print('Param count', param_count)
    wandb_config['param_count'] = param_count
    #wandb.init(project='ar-bitfield-gen', config=wandb_config)
    wandb_loss_accumulation_steps = 10#300
    steps_since_loss_report = 0
    for epoch in range(num_epochs):
        train_set = train_set.shuffle(seed=epoch)
        train_iterator = train_set.iter(batch_size)
        accumulated_losses = []
        for step in range(train_steps):
            tokens = next(train_iterator)['bitfields']
            loss, state = train_step(
                state, tokens=tokens, start_tokens=batched_start_tokens, context_length=context_length
            )
            accumulated_losses.append(loss)
            steps_since_loss_report += 1
            if steps_since_loss_report >= wandb_loss_accumulation_steps:
                average_loss = sum(accumulated_losses) / len(accumulated_losses)
                #wandb.log({'loss': average_loss}, step=state.step)
                accumulated_losses = []
                steps_since_loss_report = 0
                print(f'step {step}, loss {loss}')
                break
        
        test_set = test_set.shuffle(seed=epoch)
        test_iterator = test_set.iter(batch_size)
        losses_this_test = []
        for step in range(test_steps):
            tokens = next(test_iterator)['bitfields']
            tokens = jnp.reshape(tokens, (tokens.shape[0], context_length, 16))
            loss = test_step(
                state, tokens=tokens, start_tokens=batched_start_tokens, context_length=context_length
            )
            losses_this_test.append(loss)
            print(loss)
            break
        average_test_loss = sum(losses_this_test) / len(losses_this_test)
        #wandb.log({'test_loss': average_test_loss}, step=state.step)
        #print(f'epoch {epoch}, test_loss {average_test_loss}')
        
        sampled_tokens = sample_context(
            state, 
            prompt_tokens=[jnp.ones((16,), dtype=jnp.uint32) * start_token], 
            vocab_size=vocab_size,
            context_length=context_length, 
            temperature=sample_temperature
        )[0]
        image_re = jnp.reshape(bitfield16_to_fp32(jnp.ravel(sampled_tokens)), (image_height, image_width))
        plt.imsave(os.path.join(output_path, f'{epoch}.png'), image_re)

if __name__ == '__main__':
    main()
