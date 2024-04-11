import jax
from jax import numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
from typing import Any

with open('data/shakespeare.txt', 'r', encoding='utf8') as f:
    text = f.read()

print(text[:500])

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print('Vocab size:', vocab_size)

stoi = { ch:i for i, ch in enumerate(chars) }
itos = {i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
print(encode('hello world'))
print(decode(encode('hello world')))

data = jnp.array(encode(text), dtype=jnp.uint32)
print(data[:200])

split_index = int(0.9 * len(data))
train_data = data[:split_index]
val_data = data[split_index:]

def get_batch(split, batch_size, context_length, seed):
    data = train_data if split == 'train' else val_data
    start_indices = jax.random.randint(
        key=jax.random.PRNGKey(seed),
        shape=(batch_size,),
        minval=0,
        maxval=data.shape[0]-context_length
    )
    x = jnp.array(jnp.stack([data[i:i+context_length] for i in start_indices]))
    return x

class TransformerLM(nn.Module):
    vocab_size: int
    context_length: int
    embedding_dim: int
    hidden_dim: int
    num_attention_heads: int
    num_blocks: int
    dtype: Any
    dropout_rate: float

    @nn.compact
    def __call__(self, tokens, training):
        attention_mask = nn.make_causal_mask(tokens, dtype=self.dtype)
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embedding_dim)(tokens)
        positions = jnp.arange(self.context_length, dtype=jnp.uint32)
        pos_emb = nn.Embed(num_embeddings=self.context_length, features=self.embedding_dim)(positions)
        x = x + pos_emb
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        x = nn.Dense(features=self.hidden_dim)(x)

        for _ in range(self.num_blocks):
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.MultiHeadDotProductAttention(
                num_heads=self.num_attention_heads,
                qkv_features=self.hidden_dim,
                out_features=self.hidden_dim,
                dtype=self.dtype
            )(inputs_q=x, mask=attention_mask)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
            x = x + residual
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
            x = x + residual

        logits = nn.Dense(features=self.vocab_size)(x)
        return logits

@jax.jit
def train_step(state, tokens, attention_mask):
    key = jax.random.PRNGKey(state.step)
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params}, 
            tokens=tokens, 
            training=True,
            rngs={'dropout': jax.random.PRNGKey(state.step)}
        ) 
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, tokens))
        return loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return loss, state

@jax.jit
def test_step(state, tokens, attention_mask):
    logits = state.apply_fn({'params': state.params}, tokens=tokens, training=False)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, tokens))
    return loss

def sample_context(state, prompt_token, context_length, temperature=1.0):
    tokens_to_sample = context_length - 1
    filler_tokens = jnp.zeros((tokens_to_sample), dtype=jnp.uint32)
    tokens = jnp.array([prompt_token], dtype=jnp.uint32)
    tokens = jnp.concatenate((tokens, filler_tokens), axis=-1)
    tokens = jnp.expand_dims(tokens, axis=0)
    attention_mask = nn.make_causal_mask(tokens, dtype=jnp.bfloat16)

    @jax.jit
    def get_logits(state, tokens, attention_mask):
        return state.apply_fn({'params': state.params}, tokens=tokens, training=False)

    for i in range(tokens_to_sample):
        logits = get_logits(state, tokens, attention_mask)
        logits = logits[:, i, :] / temperature
        probs = nn.softmax(logits)
        next_token = jax.random.categorical(jax.random.PRNGKey(state.step+i), probs, shape=(1,))
        next_token = jnp.array(next_token, dtype=jnp.uint32)[0]
        tokens = tokens.at[0, i+1].set(next_token)

    return tokens

context_length = 256
batch_size = 64
embedding_dim = 64
hidden_dim = 64
num_attention_heads = 1
num_blocks = 4
dtype = jnp.bfloat16
learning_rate = 3e-4
weight_decay = 1e-4
dropout_rate = 0.2

x = get_batch('train', batch_size, context_length, 0)
attention_mask = nn.make_causal_mask(x)
print(x.shape)
print(attention_mask[0])

model = TransformerLM(
    vocab_size=vocab_size,
    context_length=context_length,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    num_attention_heads=num_attention_heads,
    num_blocks=num_blocks,
    dtype=dtype,
    dropout_rate=dropout_rate
)
params = model.init(
    jax.random.PRNGKey(0), 
    tokens=x, 
    training=False, 
    #rngs={'dropout': jax.random.PRNGKey(0)}
)['params']
opt = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)

num_epochs = 20
train_steps = 200
test_steps = 100
print('train steps', train_steps)
print('test steps', test_steps)
prompt_token = 18
sample_temperature = 0.01

for epoch in range(num_epochs):
    losses_this_epoch = []
    for step in range(train_steps):
        tokens = get_batch('train', batch_size, context_length, state.step)
        loss, state = train_step(state, tokens, attention_mask)
        losses_this_epoch.append(loss)
        #print(f'step {step}, loss {loss}')
    average_loss = sum(losses_this_epoch) / len(losses_this_epoch)
    print(f'epoch {epoch}, loss {average_loss}')

    losses_this_test = []
    for step in range(test_steps):
        tokens = get_batch('test', batch_size, context_length, step)
        loss = test_step(state, tokens, attention_mask)
        losses_this_test.append(loss)
    average_test_loss = sum(losses_this_test) / len(losses_this_test)
    print(f'epoch {epoch}, test_loss {average_test_loss}')

    tokens = sample_context(state, prompt_token, context_length, sample_temperature)[0]
    sampled_text = decode(np.array(tokens).tolist())
    print('sampled text:')
    print(sampled_text, '\n')
