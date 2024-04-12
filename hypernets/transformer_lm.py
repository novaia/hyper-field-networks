import jax
from jax import numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
from typing import Any
from functools import partial

with open('data/shakespeare.txt', 'r', encoding='utf8') as f:
    text = f.read()

#text = text.replace('\n', '')
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

base_data = jnp.array(encode(text), dtype=jnp.uint32)
print(base_data[:200])

split_index = int(0.9 * base_data.shape[0])
train_data = base_data[:split_index]
val_data = base_data[split_index:]

def get_batch(split, batch_size, context_length, seed):
    data = train_data if split == 'train' else val_data
    start_indices = jax.random.randint(
        key=jax.random.PRNGKey(seed),
        shape=(batch_size,),
        minval=0,
        maxval=data.shape[0]-context_length-1
    )
    tokens = jnp.array(jnp.stack([data[i:i+context_length+1] for i in start_indices]))
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    return x, y

class TransformerLM(nn.Module):
    vocab_size: int
    context_length: int
    embedding_dim: int
    hidden_dim: int
    num_attention_heads: int
    attention_dim: int
    feed_forward_dim: int
    num_blocks: int
    dtype: Any
    dropout_rate: float

    @nn.compact
    def __call__(self, tokens, training):
        attention_mask = nn.make_causal_mask(tokens)
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embedding_dim)(tokens)
        positions = jnp.arange(self.context_length, dtype=jnp.uint32)
        pos_emb = nn.Embed(num_embeddings=self.context_length, features=self.embedding_dim)(positions)
        x = x + pos_emb
        x = nn.Dense(features=self.hidden_dim)(x)

        for _ in range(self.num_blocks):
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.MultiHeadDotProductAttention(
                num_heads=self.num_attention_heads,
                qkv_features=self.attention_dim,
                out_features=self.hidden_dim,
                dtype=self.dtype,
                use_bias=False,
                dropout_rate=self.dropout_rate
            )(inputs_q=x, mask=attention_mask, deterministic=not training)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
            x = x + residual
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.Dense(features=self.feed_forward_dim)(x)
            x = nn.relu(x)
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
            x = x + residual

        logits = nn.Dense(features=self.vocab_size)(x)
        return logits

def cross_entropy(logits, labels):
    def inner_fn(logits, labels):
        log_probs = jax.nn.log_softmax(logits)
        return -jnp.sum(labels * log_probs)
    return jax.vmap(jax.vmap(inner_fn, in_axes=(0, 0)), in_axes=(0, 0))(logits, labels)

@jax.jit
def train_step(state, tokens, targets):
    key = jax.random.PRNGKey(state.step)
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params}, 
            tokens=tokens, 
            training=True,
            rngs={'dropout': jax.random.PRNGKey(state.step)}
        )
        loss = jnp.mean(cross_entropy(logits, jax.nn.one_hot(targets, logits.shape[-1])))
        #loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, tokens))
        return loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return loss, state

@jax.jit
def test_step(state, tokens, targets):
    logits = state.apply_fn({'params': state.params}, tokens=tokens, training=False)
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(jnp.where(targets == predictions, 1, 0))
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, targets))
    return loss, accuracy

def sample_context(state, prompt_tokens, vocab_size, context_length, temperature=1.0):
    tokens = jnp.zeros((1, context_length), dtype=jnp.uint32)
    for i in range(len(prompt_tokens)):
        tokens = tokens.at[0, i].set(prompt_tokens[i])
    vocab = jnp.arange(vocab_size, dtype=jnp.uint32)

    @jax.jit
    def get_logits(state, tokens):
        return state.apply_fn({'params': state.params}, tokens=tokens, training=False)

    for i in range(len(prompt_tokens)-1, context_length-1):
        logits = get_logits(state, tokens)
        logits = logits[0, i, :] / temperature
        probs = nn.softmax(logits)
        next_token = jax.random.choice(jax.random.PRNGKey(state.step+i), a=vocab, p=probs)
        #next_token = jax.random.categorical(jax.random.PRNGKey(state.step+i), logits, shape=(1,))
        #next_token = jnp.array(next_token, dtype=jnp.uint32)[0]
        tokens = tokens.at[0, i+1].set(next_token)
    return tokens

context_length = 256
batch_size = 64
embedding_dim = 256
hidden_dim = 256
attention_dim = 512
feed_forward_dim = 512
num_attention_heads = 8
num_blocks = 6
dtype = jnp.bfloat16
learning_rate = 3e-4
weight_decay = 1e-6
dropout_rate = 0.2

x, y = get_batch('train', batch_size, context_length, 0)
print(x.shape)

model = TransformerLM(
    vocab_size=vocab_size,
    context_length=context_length,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    attention_dim=attention_dim,
    feed_forward_dim=feed_forward_dim,
    num_attention_heads=num_attention_heads,
    num_blocks=num_blocks,
    dtype=dtype,
    dropout_rate=dropout_rate
)
params = model.init(
    jax.random.PRNGKey(0), 
    tokens=x, 
    training=False
)['params']
opt = optax.adam(learning_rate=learning_rate)
state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)

num_epochs = 100
train_steps = 1000
test_steps = 100
print('train steps', train_steps)
print('test steps', test_steps)
prompt_tokens = encode('First Citizen:')
sample_temperature = 1.0

for epoch in range(num_epochs):
    losses_this_epoch = []
    for step in range(train_steps):
        tokens, targets = get_batch('train', batch_size, context_length, state.step)
        loss, state = train_step(state, tokens, targets)
        losses_this_epoch.append(loss)
        #print(f'step {step}, loss {loss}')
    average_loss = sum(losses_this_epoch) / len(losses_this_epoch)
    print(f'epoch {epoch}, loss {average_loss}')

    losses_this_test = []
    accuracies_this_test = []
    for step in range(test_steps):
        tokens, targets = get_batch('test', batch_size, context_length, step)
        loss, accuracy = test_step(state, tokens, targets)
        losses_this_test.append(loss)
        accuracies_this_test.append(accuracy)
    average_test_loss = sum(losses_this_test) / len(losses_this_test)
    average_test_accuracy = sum(accuracies_this_test) / len(accuracies_this_test)
    print(f'epoch {epoch}, test_loss {average_test_loss}, test_accuracy {average_test_accuracy}')

    tokens = sample_context(state, prompt_tokens, vocab_size, context_length, sample_temperature)
    sampled_text = decode(np.array(tokens[0]).tolist())
    print('sampled text:')
    print(sampled_text, '\n')
