import jax
import jax.numpy as jnp

a = jnp.ones((100, 64))
b = jnp.ones((100, 64))

c = jax.vmap(lambda x, y: x @ y.T, in_axes=(0, None))(a, b)
print(c.shape)
c = jnp.sum(c, axis=0, keepdims=False)
print(c.shape)

d = jnp.ones((100, 64))
e = d.T @ c
print(e.shape)

q = jnp.ones((1, 100, 64))
k = jnp.ones((1, 100, 64))
v = jnp.ones((1, 100, 64))
kv = jnp.einsum('nsh,nsh->nh', k, v)
print('kv', kv.shape)

z = 1/(jnp.einsum("nlh,nh->nlh", q, jnp.sum(k, axis=1)))
print('z', z.shape)

v = jnp.einsum("nlh,nh,nlh->nlh", q, kv, z)
print('v', v.shape)