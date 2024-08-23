import jax
import jax.numpy as jnp
import numpy as np

def move_pytree_to_cpu(pytree, cpu_id=0):
    device = jax.devices('cpu')[0]
    def move_to_cpu(tensor):
        return np.array(jax.device_put(tensor, device=device))
    return jax.tree_map(move_to_cpu, pytree)

def move_pytree_to_gpu(pytree, gpu_id=0):
    device = jax.devices('gpu')[0]
    def move_to_gpu(tensor):
        return jax.device_put(jnp.array(tensor), device=device)
    return jax.tree_map(move_to_gpu, pytree)

# Reference: https://gist.github.com/Narsil/d5b0d747e5c8c299eb6d82709e480e3d
def flatten_dict(weights, prefix=""):
    values = {}
    for k, v in weights.items():
        newprefix = f"{prefix}.{k}" if prefix else f"{k}"
        if isinstance(v, dict):
            values.update(flatten_dict(v, prefix=newprefix))
        elif isinstance(v, np.ndarray):
            values[newprefix] = v
    return values
