import jax
import numpy as np

def move_pytree_to_cpu(pytree):
    def move_to_cpu(tensor):
        return np.array(jax.device_put(tensor, device=jax.devices('cpu')[0]))
    return jax.tree_map(move_to_cpu, pytree)

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


