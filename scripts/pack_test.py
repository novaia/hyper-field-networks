import os
import sys
sys.path.append(os.getcwd())
from hypernets.packing.ngp_nerf import pack_weights
import jax.numpy as jnp

params = jnp.load('data/synthetic_nerfs/aliens/0-39/alien_0.npy', allow_pickle=True).tolist()
params = params['params']
_ = pack_weights(params, 64)

hash_table = params['MultiResolutionHashEncoding_0']['hash_table']
print(hash_table.shape)