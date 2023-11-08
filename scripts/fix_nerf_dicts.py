import glob
import os
import jax.numpy as jnp

path_list = []
input_path = 'data/synthetic_nerfs/aliens'
file_pattern = os.path.join(input_path, '**/*.npy')
path_list.extend(glob.glob(file_pattern, recursive=True))
print(len(path_list))
print(path_list[0])
for path in path_list:
    nerf_dict = jnp.load(path, allow_pickle=True).tolist()
    param_dict = nerf_dict['params']
    hash_table = {'hash_table': param_dict['MultiResolutionHashEncoding_0']['hash_table']}
    feed_forward_0 = {'Dense_0': param_dict['Dense_0'], 'Dense_1': param_dict['Dense_1']}
    feed_forward_1 = {
        'Dense_0': param_dict['Dense_2'], 
        'Dense_1': param_dict['Dense_3'], 
        'Dense_2': param_dict['Dense_4']
    }
    nerf_dict['params'] = {
        'TcnnMultiResolutionHashEncoding_0': hash_table, 
        'FeedForward_0': feed_forward_0, 
        'FeedForward_1': feed_forward_1
    }
    jnp.save(path, nerf_dict, allow_pickle=True)
    print(f'Fixed {path}')