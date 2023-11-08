import jax.numpy as jnp
import argparse
import glob
import os
import json
import pprint

def _generate_leaf_map(leaf, leaf_name, packed_width):
    leaf_map = {}
    if leaf_name == 'hash_table':
        leaf_map['num_entries'] = leaf.shape[0]
        leaf_map['feature_dim'] = leaf.shape[1]
        leaf_map['height'] = (leaf.shape[0] * leaf_map['feature_dim']) // packed_width
        leaf_map['width'] = packed_width
        return leaf_map

    # If the matrix is one-dimensional, then it is a bias vector.
    if len(leaf.shape) == 1:
        leaf_map['height'] = 1
        leaf_map['width'] = leaf.shape[0]
        leaf_map['transposed'] = False
        return leaf_map
    
    height = leaf.shape[0]
    width = leaf.shape[1]
    height_greater_than_width = height > width
    height_equal_to_packed_width = height == packed_width
    # The second operand is important because we don't want to transpose
    # unless it will create a shape that can be concatenated with the other
    # parameters along the height axis, i.e. (parameter_height, packed_width).
    if height_greater_than_width and height_equal_to_packed_width:
        height = leaf.shape[1]
        width = leaf.shape[0]
        transposed = True
    else:
        transposed = False
    leaf_map['height'] = height
    leaf_map['width'] = width
    leaf_map['transposed'] = transposed
    return leaf_map

def generate_weight_map(module, packed_width):
    weight_map = {}
    for key in module.keys():
        sub_module = module[key]
        if isinstance(sub_module, dict):
            weight_map[key] = generate_weight_map(sub_module, packed_width)
        else:
            weight_map[key] = _generate_leaf_map(sub_module, key, packed_width)
            weight_map_with_name = weight_map[key].copy()
            weight_map_with_name['layer'] = key
    return weight_map

def _pack_leaf(leaf, leaf_name, leaf_map, target_width):
    if leaf_name == 'hash_table':
        return jnp.reshape(jnp.ravel(leaf), (leaf_map['height'], leaf_map['width']))
    packed_leaf = leaf
    if leaf_map['transposed']:
        packed_leaf = jnp.transpose(packed_leaf)
    if len(packed_leaf.shape) == 1:
        packed_leaf = jnp.expand_dims(packed_leaf, axis=0)
    actual_width = leaf_map['width']
    if actual_width < target_width:
        packed_leaf = jnp.concatenate([
            packed_leaf, jnp.zeros((packed_leaf.shape[0], target_width - actual_width))
        ], axis=-1)
    return packed_leaf

def pack_weights(module, packed_width, weight_map):
    packed_weights = []
    for key in module.keys():
        sub_module = module[key]
        sub_module_map = weight_map[key]
        if isinstance(sub_module, dict):
            packed_weights.append(pack_weights(sub_module, packed_width, sub_module_map))
        else:
            packed_weights.append(_pack_leaf(sub_module, key, sub_module_map, packed_width))
    return jnp.concatenate(packed_weights, axis=0)

def _unpack_leaf(packed_leaf, leaf_name, leaf_map):
    if leaf_name == 'hash_table':
        hash_table_shape = (leaf_map['num_entries'], leaf_map['feature_dim'])
        return jnp.reshape(jnp.ravel(packed_leaf), hash_table_shape)
    unpacked_leaf = packed_leaf
    if leaf_map['transposed']:
        unpacked_leaf = jnp.transpose(unpacked_leaf)
    if len(unpacked_leaf.shape) == 1:
        unpacked_leaf = jnp.expand_dims(unpacked_leaf, axis=0)
    return unpacked_leaf

def unpack_weights(packed_weights, module_map, start_height=0):
    end_height = start_height
    for key in module_map.keys():
        sub_module_map = module_map[key]
        if 'height' not in sub_module_map.keys():
            module_map[key], end_height = unpack_weights(
                packed_weights, sub_module_map, start_height
            )
        else:
            module_height = sub_module_map['height']
            end_height += module_height
            packed_leaf = packed_weights[start_height:end_height]
            module_map[key] = _unpack_leaf(packed_leaf, key, sub_module_map)
        start_height = end_height
    return module_map, end_height

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    help_text = 'If true, remove original files after packing.'
    parser.add_argument('--rm', type=bool, default=False, help=help_text)
    parser.add_argument('--packed_width', type=int, default=64)
    parser.add_argument('--loss_threshold', type=float, default=1e-3)
    args = parser.parse_args()

    if not os.path.isdir(args.output_path):
        print(f'Creating output directory: {args.output_path}')
        os.makedirs(args.output_path)

    path_list = []
    if os.path.isdir(args.input_path):
        file_pattern = os.path.join(args.input_path, '**/*.npy')
        path_list.extend(glob.glob(file_pattern, recursive=True))
    elif os.path.isfile(args.input_path) and args.input_path.endswith('.npy'):
        path_list.append(args.input_path)
    else:
        raise ValueError('Invalid input path: {}'.format(args.input_path))
    print(f'Found {len(path_list)} file(s)')

    first_module = jnp.load(path_list[0], allow_pickle=True).tolist()['params']
    weight_map = generate_weight_map(first_module, args.packed_width)
    print('Weight map:')
    print(json.dumps(weight_map, indent=4))
    with open(os.path.join(args.output_path, f'weight_map.json'), 'w') as f:
        json.dump(weight_map, f, indent=4)

    for path in path_list:
        basename = os.path.basename(path)
        nerf_dict = jnp.load(path, allow_pickle=True).tolist()
        final_loss = nerf_dict['final_loss']
        if final_loss > args.loss_threshold:
            print(f'Skipping {basename} due to high loss: {final_loss}')
            continue
        print(f'Packing {basename}')
        module = nerf_dict['params']
        packed_weights = pack_weights(module, args.packed_width, weight_map)
        jnp.save(os.path.join(args.output_path, basename), packed_weights)
        print(f'Saved {basename} to {args.output_path}')
        if args.rm:
            os.remove(path)
            print(f'Deleted {path}')

if __name__ == '__main__':
    main()