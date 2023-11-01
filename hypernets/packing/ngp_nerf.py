import jax.numpy as jnp

def pack_weights(params, packed_width):
    packed_height = 0
    weight_map = []
    packed_weights = []
    for key in params.keys():
        new_weight_map_entry = {'layer': key}
        if key == 'MultiResolutionHashEncoding_0':
            parameter = params[key]['hash_table']
            new_weight_map_entry['num_entries'] = parameter.shape[0]
            new_weight_map_entry['feature_dim'] = parameter.shape[1]
            parameter = jnp.ravel(parameter)
            parameter_shape = parameter.shape
            packed_weights.append(jnp.reshape(
                parameter, (parameter_shape[0] // packed_width, packed_width)
            ))
            new_weight_map_entry['table_height'] = packed_weights[-1].shape[0]
            new_weight_map_entry['table_width'] = packed_weights[-1].shape[1]
        for sub_key in params[key].keys():
            parameter = params[key][sub_key]
            parameter_shape = parameter.shape
            if sub_key == 'bias':
                new_weight_map_entry['bias_width'] = parameter_shape[0]
                packed_weights.append(jnp.expand_dims(
                    jnp.concatenate([
                        parameter, jnp.zeros((packed_width - parameter_shape[0]))
                    ], axis=-1),
                    axis=0
                ))
                packed_height += 1
            elif sub_key == 'kernel':
                height_greater_than_width = parameter_shape[0] > parameter_shape[1]
                height_equal_to_packed_width = parameter_shape[0] == packed_width
                # The second operand is important because we don't want to transpose
                # unless it will create a shape that can be concatenated with the other
                # parameters along the height axis, i.e. (parameter_height, packed_width).
                if height_greater_than_width and height_equal_to_packed_width:
                    kernel_height = parameter_shape[1]
                    kernel_width = parameter_shape[0]
                    packed_weights.append(jnp.transpose(parameter))
                    new_weight_map_entry['kernel_transposed'] = True
                    packed_height += kernel_height
                else:
                    kernel_height = parameter_shape[0]
                    kernel_width = parameter_shape[1]
                    packed_weights.append(parameter)
                    new_weight_map_entry['kernel_transposed'] = False
                    packed_height += kernel_height
                new_weight_map_entry['kernel_height'] = kernel_height
                new_weight_map_entry['kernel_width'] = kernel_width
        weight_map.append(new_weight_map_entry)
    packed_weights = jnp.concatenate(packed_weights, axis=0)
    return packed_weights, weight_map

def unpack_weights(packed_weights, weight_map):
    unpacked_weights = {}
    current_height = 0
    for layer in weight_map:
        layer_name = layer['layer']
        if layer_name != 'MultiResolutionHashEncoding_0':
            bias_width = layer['bias_width']
            bias = packed_weights[current_height, :bias_width]
            current_height += 1
            kernel_height = layer['kernel_height']
            kernel_width = layer['kernel_width']
            kernel = packed_weights[current_height:current_height+kernel_height, :kernel_width]
            current_height += kernel_height
            if layer['kernel_transposed']:
                kernel = jnp.transpose(kernel)
            unpacked_weights[layer_name] = {'bias': bias, 'kernel': kernel}
        else:
            table_height = layer['table_height']
            table_width = layer['table_width']
            num_entries = layer['num_entries']
            feature_dim = layer['feature_dim']
            hash_table = packed_weights[current_height:current_height+table_height, :table_width]
            current_height += table_height
            hash_table = jnp.reshape(jnp.ravel(hash_table), (num_entries, feature_dim))
            unpacked_weights[layer_name] = {'hash_table': hash_table}
    print(unpacked_weights.keys())
    print(current_height)
    return unpacked_weights