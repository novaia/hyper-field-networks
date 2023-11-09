import jax.numpy as jnp

def pack_weights(state, packed_width):
    packed_height = 0
    weight_map = []
    packed_weights = []
    for key in state.params.keys():
        new_weight_map_entry = {'layer': key}
        for sub_key in state.params[key].keys():
            parameter = state.params[key][sub_key]
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