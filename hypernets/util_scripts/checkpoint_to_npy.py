import os, argparse, json
from orbax import checkpoint as ocp
import numpy as np
import jax

from hypernets.common.pytree_utils import move_pytree_to_cpu, flatten_dict

def convert_split_field_conv_ae(model_config_dict: dict, args: argparse.Namespace):
    from hypernets import split_field_conv_ae as target
    model_config = target.SplitFieldConvAeConfig(model_config_dict)
    autoencoder_model, _, _ = target.init_model_from_config(model_config)
    state = target.init_model_state(
        key=jax.random.PRNGKey(0), 
        model=autoencoder_model,
        model_config=model_config,
        use_batch_size=False
    )
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=True))
    checkpointer.restore(os.path.abspath(args.checkpoint), item=state)
    cpu_params = move_pytree_to_cpu(state.params)
    
    abs_output_path = os.path.abspath(args.output)
    encoder_save_path = os.path.join(
        abs_output_path, 
        f'{model_config.model_name}_encoder.npy'
    )
    decoder_save_path = os.path.join(
        abs_output_path,
        f'{model_config.model_name}_decoder.npy'
    )
    np.save(encoder_save_path, cpu_params['encoder'])
    print(f'Saved encoder to: {encoder_save_path}')
    np.save(decoder_save_path, cpu_params['decoder'])
    print(f'Saved decoder to: {decoder_save_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to the Orbax checkpoint folder'
    )
    parser.add_argument(
        '--config', type=str, required=True, 
        help='Path to the checkpoint\'s corresponding configuration file'
    )
    parser.add_argument(
        '--output', type=str, default='data/npy_models/',
        help='Directory where the npy file(s) will be saved'
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        model_config_dict = json.load(f)
    print(f'Loaded {args.config}:')
    print(json.dumps(model_config_dict, indent=4))

    if not os.path.exists(args.output):
        print('Created output directory because it did not exist')
        os.makedirs(args.output)
    
    model_type = model_config_dict['model_type']
    if model_type == 'split_field_conv_ae':
        convert_split_field_conv_ae(model_config_dict, args)
    else:
        raise ValueError(f'Unkown model type: {model_type}')

if __name__ == '__main__':
    main()
