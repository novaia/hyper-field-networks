import os, glob, argparse, copy, json, io
from fields import ngp_image
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from datasets import load_dataset, Dataset
import pyarrow as pa
import pyarrow.parquet as pq
from fields.common.flattening import generate_param_map, flatten_params
import matplotlib.pyplot as plt

def get_dataset(dataset_path):
    if dataset_path.endswith('/'):
        glob_pattern = f'{dataset_path}*.parquet'
    else:
        glob_pattern = f'{dataset_path}/*.parquet'
    parquet_files = glob.glob(glob_pattern)
    assert len(parquet_files) > 0, 'No parquet files were found in dataset directory.'
    print(f'Found {len(parquet_files)} parquet files in dataset directory.')
    dataset = load_dataset(
        'parquet', 
        data_files={'train': parquet_files},
        split='train',
        num_proc=8
    )
    dataset = dataset.with_format('jax')
    return dataset

def save_table(table_data, num_params, table_number, output_path, zfill_amount):
    print(f'Entries in table {table_number}: {len(table_data)}')
    schema = pa.schema(
        fields=[
            ('params', pa.list_(pa.float32(), list_size=num_params)),
            ('image', pa.struct([('bytes', pa.binary()), ('path', pa.string())]))
        ],
        metadata={
            b'huggingface': json.dumps({
                'info': {
                    'features': {
                        'params': {'_type': 'Sequence', 'feature': {'_type': 'Value', 'dtype': 'float32'}},
                        'image': {'_type': 'Image'}
                    }
                }
            }).encode('utf-8')
        }
    )
    table = pa.Table.from_pylist(table_data, schema=schema)
    pq.write_table(table, f'{output_path}/{str(table_number).zfill(zfill_amount)}.parquet')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--field_type', type=str, choices=['ngp_image', 'ngp_nerf'], required=True)
    parser.add_argument('--start_table', type=int, default=0)
    parser.add_argument('--samples_per_table', type=int, default=1500)
    parser.add_argument('--loss_threshold', type=float, default=4e-6)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--image_column', type=str, default='image')
    args = parser.parse_args()
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    channels = config['channels']

    dataset = get_dataset(args.input_path)
    dataset_iterator = dataset.iter(batch_size=1)
    num_samples = len(dataset)
    print('Samples in dataset:', num_samples)

    model = ngp_image.create_model_from_config(config)
    state = ngp_image.create_train_state(model, config['learning_rate'], jax.random.PRNGKey(0))
    initial_params = copy.deepcopy(state.params)
    initial_opt_state = copy.deepcopy(state.opt_state)
    initial_tx = copy.deepcopy(state.tx)
    param_map, _ = generate_param_map(state.params)

    with open(os.path.join(args.output_path, 'param_map.json'), 'w') as f:
        json.dump(param_map, f, indent=4)

    first_image = dataset[0][args.image_column]
    num_pixels = first_image.shape[0] * first_image.shape[1]
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f'Num pixels: {num_pixels:,}')
    print(f'Num params: {num_params:,}')
    print('Param to pixel ratio:', num_params/num_pixels)

    samples_per_table = args.samples_per_table 
    samples_in_current_table = 0
    start_sample = samples_per_table * args.start_table
    current_table_index = args.start_table
    num_retries = config['num_retries']
    loss_threshold = args.loss_threshold
    pq_table_data = []
    skip_on_threshold_fail = config['skip_on_threshold_fail']
    for i in range(start_sample, num_samples):
        image = dataset[i][args.image_column] / 255.0
        #print(image)
        state = state.replace(params=initial_params, tx=initial_tx, opt_state=initial_opt_state, step=0)
        
        passed_loss_threshold = False
        for k in range(num_retries):
            state = ngp_image.train_loop(
                config['train_steps'], state, image, config['batch_size'], channels
            )
            rendered_image = ngp_image.render_image(
                state, image.shape[0], image.shape[1], channels=channels
            )
            full_image_loss = jnp.mean(image - rendered_image)**2
            print(f'Sample {i}, attempt {k}, loss: {full_image_loss}')
            if full_image_loss < loss_threshold:
                passed_loss_threshold = True
                break
        if skip_on_threshold_fail and not passed_loss_threshold:
            print('skipped')
            continue
        flat_params = flatten_params(state.params, param_map, num_params)
        rendered_image = np.array(jnp.clip(rendered_image * 255, 0, 255), dtype=np.uint8)
        pil_image_bytes = io.BytesIO()
        if channels == 1:
            rendered_image = np.squeeze(rendered_image, axis=-1)
        #print(rendered_image)
        #exit()
        pil_image = Image.fromarray(rendered_image)
        pil_image.save(pil_image_bytes, format='PNG')
        pq_row_data = {
            'params': np.array(flat_params, dtype=np.float32).tolist(),
            'image': {
                'bytes': pil_image_bytes.getvalue(),
                'path': f'{i}.png'
            }
        }
        pq_table_data.append(pq_row_data)
        samples_in_current_table += 1
        if samples_in_current_table > samples_per_table:
            save_table(pq_table_data, num_params, current_table_index, args.output_path, 4)
            pq_table_data = []
            samples_in_current_table = 0
            current_table_index += 1
        #pil_image.save(os.path.join(args.output_path, f'{i}.png'))
    if samples_in_current_table > 0:
        save_table(pq_table_data, num_params, current_table_index, args.output_path, 4)

if __name__ == '__main__':
    main()
