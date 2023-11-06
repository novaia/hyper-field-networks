import os
import sys
sys.path.append(os.getcwd())
import gc
import pandas as pd
import time
import argparse
import jax
import jax.numpy as jnp
from fields import ngp_nerf

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--csv_path', type=str)
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

if args.csv_path is None:
    args.csv_path = os.path.join(args.output_dir, 'fields.csv')
if not os.path.exists(args.csv_path):
    with open(args.csv_path, 'w+') as f:
        f.write('dataset_path,final_loss,field_path\n')
already_trained_datasets = pd.read_csv(args.csv_path)['dataset_path'].values

num_hash_table_levels = 16
max_hash_table_entries = 2**20
hash_table_feature_dim = 2
coarsest_resolution = 16
finest_resolution = 2**19
density_mlp_width = 64
color_mlp_width = 64
high_dynamic_range = False
exponential_density_activation = True

learning_rate = 1e-2
epsilon = 1e-15
weight_decay_coefficient = 1e-6
batch_size = 256 * 1024
scene_bound = 1.0
grid_resolution = 128
grid_update_interval = 16
grid_warmup_steps = 256
diagonal_n_steps = 1024
train_steps = 1000
stepsize_portion = 1.0 / 256.0
state_init_key = jax.random.PRNGKey(0)

dataset_list = os.listdir(args.dataset_dir)
for dataset_name in dataset_list:
    dataset_path = os.path.join(args.dataset_dir, dataset_name)
    if dataset_path in already_trained_datasets:
        print(f'Skipped {dataset_name}, already trained')
        continue
    print(f'Training NGP-NeRF for {dataset_name}...')
    start_time = time.time()
    dataset = ngp_nerf.load_dataset(dataset_path, 1)
    
    model = ngp_nerf.NGPNerf(
        number_of_grid_levels=num_hash_table_levels,
        max_hash_table_entries=max_hash_table_entries,
        hash_table_feature_dim=hash_table_feature_dim,
        coarsest_resolution=coarsest_resolution,
        finest_resolution=finest_resolution,
        density_mlp_width=density_mlp_width,
        color_mlp_width=color_mlp_width,
        high_dynamic_range=high_dynamic_range,
        exponential_density_activation=exponential_density_activation,
        scene_bound=scene_bound
    )

    state = ngp_nerf.create_train_state(
        model=model, 
        rng=state_init_key, 
        learning_rate=learning_rate,
        epsilon=epsilon,
        weight_decay_coefficient=weight_decay_coefficient
    )

    occupancy_grid = ngp_nerf.create_occupancy_grid(
        resolution=grid_resolution, 
        update_interval=grid_update_interval, 
        warmup_steps=grid_warmup_steps
    )

    state, occupancy_grid, final_loss = ngp_nerf.train_loop(
        batch_size=batch_size,
        train_steps=train_steps,
        dataset=dataset,
        scene_bound=scene_bound,
        diagonal_n_steps=diagonal_n_steps,
        stepsize_portion=stepsize_portion,
        occupancy_grid=occupancy_grid,
        state=state,
        return_final_loss=True
    )
    end_time = time.time()
    train_time = end_time - start_time
    print(f'Training time (sec): {train_time}')
    print('Final loss:', final_loss)
    file_name = os.path.join(args.output_dir, dataset_name) + '.npy'
    output_dict = {'final_loss': final_loss, 'params': dict(state.params)}
    jnp.save(file_name, output_dict)
    with open(args.csv_path, 'a') as f:
        f.write(f'{dataset_path},{final_loss},{file_name}\n')
    
    del dataset
    del model
    del state
    del occupancy_grid
    gc.collect()