import os
import sys
sys.path.append(os.getcwd())
import pytest
import jax.numpy as jnp
import hypernets.packing.ngp as ngp_packing
import json

@pytest.fixture
def gt_weight_shapes():
    packed_width = 64
    hash_table_entries = 1536
    hash_table_feature_dim = 2
    return {
        'packed_width': packed_width,
        'bias_1_width': packed_width,
        'kernel_1_height': 6,
        'kernel_1_width': packed_width,
        'bias_2_width': packed_width,
        'kernel_2_height': packed_width,
        'kernel_2_width': packed_width,
        'bias_3_width': 3,
        'kernel_3_height': packed_width,
        'kernel_3_width': 3,
        'hash_table_entries': hash_table_entries,
        'hash_table_feature_dim': hash_table_feature_dim,
        'hash_table_height': (hash_table_entries * hash_table_feature_dim) // packed_width,
        'hash_table_width': packed_width
    }

@pytest.fixture
def gt_weights(gt_weight_shapes):
    def arange_kernel(height, width):
        return jnp.arange(height * width).reshape(height, width)
    def arange_bias(width):
        return jnp.arange(width)
    def arange_hash_table(num_entries, feature_dim):
        return jnp.arange(num_entries * feature_dim).reshape(num_entries, feature_dim)
    return {
        'FeedForward': {
            'Dense_0': {
                'bias': arange_bias(gt_weight_shapes['bias_1_width']),
                'kernel': arange_kernel(
                    gt_weight_shapes['kernel_1_height'], 
                    gt_weight_shapes['kernel_1_width']
                )     
            },
            'Dense_1': {
                'bias': arange_bias(gt_weight_shapes['bias_2_width']),
                'kernel': arange_kernel(
                    gt_weight_shapes['kernel_2_height'], 
                    gt_weight_shapes['kernel_2_width']
                )
            },
            'Dense_2': {
                'bias': arange_bias(gt_weight_shapes['bias_3_width']),
                'kernel': arange_kernel(
                    gt_weight_shapes['kernel_3_height'], 
                    gt_weight_shapes['kernel_3_width']
                )
            }
        },
        'MultiResolutionHashEncoding': {
            'hash_table': arange_hash_table(
                gt_weight_shapes['hash_table_entries'], 
                gt_weight_shapes['hash_table_feature_dim']
            )
        }
    }

@pytest.fixture
def gt_weight_map(gt_weight_shapes):
    bias_height = 1
    return {
        'FeedForward': {
            'Dense_0': {
                'bias': {
                    'height': bias_height,
                    'width': gt_weight_shapes['bias_1_width'],
                    'transposed': False
                },
                'kernel': {
                    'height': gt_weight_shapes['kernel_1_height'],
                    'width': gt_weight_shapes['kernel_1_width'],
                    'transposed': False
                }
            },
            'Dense_1': {
                'bias': {
                    'height': bias_height,
                    'width': gt_weight_shapes['bias_2_width'],
                    'transposed': False
                },
                'kernel': {
                    'height': gt_weight_shapes['kernel_2_height'],
                    'width': gt_weight_shapes['kernel_2_width'],
                    'transposed': False
                }
            },
            'Dense_2': {
                'bias': {
                    'height': bias_height,
                    'width': gt_weight_shapes['bias_3_width'],
                    'transposed': False
                },
                'kernel': {
                    'height': gt_weight_shapes['kernel_3_width'],
                    'width': gt_weight_shapes['kernel_3_height'],
                    'transposed': True
                }
            }
        },
        'MultiResolutionHashEncoding': {
            'hash_table': {
                'num_entries': gt_weight_shapes['hash_table_entries'],
                'feature_dim': gt_weight_shapes['hash_table_feature_dim'],
                'height': gt_weight_shapes['hash_table_height'],
                'width': gt_weight_shapes['hash_table_width'],
                'transposed': False
            }
        }
    }

@pytest.fixture
def gt_packed_weights_shape(gt_weight_shapes):
    height = \
        gt_weight_shapes['kernel_1_height'] + gt_weight_shapes['kernel_2_height'] + \
        gt_weight_shapes['kernel_3_width'] + gt_weight_shapes['hash_table_height'] + 3
    width = gt_weight_shapes['packed_width']
    return (height, width)

def test_ngp_weight_map_generation(gt_weights, gt_weight_shapes, gt_weight_map):
    generated_weight_map = \
        ngp_packing.generate_weight_map(gt_weights, gt_weight_shapes['packed_width'])
    print(json.dumps(gt_weight_map, indent=4))
    print(json.dumps(generated_weight_map, indent=4))
    assert generated_weight_map == gt_weight_map

def test_ngp_weight_packing(
    gt_weights, gt_weight_shapes, gt_weight_map, gt_packed_weights_shape
):
    packed_weights = \
        ngp_packing.pack_weights(gt_weights, gt_weight_shapes['packed_width'], gt_weight_map)
    assert packed_weights.shape == gt_packed_weights_shape