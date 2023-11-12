import os
import sys
sys.path.append(os.getcwd())
import pytest
import jax.numpy as jnp
import hypernets.packing.ngp as ngp_packing

@pytest.fixture
def hash_table_data():
    num_entries = 32
    feature_dim = 2
    num_parameters = num_entries * feature_dim
    width = 16
    height = num_parameters // width
    hash_table = jnp.arange(num_parameters).reshape(num_entries, feature_dim)
    return {
        'num_entries': num_entries,
        'feature_dim': feature_dim,
        'width': width,
        'height': height,
        'hash_table': hash_table,
        'num_parameters': num_parameters
    }

def test_leaf_map_generation_for_hash_table(hash_table_data):
    leaf_map = ngp_packing._generate_leaf_map(
        hash_table_data['hash_table'], 
        'hash_table', 
        hash_table_data['width']
    )
    assert leaf_map['num_entries'] == hash_table_data['num_entries'], \
        'Number of entries do not match'
    assert leaf_map['feature_dim'] == hash_table_data['feature_dim'], \
        'Feature dimensions do not match'
    assert leaf_map['height'] == hash_table_data['height'], 'Heights do not match'
    assert leaf_map['width'] == hash_table_data['width'], 'Widths do not match'
    assert leaf_map['transposed'] == False, 'Hash table should not be transposed'

def test_leaf_map_generation_for_transposed_hash_table(hash_table_data):
    leaf_map = ngp_packing._generate_leaf_map(
        hash_table_data['hash_table'].T, 
        'hash_table', 
        hash_table_data['width']
    )
    assert leaf_map['num_entries'] == hash_table_data['num_entries'], \
        'Number of entries do not match'
    assert leaf_map['feature_dim'] == hash_table_data['feature_dim'], \
        'Feature dimensions do not match'
    assert leaf_map['height'] == hash_table_data['height'], 'Heights do not match'
    assert leaf_map['width'] == hash_table_data['width'], 'Widths do not match'
    assert leaf_map['transposed'] == True, 'Hash table should be transposed'

def test_leaf_map_generation_for_bias():
    packed_width = 8
    bias_width = packed_width
    bias = jnp.arange(bias_width)
    leaf_map = ngp_packing._generate_leaf_map(bias, 'bias', packed_width)
    assert leaf_map['height'] == 1, 'Height should be 1'
    assert leaf_map['width'] == bias_width, 'Widths do not match'
    assert leaf_map['transposed'] == False, 'Bias should not be transposed'

def test_leaf_map_generation_for_padded_bias():
    packed_width = 8 
    bias_width = 4
    bias = jnp.arange(bias_width)
    leaf_map = ngp_packing._generate_leaf_map(bias, 'bias', packed_width)
    assert leaf_map['height'] == 1, 'Height should be 1'
    assert leaf_map['width'] == bias_width, 'Widths do not match'
    assert leaf_map['transposed'] == False, 'Bias should not be transposed'

def test_leaf_map_generation_for_kernel():
    packed_width = 8
    kernel_width = packed_width
    kernel_height = 4
    kernel = jnp.arange(kernel_width*kernel_height).reshape(kernel_height, kernel_width)
    leaf_map = ngp_packing._generate_leaf_map(kernel, 'kernel', packed_width)
    assert leaf_map['height'] == kernel_height, 'Heights do not match'
    assert leaf_map['width'] == kernel_width, 'Widths do not match'
    assert leaf_map['transposed'] == False, 'Kernel should not be transposed'

def test_leaf_map_generation_for_transposed_kernel():
    packed_width = 8
    kernel_width = 4
    kernel_height = packed_width
    kernel = jnp.arange(kernel_width*kernel_height).reshape(kernel_height, kernel_width)
    leaf_map = ngp_packing._generate_leaf_map(kernel, 'kernel', packed_width)
    assert leaf_map['height'] == kernel_width, 'Heights do not match'
    assert leaf_map['width'] == kernel_height, 'Widths do not match'
    assert leaf_map['transposed'] == True, 'Kernel should be transposed'

@pytest.fixture
def packed_hash_table_data(hash_table_data):
    packed_hash_table = jnp.reshape(
        jnp.arange(hash_table_data['num_parameters']), 
        (hash_table_data['height'], hash_table_data['width'])
    )
    hash_table_map = {
        'num_entries': hash_table_data['num_entries'],
        'feature_dim': hash_table_data['feature_dim'],
        'height': hash_table_data['height'],
        'width': hash_table_data['width'],
        'transposed': False
    }
    return packed_hash_table, hash_table_data['hash_table'], hash_table_map

def test_leaf_unpacking_for_hash_table(packed_hash_table_data):
    packed_hash_table, hash_table, hash_table_map = packed_hash_table_data
    unpacked_hash_table = \
        ngp_packing._unpack_leaf(packed_hash_table, 'hash_table', hash_table_map)
    assert jnp.array_equal(unpacked_hash_table, hash_table), \
        'Unpacked hash table does not match original hash table'

def test_leaf_unpacking_for_transposed_hash_table(packed_hash_table_data):
    packed_hash_table, hash_table, hash_table_map = packed_hash_table_data
    hash_table_map['transposed'] = True
    unpacked_hash_table = \
        ngp_packing._unpack_leaf(packed_hash_table, 'hash_table', hash_table_map)
    assert jnp.array_equal(unpacked_hash_table, hash_table.T), \
        'Unpacked hash table does not match original hash table'
    
def test_leaf_unpacking_for_bias():
    bias_width = 4
    bias = jnp.arange(bias_width)
    packed_bias = jnp.expand_dims(bias, axis=0)
    bias_map = {'height': 1, 'width': bias_width, 'transposed': False}
    unpacked_bias = ngp_packing._unpack_leaf(packed_bias, 'bias', bias_map)
    assert jnp.array_equal(unpacked_bias, bias), \
        'Unpacked bias does not match original bias'

def test_leaf_unpacking_for_padded_bias():
    packed_width = 8
    bias_width = 4
    bias = jnp.arange(bias_width)
    packed_bias = jnp.expand_dims(
        jnp.concatenate([bias, jnp.zeros((packed_width - bias_width))], axis=-1),
        axis=0
    )
    bias_map = {'height': 1, 'width': bias_width, 'transposed': False}
    unpacked_bias = ngp_packing._unpack_leaf(packed_bias, 'bias', bias_map)
    assert jnp.array_equal(unpacked_bias, bias), \
        'Unpacked bias does not match original bias'
    
def test_leaf_unpacking_for_kernel():
    kernel_width = 8
    kernel_height = 4
    kernel = jnp.arange(kernel_width*kernel_height).reshape(kernel_height, kernel_width)
    kernel_map = {'height': kernel_height, 'width': kernel_width, 'transposed': False}
    unpacked_kernel = ngp_packing._unpack_leaf(kernel, 'kernel', kernel_map)
    assert jnp.array_equal(unpacked_kernel, kernel), \
        'Unpacked kernel does not match original kernel'
    
def test_leaf_unpacking_for_transposed_kernel():
    kernel_width = 4
    kernel_height = 8
    kernel = jnp.arange(kernel_width*kernel_height).reshape(kernel_height, kernel_width)
    kernel_map = {'height': kernel_width, 'width': kernel_height, 'transposed': True}
    unpacked_kernel = ngp_packing._unpack_leaf(kernel.T, 'kernel', kernel_map)
    assert jnp.array_equal(unpacked_kernel, kernel), \
        'Unpacked kernel does not match original kernel'
