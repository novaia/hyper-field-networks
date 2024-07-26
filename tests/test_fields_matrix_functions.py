import os
import sys
sys.path.append(os.getcwd())
import pytest
import jax.numpy as jnp
from fields.common import matrices
import opengl_proc_gen as opg

'''
def test_process_3x4_transformation_matrix():
    scale = 2.0
    input_matrix = jnp.array([
        [1.0, 2.0, 3.0, 1.3],
        [4.0, 5.0, 6.0, 1.6],
        [7.0, 8.0, 9.0, 1.9]
    ])
    expected_matrix = jnp.array([
        [4.0, -5.0, -6.0, 1.6 * scale],
        [7.0, -8.0, -9.0, 1.9 * scale],
        [1.0, -2.0, -3.0, 1.3 * scale]
    ])
    computed_matrix = matrices.process_3x4_transformation_matrix(input_matrix, scale)
    assert jnp.allclose(expected_matrix, computed_matrix)
'''

@pytest.fixture
def pi_over_2():
    return jnp.pi/2

@pytest.fixture
def rotation_tolerance():
    return 5e-8

def test_get_rotation_matrix_2d(pi_over_2, rotation_tolerance):
    expected_matrix = jnp.array([[0.0, -1.0], [1.0, 0.0]])
    computed_matrix = matrices.get_rotation_matrix_2d(pi_over_2)
    assert jnp.allclose(expected_matrix, computed_matrix, atol=rotation_tolerance)

def test_get_x_rotation_matrix_3d(pi_over_2, rotation_tolerance):
    expected_matrix = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    computed_matrix = matrices.get_x_rotation_matrix_3d(pi_over_2)
    assert jnp.allclose(expected_matrix, computed_matrix, atol=rotation_tolerance)

def test_get_y_rotation_matrix_3d(pi_over_2, rotation_tolerance):
    expected_matrix = jnp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    computed_matrix = matrices.get_y_rotation_matrix_3d(pi_over_2)
    assert jnp.allclose(expected_matrix, computed_matrix, atol=rotation_tolerance)

def test_get_z_rotation_matrix_3d(pi_over_2, rotation_tolerance):
    expected_matrix = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    computed_matrix = matrices.get_z_rotation_matrix_3d(pi_over_2)
    assert jnp.allclose(expected_matrix, computed_matrix, atol=rotation_tolerance)

def test_opg_mat4_make_x_rotation_matrix(pi_over_2, rotation_tolerance):
    opg_matrix = jnp.array(opg.mat4_make_x_rotation(90.0))
    final_col = jnp.ravel(opg_matrix[:, -1])
    final_row = jnp.ravel(opg_matrix[-1, :])
    expected_final_col_and_row = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    assert jnp.allclose(final_col, expected_final_col_and_row)
    assert jnp.allclose(final_row, expected_final_col_and_row)
    opg_matrix = opg_matrix[:-1, :-1]
    expected_matrix = matrices.get_x_rotation_matrix_3d(pi_over_2)
    print(opg_matrix)
    print(expected_matrix)
    assert jnp.allclose(opg_matrix, expected_matrix, atol=rotation_tolerance)

'''
def test_z_axis_camera_orbit_matrix():
    expected_matrix = jnp.array([
        [0.0, 0.0, -1.0, 2.0], 
        [0.0, -1.0, 0.0, 0.0], 
        [-1.0, 0.0, 0.0, 0.0]
    ])
    computed_matrix = matrices.get_z_axis_camera_orbit_matrix(jnp.pi, 2.0)
    print(expected_matrix)
    print(computed_matrix)
    # This function requires a slightly larger tolerance than the others.
    assert jnp.allclose(expected_matrix, computed_matrix, atol=2e-7)
'''
