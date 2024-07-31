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

@pytest.mark.parametrize('degrees', [20.0, 30.0, 180.0, 0.0, 360.0])
def test_opg_mat4_make_x_rotation_matrix(degrees, rotation_tolerance):
    radians = matrices.degrees_to_radians(degrees)
    opg_matrix = jnp.array(opg.mat4_make_x_rotation(degrees))
    final_col = jnp.ravel(opg_matrix[:, -1])
    final_row = jnp.ravel(opg_matrix[-1, :])
    expected_final_col_and_row = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    assert jnp.allclose(final_col, expected_final_col_and_row)
    assert jnp.allclose(final_row, expected_final_col_and_row)
    opg_matrix = jnp.transpose(opg_matrix[:-1, :-1])
    expected_matrix = matrices.get_x_rotation_matrix_3d(radians)
    print(opg_matrix)
    print(expected_matrix)
    assert jnp.allclose(opg_matrix, expected_matrix, atol=rotation_tolerance)

@pytest.mark.parametrize('degrees', [20.0, 30.0, 180.0, 0.0, 360.0])
def test_opg_mat4_make_y_rotation_matrix(degrees, rotation_tolerance):
    radians = matrices.degrees_to_radians(degrees)
    opg_matrix = jnp.array(opg.mat4_make_y_rotation(degrees))
    final_col = jnp.ravel(opg_matrix[:, -1])
    final_row = jnp.ravel(opg_matrix[-1, :])
    expected_final_col_and_row = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    assert jnp.allclose(final_col, expected_final_col_and_row)
    assert jnp.allclose(final_row, expected_final_col_and_row)
    opg_matrix = jnp.transpose(opg_matrix[:-1, :-1])
    expected_matrix = matrices.get_y_rotation_matrix_3d(radians)
    print(opg_matrix)
    print(expected_matrix)
    assert jnp.allclose(opg_matrix, expected_matrix, atol=rotation_tolerance)

@pytest.mark.parametrize('degrees', [20.0, 30.0, 180.0, 0.0, 360.0])
def test_opg_mat4_make_z_rotation_matrix(degrees, rotation_tolerance):
    radians = matrices.degrees_to_radians(degrees)
    opg_matrix = jnp.array(opg.mat4_make_z_rotation(degrees))
    final_col = jnp.ravel(opg_matrix[:, -1])
    final_row = jnp.ravel(opg_matrix[-1, :])
    expected_final_col_and_row = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    assert jnp.allclose(final_col, expected_final_col_and_row)
    assert jnp.allclose(final_row, expected_final_col_and_row)
    opg_matrix = jnp.transpose(opg_matrix[:-1, :-1])
    expected_matrix = matrices.get_z_rotation_matrix_3d(radians)
    print(opg_matrix)
    print(expected_matrix)
    assert jnp.allclose(opg_matrix, expected_matrix, atol=rotation_tolerance)

@pytest.mark.parametrize('x_rot', [20.0])
@pytest.mark.parametrize('y_rot', [0.0])
@pytest.mark.parametrize('z_rot', [0.0])
@pytest.mark.parametrize('zoom', [1.0])
def test_opg_mat4_make_camera_model_matrix(x_rot, y_rot, z_rot, zoom):
    def get_expected():   
        rot_x_matrix = matrices.get_x_rotation_matrix_3d(matrices.degrees_to_radians(x_rot))
        rot_y_matrix = matrices.get_y_rotation_matrix_3d(matrices.degrees_to_radians(y_rot))
        rot_matrix = rot_x_matrix @ rot_y_matrix
        # OpenGL uses -z as forward. We'll use it for the NeRF math as well in order to minimize
        # the number of required transformations.
        position = jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32)
        position = rot_matrix @ position
        position = jnp.reshape(position, [3, 1])
        expected_matrix = jnp.concatenate([rot_matrix, position], axis=-1)
        expected_matrix = jnp.concatenate([
            expected_matrix, jnp.array([[0.0, 0.0, 0.0, 1.0],], dtype=jnp.float32)
        ], axis=0)
        return expected_matrix

    opg_matrix = opg.mat4_make_camera_model_matrix(x_rot, y_rot, z_rot, zoom)
    opg_matrix = jnp.transpose(opg_matrix)
    expected_matrix = get_expected()
    print(expected_matrix)
    print(opg_matrix)
    assert jnp.allclose(opg_matrix, expected_matrix)

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
