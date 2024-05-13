from fields.common import matrices
import numpy as np

def main():
    test_inv_mat = np.array([
        [0.990268, 0.130780, -0.047600, 0.000000],
        [0.000000, 0.342020, 0.939693, 0.000000],
        [0.139173, -0.930548, 0.338692, 0.000000],
        [0.000000, 1.200000, 0.000000, 1.000000],
    ])
    print(test_inv_mat)
    inv_mat = np.linalg.inv(test_inv_mat)
    print(inv_mat)
    exit()

    x_rot = np.radians(-65.0)
    y_rot = np.radians(0.0)

    x_mat = matrices.get_x_rotation_matrix_3d(x_rot)
    print("x rotation matrix:")
    print(x_mat.T)
    y_mat = matrices.get_y_rotation_matrix_3d(y_rot)
    print("y rotation matrix:")
    print(y_mat.T)
    rot_mat =y_mat @ x_mat
    print("combined rotation matrix:")
    print(rot_mat.T)

    test_mat = np.array([
        [0.990268, 0.130780, -0.047600, 0.000000],
        [0.000000, 0.342020, 0.939693, 0.000000],
        [0.139173, -0.930548, 0.338692, 0.000000],
        [0.000000, 1.200000, 0.000000, 1.000000],
    ])
    print("test matrix:")
    print(test_mat)
    processed = matrices.process_3x4_transformation_matrix(test_mat.T, 1.0)
    print("processed matrix:")
    print(processed)
    print("non transposed combined:")
    print(rot_mat)
    alt_processed = np.swapaxes(np.expand_dims(test_mat, axis=0), -1, -2)
    print("alt processed matrix:")
    print(alt_processed)
if __name__ == '__main__':
    main()
