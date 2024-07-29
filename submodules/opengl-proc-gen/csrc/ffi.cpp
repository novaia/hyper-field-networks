#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "vector_matrix_math.h"

namespace py = pybind11;

// Helper function to convert numpy array to mat4.
void numpy_to_mat4(py::array_t<float> np_array, mat4 m) 
{
    auto r = np_array.unchecked<2>();
    for (int i = 0; i < 4; i++) 
    {
        for (int j = 0; j < 4; j++) 
        {
            m[i][j] = r(i, j);
        }
    }
}

// Helper function to convert mat4 to numpy array.
py::array_t<float> mat4_to_numpy(const mat4 m) 
{
    auto result = py::array_t<float>({4, 4});
    auto r = result.mutable_unchecked<2>();
    for(int i = 0; i < 4; ++i) 
    {
        for(int j = 0; j < 4; ++j) 
        {
            r(i, j) = m[i][j];
        }
    }
    return result;
}

PYBIND11_MODULE(cpp_ffi, m) 
{
    m.def(
        "mat4_mul", 
        [](py::array_t<float> a, py::array_t<float> b)
        {
            mat4 mat_a, mat_b, result;
            numpy_to_mat4(a, mat_a);
            numpy_to_mat4(b, mat_b);
            mat4_mul(mat_a, mat_b, result);
            return mat4_to_numpy(result);
        }
    );

    m.def(
        "mat4_make_identity", 
        []() 
        {
            mat4 result;
            mat4_make_identity(result);
            return mat4_to_numpy(result);
        }
    );

    m.def(
        "mat4_make_x_rotation",
        [](float degrees)
        {
            mat4 result;
            mat4_make_x_rotation(degrees, result);
            return mat4_to_numpy(result);
        }
    );

    m.def(
        "mat4_make_y_rotation",
        [](float degrees)
        {
            mat4 result;
            mat4_make_y_rotation(degrees, result);
            return mat4_to_numpy(result);
        }
    );
    
    m.def(
        "mat4_make_z_rotation",
        [](float degrees)
        {
            mat4 result;
            mat4_make_z_rotation(degrees, result);
            return mat4_to_numpy(result);
        }
    );
}
