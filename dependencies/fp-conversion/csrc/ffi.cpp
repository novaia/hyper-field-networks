#include <pybind11/pybind11.h>
#include <cstdint>
#include <iostream>

namespace fp_conversion
{

std::uint32_t float_to_token(float input)
{
    std::cout << input << "\n";
    return 0;
}

PYBIND11_MODULE(cpp_ffi, m)
{
    m.def("float_to_token", &float_to_token);
}

}
