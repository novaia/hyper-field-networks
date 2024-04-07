#include <pybind11/pybind11.h>
#include <cstdint>
#include <climits>
#include <bitset>
#include <iostream>

namespace fp_conversion
{

std::uint32_t float_to_token(float input)
{
    union
    {
        float in;
        std::int32_t out;
    } data;
    data.in = input;
    std::bitset<sizeof(float) * CHAR_BIT> bits(data.out);
    std::cout << bits[31] << " ";
    for(int i = 30; i > 30-8; i--)
    {
        std::cout << bits[i];
    }
    std::cout << " ";
    for(int i = 30-8; i > -1; i--)
    {
        std::cout << bits[i];
    }
    std::cout << "\n";
    std::cout << bits << "\n";
    return 0;
}

PYBIND11_MODULE(cpp_ffi, m)
{
    m.def("float_to_token", &float_to_token);
}

}
