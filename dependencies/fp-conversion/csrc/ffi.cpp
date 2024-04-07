#include <pybind11/pybind11.h>
#include <cstdint>
#include <climits>
#include <cstring>
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
    
    const unsigned int mantissa_bits_to_truncate = 19;
    unsigned int token_exponent = 0;
    std::uint32_t token = 0;
    for(unsigned int i = 31; i > mantissa_bits_to_truncate; i--)
    {
        if(bits[i])
        {
            token += (1 << token_exponent);
        }
        token_exponent++;
        std::cout << bits[i];
    }
    std::cout << "\n";
    return token;
}

float token_to_float(std::uint32_t input)
{
    std::bitset<sizeof(std::uint32_t) * CHAR_BIT> token_bits(input);
    std::bitset<sizeof(float) * CHAR_BIT> float_bits(0.0f);
    unsigned int float_bits_offset = 0;
    for(int i = 31; i > -1; i--)
    {
        std::cout << token_bits[i];
        float_bits[float_bits_offset++] = token_bits[i];
    }
    std::cout << "\n";
    std::cout << float_bits << "\n";
    float output;
    memcpy(&output, &float_bits, sizeof(float));
    return output;
}

PYBIND11_MODULE(cpp_ffi, m)
{
    m.def("float_to_token", &float_to_token);
    m.def("token_to_float", &token_to_float);
}

}
