#include <pybind11/pybind11.h>
#include <serde-helper/serde.h>
#include <cstdint>
#include "common.h"

namespace float_tokenization
{

template<typename T> pybind11::bytes to_pybind11_bytes(T const &descriptor) 
{
    return pybind11::bytes(serialize<T>(descriptor));
}

template<typename T> pybind11::capsule encapsulate_function(T *fn) 
{
    return pybind11::capsule(bit_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict get_function_registrations()
{
    pybind11::dict dict;
    dict["tokenize"] = encapsulate_function(launch_tokenization);
    dict["detokenize"] = encapsulate_function(launch_detokenization);
    return dict;
}

pybind11::bytes make_tokenization_descriptor(
    std::uint32_t mantissa_shift, std::uint32_t n_tokens
){
    return to_pybind11_bytes(tokenization_descriptor_t{ 
        .mantissa_shift = mantissa_shift,
        .n_tokens = n_tokens
    });
}

PYBIND11_MODULE(cuda_ffi, m)
{
    m.def("get_function_registrations", &get_function_registrations);
    m.def("make_tokenization_descriptor", &make_tokenization_descriptor);
}

} // namespace float_tokenization
