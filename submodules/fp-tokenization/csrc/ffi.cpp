#include <pybind11/pybind11.h>
#include "serde.h"
#include "tokenization.h"

namespace fp_tokenization
{

template<typename T> pybind11::bytes to_pybind11_bytes(T const &descriptor) 
{
    return pybind11::bytes(serialize<T>(descriptor));
}

template<typename T> pybind11::capsule encapsulate_function(T *fn) 
{
    return pybind11::capsule(bit_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict get_fp32_to_token_registration() 
{
    pybind11::dict dict;
    dict["fp32_to_token"] = encapsulate_function(fp32_to_token);
    return dict;
}

pybind11::dict get_token_to_fp32_registration() 
{
    pybind11::dict dict;
    dict["token_to_fp32"] = encapsulate_function(token_to_fp32);
    return dict;
}

pybind11::dict get_fp32_to_bitfield16_registration() 
{
    pybind11::dict dict;
    dict["fp32_to_bitfield16"] = encapsulate_function(fp32_to_bitfield16);
    return dict;
}

pybind11::dict get_bitfield16_to_fp32_registration() 
{
    pybind11::dict dict;
    dict["bitfield16_to_fp32"] = encapsulate_function(bitfield16_to_fp32);
    return dict;
}

pybind11::dict get_fp32_to_u8_token_registration() 
{
    pybind11::dict dict;
    dict["fp32_to_u8_token"] = encapsulate_function(fp32_to_u8_token);
    return dict;
}

pybind11::dict get_u8_token_to_fp32_registration() 
{
    pybind11::dict dict;
    dict["u8_token_to_fp32"] = encapsulate_function(u8_token_to_fp32);
    return dict;
}

pybind11::bytes make_tokenization_descriptor(std::uint32_t const n_elements)
{
    return to_pybind11_bytes(tokenization_descriptor_t { .n_elements = n_elements } );
}

PYBIND11_MODULE(cuda_ffi, m) 
{
    m.def("make_tokenization_descriptor", &make_tokenization_descriptor);

    m.def("get_fp32_to_token_registration", &get_fp32_to_token_registration);
    m.def("get_token_to_fp32_registration", &get_token_to_fp32_registration);
    m.def("get_fp32_to_token_vocab_size", &get_fp32_to_token_vocab_size);
    
    m.def("get_fp32_to_bitfield16_registration", &get_fp32_to_bitfield16_registration);
    m.def("get_bitfield16_to_fp32_registration", &get_bitfield16_to_fp32_registration);

    m.def("get_fp32_to_u8_token_registration", &get_fp32_to_u8_token_registration);
    m.def("get_u8_token_to_fp32_registration", &get_u8_token_to_fp32_registration);
    m.def("get_fp32_to_u8_token_vocab_size", &get_fp32_to_u8_token_vocab_size);
}

} // namespace fp_tokenization
