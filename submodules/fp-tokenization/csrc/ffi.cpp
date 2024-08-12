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

pybind11::bytes make_tokenization_descriptor(std::uint32_t const n_elements)
{
    return to_pybind11_bytes(tokenization_descriptor_t { .n_elements = n_elements } );
}

PYBIND11_MODULE(cuda_ffi, m) 
{
    m.def("get_fp32_to_token_registration", &get_fp32_to_token_registration);
    m.def("make_tokenization_descriptor", &make_tokenization_descriptor);
}


} // namespace fp_tokenization
