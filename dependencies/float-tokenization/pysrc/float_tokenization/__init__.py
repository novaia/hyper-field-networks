import jax
from jax import core
from jax import numpy as jnp
from jax.lib import xla_client
from jax.interpreters import xla, mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call
from functools import partial
from float_tokenization import cuda_ffi
from float_tokenization.lowering_helper import (
    _make_ir_tensor_info, _get_ir_tensor_info, _default_layouts
)

def _tokenize_abstract(samples):
    return core.ShapedArray(shape=samples.shape, dtype=jnp.uint32)

def _detokenize_abstract(tokens):
    return core.ShapedArray(shape=tokens.shape, dtype=jnp.float16)

def _tokenize_cuda_lowering_rule(ctx, samples):
    samples_type, samples_shape = _get_ir_tensor_info(samples)
    output_type, output_shape = _make_ir_tensor_info(samples_shape, 'uint32')
    opaque = cuda_ffi.make_tokenization_descriptor(5, samples_shape[0])
    out = custom_call(
        call_target_name="tokenize",
        result_types=[output_type],
        operands=[samples],
        backend_config=opaque,
        operand_layouts=_default_layouts(samples_shape),
        result_layouts=_default_layouts(output_shape)
    ).results
    return out

def _detokenize_cuda_lowering_rule(ctx, tokens):
    tokens_type, tokens_shape = _get_ir_tensor_info(tokens)
    output_type, output_shape = _make_ir_tensor_info(tokens_shape, 'fp16')
    opaque = cuda_ffi.make_tokenization_descriptor(5, tokens_shape[0])
    out = custom_call(
        call_target_name="detokenize",
        result_types=[output_type],
        operands=[tokens],
        backend_config=opaque,
        operand_layouts=_default_layouts(tokens_shape),
        result_layouts=_default_layouts(output_shape)
    ).results
    return out

for name, value in cuda_ffi.get_function_registrations().items():
    xla_client.register_custom_call_target(name, value, platform="gpu")

# Setup tokenization primitive.
_tokenize_p=core.Primitive('tokenize')
_tokenize_p.multiple_results = False
_tokenize_p.def_impl(partial(xla.apply_primitive, _tokenize_p))
mlir.register_lowering(
    prim=_tokenize_p, 
    rule=_tokenize_cuda_lowering_rule, 
    platform='gpu'
)
_tokenize_p.def_abstract_eval(_tokenize_abstract)

# Setup detokenization primitive.
_detokenize_p= core.Primitive('detokenize')
_detokenize_p.multiple_results = False
_detokenize_p.def_impl(partial(xla.apply_primitive, _detokenize_p))
mlir.register_lowering(
    prim=_detokenize_p, 
    rule=_detokenize_cuda_lowering_rule, 
    platform='gpu'
)
_detokenize_p.def_abstract_eval(_detokenize_abstract)

def tokenize(samples):
    return _tokenize_p.bind(samples)

def detokenize(tokens):
    return _detokenize_p.bind(tokens)
