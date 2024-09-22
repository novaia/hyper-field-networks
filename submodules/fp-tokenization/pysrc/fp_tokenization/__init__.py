import jax
from jax import numpy as jnp
from jax.core import ShapedArray
from jax.lib import xla_client
from jax.interpreters import mlir, xla
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call
from functools import partial
from fp_tokenization import cuda_ffi
from fp_tokenization.lowering_helper import \
    _default_layouts, _get_ir_tensor_info, _make_ir_tensor_info

def _fp32_to_token_abstract(batch: jax.Array):
    return (ShapedArray(shape=batch.shape, dtype=jnp.uint32))

def _token_to_fp32_abstract(batch: jax.Array):
    return (ShapedArray(shape=batch.shape, dtype=jnp.float32))

def _fp32_to_bitfield16_abstract(batch: jax.Array):
    return (ShapedArray(shape=(batch.shape[-1]*16,), dtype=jnp.uint32))

def _bitfield16_to_fp32_abstract(batch: jax.Array):
    return (ShapedArray(shape=(int(batch.shape[-1]//16),), dtype=jnp.float32))

def _fp32_to_token_lowering_rule(ctx: mlir.LoweringRuleContext, batch: ir.Value):
    _, batch_shape = _get_ir_tensor_info(batch)
    out_type, _ = _make_ir_tensor_info(batch_shape, 'uint32')
    
    opaque = cuda_ffi.make_tokenization_descriptor(batch_shape[-1])

    out = custom_call(
        call_target_name='fp32_to_token',
        result_types=[out_type],
        operands=[batch],
        backend_config=opaque,
        operand_layouts=_default_layouts(*[batch_shape]),
        result_layouts=_default_layouts(*[batch_shape])
    ).results
    return out

def _token_to_fp32_lowering_rule(ctx: mlir.LoweringRuleContext, batch: ir.Value):
    _, batch_shape = _get_ir_tensor_info(batch)
    out_type, _ = _make_ir_tensor_info(batch_shape, 'fp32')
    
    opaque = cuda_ffi.make_tokenization_descriptor(batch_shape[-1])

    out = custom_call(
        call_target_name='token_to_fp32',
        result_types=[out_type],
        operands=[batch],
        backend_config=opaque,
        operand_layouts=_default_layouts(*[batch_shape]),
        result_layouts=_default_layouts(*[batch_shape])
    ).results
    return out

def _fp32_to_bitfield16_lowering_rule(ctx: mlir.LoweringRuleContext, batch: ir.Value):
    _, batch_shape = _get_ir_tensor_info(batch)
    out_type, out_shape = _make_ir_tensor_info((batch_shape[-1]*16,), 'uint32')
    
    opaque = cuda_ffi.make_tokenization_descriptor(batch_shape[-1])

    out = custom_call(
        call_target_name='fp32_to_bitfield16',
        result_types=[out_type],
        operands=[batch],
        backend_config=opaque,
        operand_layouts=_default_layouts(*[batch_shape]),
        result_layouts=_default_layouts(*[out_shape])
    ).results
    return out

def _bitfield16_to_fp32_lowering_rule(ctx: mlir.LoweringRuleContext, batch: ir.Value):
    _, batch_shape = _get_ir_tensor_info(batch)
    out_type, out_shape = _make_ir_tensor_info((int(batch_shape[-1]//16),), 'fp32')
    
    opaque = cuda_ffi.make_tokenization_descriptor(out_shape[-1])

    out = custom_call(
        call_target_name='bitfield16_to_fp32',
        result_types=[out_type],
        operands=[batch],
        backend_config=opaque,
        operand_layouts=_default_layouts(*[batch_shape]),
        result_layouts=_default_layouts(*[out_shape])
    ).results
    return out

# Define and lower fp32_to_token primitive.
for name, value in cuda_ffi.get_fp32_to_token_registration().items():
    xla_client.register_custom_call_target(name, value, platform='gpu')
_fp32_to_token_p = jax.core.Primitive('fp32_to_token')
_fp32_to_token_p.multiple_results = False
_fp32_to_token_p.def_impl(partial(xla.apply_primitive, _fp32_to_token_p))
_fp32_to_token_p.def_abstract_eval(_fp32_to_token_abstract)
mlir.register_lowering(
    prim=_fp32_to_token_p,
    rule=_fp32_to_token_lowering_rule,
    platform='gpu',
)

# Define and lower token_to_fp23 primitive. 
for name, value in cuda_ffi.get_token_to_fp32_registration().items():
    xla_client.register_custom_call_target(name, value, platform='gpu')
_token_to_fp32_p= jax.core.Primitive('token_to_fp32')
_token_to_fp32_p.multiple_results = False
_token_to_fp32_p.def_impl(partial(xla.apply_primitive, _token_to_fp32_p))
_token_to_fp32_p.def_abstract_eval(_token_to_fp32_abstract)
mlir.register_lowering(
    prim=_token_to_fp32_p,
    rule=_token_to_fp32_lowering_rule,
    platform='gpu',
)

# Define and lower fp32_to_bitfield16 primitive.
for name, value in cuda_ffi.get_fp32_to_bitfield16_registration().items():
    xla_client.register_custom_call_target(name, value, platform='gpu')
_fp32_to_bitfield16_p = jax.core.Primitive('fp32_to_bitfield16')
_fp32_to_bitfield16_p.multiple_results = False
_fp32_to_bitfield16_p.def_impl(partial(xla.apply_primitive, _fp32_to_bitfield16_p))
_fp32_to_bitfield16_p.def_abstract_eval(_fp32_to_bitfield16_abstract)
mlir.register_lowering(
    prim=_fp32_to_bitfield16_p,
    rule=_fp32_to_bitfield16_lowering_rule,
    platform='gpu',
)

# Define and lower bitfield16_to_fp32_primitive.
for name, value in cuda_ffi.get_bitfield16_to_fp32_registration().items():
    xla_client.register_custom_call_target(name, value, platform='gpu')
_bitfield16_to_fp32_p = jax.core.Primitive('bitfield16_to_fp32')
_bitfield16_to_fp32_p.multiple_results = False
_bitfield16_to_fp32_p.def_impl(partial(xla.apply_primitive, _bitfield16_to_fp32_p))
_bitfield16_to_fp32_p.def_abstract_eval(_bitfield16_to_fp32_abstract)
mlir.register_lowering(
    prim=_bitfield16_to_fp32_p,
    rule=_bitfield16_to_fp32_lowering_rule,
    platform='gpu',
)

def tokenize(batch: jax.Array):
    return _fp32_to_token_p.bind(batch)

def detokenize(batch: jax.Array):
    return _token_to_fp32_p.bind(batch)

def fp32_to_bitfield16(batch: jax.Array):
    return _fp32_to_bitfield16_p.bind(batch)

@jax.jit
def bitfield16_to_fp32(batch: jax.Array):
    return _bitfield16_to_fp32_p.bind(batch)

def get_vocab_size():
    return cuda_ffi.get_fp32_to_token_vocab_size()
