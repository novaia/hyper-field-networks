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

def _default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]

def _get_ir_tensor_info(tensor):
    tensor_type = ir.RankedTensorType(tensor.type)
    tensor_shape = tensor_type.shape
    return tensor_type, tensor_shape

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

def tokenize(batch: jax.Array):
    return _fp32_to_token_p.bind(batch)
