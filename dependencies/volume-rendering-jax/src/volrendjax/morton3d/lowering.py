from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call
from volrendjax import volrendutils_cuda
from volrendjax.lowering_helper import \
    _default_layouts, _get_ir_tensor_info, _make_ir_tensor_info

def morton3d_lowering_rule(ctx: mlir.LoweringRule, xyzs: ir.Value):
    _, xyzs_shape = _get_ir_tensor_info(xyzs)
    
    operands = [xyzs]
    operand_shapes = [xyzs_shape]
    
    length, _ = xyzs_shape
    opaque = volrendutils_cuda.make_morton3d_descriptor(length)
    
    idcs_type, idcs_shape = _make_ir_tensor_info((length,), 'uint32')

    result_types = [idcs_type]
    result_shapes = [idcs_shape]

    out = custom_call(
        call_target_name="morton3d",
        result_types=result_types,
        operands=operands,
        backend_config=opaque,
        operand_layouts=_default_layouts(*operand_shapes),
        result_layouts=_default_layouts(*result_shapes)
    ).results
    return out

def morton3d_invert_lowering_rule(ctx: mlir.LoweringRule, idcs: ir.Value):
    _, idcs_shape = _get_ir_tensor_info(idcs)
    
    operands = [idcs]
    operand_shapes = [idcs_shape]

    length, = idcs_shape
    opaque = volrendutils_cuda.make_morton3d_descriptor(length)
    
    xyzs_type, xyzs_shape = _make_ir_tensor_info((length, 3), 'uint32')
    
    result_types = [xyzs_type]
    result_shapes = [xyzs_shape]

    out = custom_call(
        call_target_name="morton3d_invert",
        result_types=result_types,
        operands=operands,
        backend_config=opaque,
        operand_layouts=_default_layouts(*operand_shapes),
        result_layouts=_default_layouts(*result_shapes)
    ).results
    return out
