from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call
from ngp_volume_rendering import cuda_ffi 
from ngp_volume_rendering.lowering_helper import \
    _default_layouts, _get_ir_tensor_info, _make_ir_tensor_info

def packbits_lowering_rule(
    ctx: mlir.LoweringRule,
    density_threshold: ir.Value, density_grid: ir.Value,
):
    _, density_threshold_shape = _get_ir_tensor_info(density_threshold)
    _, density_grid_shape = _get_ir_tensor_info(density_grid)

    operands = [density_threshold, density_grid]
    operand_shapes = [density_threshold_shape, density_grid_shape]
    
    n_bits = density_grid_shape[0]
    n_bytes = n_bits // 8
    opaque = cuda_ffi.make_packbits_descriptor(n_bytes)
    
    occupied_mask_type, occupied_mask_shape = _make_ir_tensor_info((n_bits,), 'bool')
    occupancy_bitfield_type, occupancy_bitfield_shape = _make_ir_tensor_info((n_bytes,), 'uint8')
    
    result_types = [occupied_mask_type, occupancy_bitfield_type]
    result_shapes = [occupied_mask_shape, occupancy_bitfield_shape]

    out = custom_call(
        call_target_name="pack_density_into_bits",
        result_types=result_types,
        operands=operands,
        backend_config=opaque,
        operand_layouts=_default_layouts(*operand_shapes),
        result_layouts=_default_layouts(*result_shapes),
    ).results
    return out
