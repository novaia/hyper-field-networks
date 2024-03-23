from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call
from volrendjax import volrendutils_cuda
from volrendjax.lowering_helper import \
    _default_layouts, _get_ir_tensor_info, _make_ir_tensor_info

def _march_rays_cuda_lowering_rule(
    ctx: mlir.LoweringRule,
    rays_o: ir.Value, rays_d: ir.Value, t_starts: ir.Value, t_ends: ir.Value, noises: ir.Value, 
    occupancy_bitfield: ir.Value,
    total_samples: int, diagonal_n_steps: int, K: int, G: int, bound: float, stepsize_portion: float,
):
    n_rays, _ = ir.RankedTensorType(rays_o.type).shape
    opaque = volrendutils_cuda.make_marching_descriptor(
        n_rays, total_samples, diagonal_n_steps, K, G, bound, stepsize_portion,
    )
    
    rays_o_type, rays_o_shape = _get_ir_tensor_info(rays_o)
    rays_d_type, rays_d_shape = _get_ir_tensor_info(rays_d)
    t_starts_type, t_starts_shape = _get_ir_tensor_info(t_starts)
    t_ends_type, t_ends_shape = _get_ir_tensor_info(t_ends)
    noises_type, noises_shape = _get_ir_tensor_info(noises)
    occupancy_bitfield_type, occupancy_bitfield_shape = _get_ir_tensor_info(occupancy_bitfield)
    
    operands = [rays_o, rays_d, t_starts, t_ends, noises, occupancy_bitfield]
    operand_shapes = [
        rays_o_shape, rays_d_shape, t_starts_shape, t_ends_shape, 
        noises_shape, occupancy_bitfield_shape
    ]
    
    next_sample_write_location_type, next_sample_write_location_shape = _make_ir_tensor_info((1,), 'uint32')
    num_exceeded_samples_type, num_exceeded_samples_shape = _make_ir_tensor_info((1,), 'uint32')
    ray_is_valid_type, ray_is_valid_shape = _make_ir_tensor_info((n_rays,), 'bool')
    rays_n_samples_type, rays_n_samples_shape = _make_ir_tensor_info((n_rays,), 'uint32')
    rays_sample_startidx_type, rays_sample_start_idx_shape = _make_ir_tensor_info((n_rays,), 'uint32')
    idcs_type, idcs_shape = _make_ir_tensor_info((total_samples,), 'uint32')
    xyzs_type, xyzs_shape = _make_ir_tensor_info((total_samples, 3), 'fp32')
    dirs_type, dirs_shape = _make_ir_tensor_info((total_samples, 3), 'fp32')
    dss_type, dss_shape = _make_ir_tensor_info((total_samples,), 'fp32')
    z_vals_type, z_vals_shape = _make_ir_tensor_info((total_samples,), 'fp32')
    
    result_types = [
        next_sample_write_location_type, num_exceeded_samples_type, ray_is_valid_type,
        rays_n_samples_type, rays_sample_startidx_type, idcs_type, xyzs_type, dirs_type,
        dss_type, z_vals_type
    ]
    result_shapes = [
        next_sample_write_location_shape, num_exceeded_samples_shape, ray_is_valid_shape,
        rays_n_samples_shape, rays_sample_start_idx_shape, idcs_shape, xyzs_shape, dirs_shape,
        dss_shape, z_vals_shape
    ]

    out = custom_call(
        b'march_rays',
        result_types=result_types,
        operands=operands,
        backend_config=opaque,
        operand_layouts=_default_layouts(*operand_shapes),
        result_layouts=_default_layouts(*result_shapes)
    ).results
    return out


def _march_rays_inference_lowering_rule(
    ctx: mlir.LoweringRule,
    rays_o: ir.BlockArgument, rays_d: ir.BlockArgument, t_starts: ir.BlockArgument, 
    t_ends: ir.BlockArgument, occupancy_bitfield: ir.BlockArgument, next_ray_index_in: ir.BlockArgument, 
    terminated: ir.BlockArgument, indices_in: ir.BlockArgument,
    diagonal_n_steps: int, K: int, G: int, march_steps_cap: int, bound: float, stepsize_portion: float,
):
    n_total_rays, _ = ir.RankedTensorType(rays_o.type).shape
    n_rays = ir.RankedTensorType(terminated.type).shape
    opaque = volrendutils_cuda.make_marching_inference_descriptor(
        n_total_rays, n_rays, diagonal_n_steps, K, G, march_steps_cap, bound, stepsize_portion,
    )
    
    rays_o_type, rays_o_shape = _get_ir_tensor_info(rays_o)
    rays_d_type, rays_d_shape = _get_ir_tensor_info(rays_d)
    t_starts_type, t_starts_shape = _get_ir_tensor_info(t_starts)
    t_ends_type, t_ends_shape = _get_ir_tensor_info(t_ends)
    occupancy_bitfield_type, occupancy_bitfield_shape = _get_ir_tensor_info(occupancy_bitfield)
    next_ray_index_in_type, next_ray_index_in_shape = _get_ir_tensor_info(next_ray_index_in)
    terminated_type, terminated_shape = _get_ir_tensor_info(terminated)
    indices_in_type, indices_in_shape = _get_ir_tensor_info(indices_in)

    operands = [
        rays_o, rays_d, t_starts, t_ends, occupancy_bitfield, 
        next_ray_index_in, terminated, indices_in
    ]
    operand_shapes = [
        rays_o_shape, rays_d_shape, t_starts_shape, t_ends_shape, occupancy_bitfield_shape,
        next_ray_index_ind_shape, terminated_shape, indices_in_shape
    ]
    
    next_ray_index_type, next_ray_index_shape = _make_ir_tensor_info((1,), 'uint32')
    indices_out_type, indices_out_shape = _make_ir_tensor_info((n_rays,), 'uint32')
    n_samples_type, n_samples_shape = _make_ir_tensor_info((n_rays,), 'uint32')
    t_starts_out_type, t_starts_out_shape = _make_ir_tensor_info((n_rays,), 'fp32')
    xyzs_type, xyzs_shape = _make_ir_tensor_info((n_rays, march_steps_cap, 3), 'fp32')
    dss_type, dss_shape = _make_ir_tensor_info((n_rays, march_steps_cap), 'fp32')
    z_vals_type, z_vals_shape = _make_ir_tensor_info((n_rays, march_steps_cap), 'fp32')
    
    result_types = [
        next_ray_index_type, indices_out_type, n_samples_type, t_starts_out_type,
        xyzs_type, dss_type, z_vals_type
    ]
    result_shapes = [
        next_ray_index_shape, indices_out_shape, n_samples_shape, t_starts_out_shape,
        xyzs_shape, dss_shape, z_vals_shape
    ]

    out = custom_call(
        b'march_rays_inference',
        result_types=result_types,
        operands=operands,
        backend_config=opaque,
        operand_layouts=_default_layouts(*operand_shapes),
        result_layouts=_default_layouts(*result_shapes)
    ).results
    return out
