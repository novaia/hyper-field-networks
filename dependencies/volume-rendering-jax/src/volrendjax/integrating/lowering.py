from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call
from volrendjax import volrendutils_cuda
from volrendjax.lowering_helper import \
    _default_layouts, _get_ir_tensor_info, _make_ir_tensor_info

def integrate_rays_lowering_rule(
    ctx: mlir.LoweringRuleContext,
    rays_sample_start_idx: ir.Value, rays_n_samples: ir.Value, bgs: ir.Value, dss: ir.Value,
    z_vals: ir.Value, drgbs: ir.Value,
):
    _, rays_sample_start_idx_shape = _get_ir_tensor_info(rays_sample_start_idx)
    _, rays_n_samples_shape = _get_ir_tensor_info(rays_n_samples)
    _, bgs_shape = _get_ir_tensor_info(bgs)
    _, dss_shape = _get_ir_tensor_info(dss)
    _, z_vals_shape = _get_ir_tensor_info(z_vals)
    _, drgbs_shape = _get_ir_tensor_info(drgbs)

    operands = [rays_sample_start_idx, rays_n_samples, bgs, dss, z_vals, drgbs]
    operand_shapes = [
        rays_sample_start_idx_shape, rays_n_samples_shape, 
        bgs_shape, dss_shape, z_vals_shape, drgbs_shape
    ]
    
    n_rays, = rays_sample_start_idx_shape
    total_samples, = z_vals_shape
    opaque = volrendutils_cuda.make_integrating_descriptor(n_rays, total_samples)

    measured_batch_size_type, measured_batch_size_shape = _make_ir_tensor_info((1,), 'uint32')
    final_rgbds_type, final_rgbds_shape = _make_ir_tensor_info((n_rays, 4), 'fp32')
    final_opacities_type, final_opacities_shape = _make_ir_tensor_info((n_rays,), 'fp32')
    
    result_types = [measured_batch_size_type, final_rgbds_type, final_opacities_type]
    result_shapes = [measured_batch_size_shape, final_rgbds_shape, final_opacities_shape]
    
    out = custom_call(
        call_target_name="integrate_rays",
        result_types=result_types,
        operands=operands,
        backend_config=opaque,
        operand_layouts=_default_layouts(*operand_shapes),
        result_layouts=_default_layouts(*result_shapes),
    ).results
    return out

def integrate_rays_backward_lowring_rule(
    ctx: mlir.LoweringRuleContext,
    rays_sample_start_idx: ir.Value, rays_n_samples: ir.Value,
    # Original inputs.
    bgs: ir.Value, dss: ir.Value, z_vals: ir.Value, drgbs: ir.Value,
    # Original outputs.
    final_rgbds: ir.Value, final_opacities: ir.Value,
    # Gradient inputs.
    dL_dfinal_rgbds: ir.Value,
    # Static argument.
    near_distance: float,
):
    _, rays_sample_start_idx_shape = _get_ir_tensor_info(rays_sample_start_idx)
    _, rays_n_samples_shape = _get_ir_tensor_info(rays_n_samples)
    _, bgs_shape = _get_ir_tensor_info(bgs)
    _, dss_shape = _get_ir_tensor_info(dss)
    _, z_vals_shape = _get_ir_tensor_info(z_vals)
    _, drgbs_shape = _get_ir_tensor_info(drgbs)
    _, final_rgbds_shape = _get_ir_tensor_info(final_rgbds)
    _, final_opacities_shape = _get_ir_tensor_info(final_opacities)
    _, dl_dfinal_rgbds_shape = _get_ir_tensor_info(dL_dfinal_rgbds)

    operands = [
        rays_sample_start_idx, rays_n_samples, bgs, dss, z_vals, drgbs, final_rgbds,
        final_opacities, dL_dfinal_rgbds,
    ]
    operand_shapes = [
        rays_sample_start_idx_shape, rays_n_samples_shape, bgs_shape, dss_shape, z_vals_shape,
        drgbs_shape, final_rgbds_shape, final_opacities_shape, dl_dfinal_rgbds_shape
    ]

    n_rays, = rays_sample_start_idx_shape
    total_samples, = z_vals_shape
    opaque = volrendutils_cuda.make_integrating_backward_descriptor(
        n_rays, total_samples, near_distance
    )

    dl_dbgs_type, dl_dbgs_shape = _make_ir_tensor_info((n_rays, 3), 'fp32')
    dl_dz_vals_type, dl_dz_vals_shape = _make_ir_tensor_info((total_samples,), 'fp32')
    dl_ddrgbs_type, dl_ddrgbs_shape = _make_ir_tensor_info((total_samples, 4), 'fp32')

    result_types = [dl_dbgs_type, dl_dz_vals_type, dl_ddrgbs_type]
    result_shapes = [dl_dbgs_shape, dl_dz_vals_shape, dl_ddrgbs_shape]

    out = custom_call(
        call_target_name="integrate_rays_backward",
        result_types=result_types,
        operands=operands,
        backend_config=opaque,
        operand_layouts=_default_layouts(*operand_shapes),
        result_layouts=_default_layouts(*result_shapes),
    ).results
    return out

def integrate_rays_inference_lowering_rule(
    ctx: mlir.LoweringRuleContext,
    rays_bg: ir.Value, rays_rgbd: ir.Value, rays_T: ir.Value, n_samples: ir.Value, 
    indices: ir.Value, dss: ir.Value, z_vals: ir.Value, drgbs: ir.Value,
):
    _, rays_bg_shape = _get_ir_tensor_info(rays_bg)
    _, rays_rgbd_shape = _get_ir_tensor_info(rays_rgbd)
    _, rays_T_shape = _get_ir_tensor_info(rays_T)
    _, n_samples_shape = _get_ir_tensor_info(n_samples)
    _, indices_shape = _get_ir_tensor_info(indices)
    _, dss_shape = _get_ir_tensor_info(dss)
    _, z_vals_shape = _get_ir_tensor_info(z_vals)
    _, drgbs_shape = _get_ir_tensor_info(drgbs)

    operands = [rays_bg, rays_rgbd, rays_T, n_samples, indices, dss, z_vals, drgbs]
    operand_shapes = [
        rays_bg_shape, rays_rgbd_shape, rays_T_shape, n_samples_shape, indices_shape,
        dss_shape, z_vals_shape, drgbs_shape
    ]
    
    n_total_rays, _ = rays_rgbd_shape
    n_rays, march_steps_cap = dss_shape
    opaque = volrendutils_cuda.make_integrating_inference_descriptor(
        n_total_rays, n_rays, march_steps_cap
    )

    terminate_cnt_type, terminate_cnt_shape = _make_ir_tensor_info((1,), 'uint32')
    terminated_type, terminated_shape = _make_ir_tensor_info((n_rays,), 'bool')
    rays_rgbd_out_type, rays_rgbd_out_shape = _make_ir_tensor_info((n_rays, 4), 'fp32')
    rays_T_out_type, rays_T_out_shape = _make_ir_tensor_info((n_rays,), 'fp32')
    
    result_types = [terminate_cnt_type, terminated_type, rays_rgbd_out_type, rays_T_out_type]
    result_shapes = [
        terminated_cnt_shape, terminated_shape, rays_rgbd_out_shape, rays_T_out_shape
    ]

    out = custom_call(
        call_target_name="integrate_rays_inference",
        result_types=result_types,
        operands=operands,
        backend_config=opaque,
        operand_layouts=_default_layouts(*operand_shapes),
        result_layouts=_default_layouts(*result_shapes)
    ).results
    return out
