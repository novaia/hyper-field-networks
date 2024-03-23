from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call
from volrendjax import volrendutils_cuda

def _default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]

def get_ir_tensor_info(tensor):
    tensor_type = ir.RankedTensorType(tensor.type)
    tensor_shape = tensor_type.shape
    return tensor_type, tensor_shape

def make_ir_tensor_info(shape, element_type: str):
    ir_element_type_map = {
        'uint32': ir.IntegerType.get_unsigned(32),
        'bool': ir.IntegerType.get_signless(1),
        'fp32': ir.F32Type.get()
    }
    assert element_type in ir_element_type_map.keys(), (
        f'Invalid element type {element_type}. Must be one of: {element_type.keys()}'
    )
    return ir.RankedTensorType.get(shape, ir_element_type_map[element_type]), shape

def _march_rays_cuda_lowering_rule(
    ctx: mlir.LoweringRule,
    rays_o: ir.Value, rays_d: ir.Value, t_starts: ir.Value, t_ends: ir.Value, 
    noises: ir.Value, occupancy_bitfield: ir.Value,
    total_samples: int, diagonal_n_steps: int, K: int, G: int, bound: float,
    stepsize_portion: float,
):
    n_rays, _ = ir.RankedTensorType(rays_o.type).shape
    opaque = volrendutils_cuda.make_marching_descriptor(
        n_rays, total_samples, diagonal_n_steps, K, G, bound, stepsize_portion,
    )
    
    rays_o_type, rays_o_shape = get_ir_tensor_info(rays_o)
    rays_d_type, rays_d_shape = get_ir_tensor_info(rays_d)
    t_starts_type, t_starts_shape = get_ir_tensor_info(t_starts)
    t_ends_type, t_ends_shape = get_ir_tensor_info(t_ends)
    noises_type, noises_shape = get_ir_tensor_info(noises)
    occupancy_bitfield_type, occupancy_bitfield_shape = get_ir_tensor_info(occupancy_bitfield)
    
    operands = [noises, rays_d, t_starts, t_ends, noises, occupancy_bitfield]
    operand_shapes = [
        rays_o_shape, rays_d_shape, t_starts_shape, t_ends_shape, 
        noises_shape, occupancy_bitfield_shape
    ]
    
    next_sample_write_location_type, next_sample_write_location_shape = make_ir_tensor_info((1,), 'uint32')
    num_exceeded_samples_type, num_exceeded_samples_shape = make_ir_tensor_info((1,), 'uint32')
    ray_is_valid_type, ray_is_valid_shape = make_ir_tensor_info((n_rays,), 'bool')
    rays_n_samples_type, rays_n_samples_shape = make_ir_tensor_info((n_rays,), 'uint32')
    rays_sample_startidx_type, rays_sample_start_idx_shape = make_ir_tensor_info((n_rays,), 'uint32')
    idcs_type, idcs_shape = make_ir_tensor_info((total_samples,), 'uint32')
    xyzs_type, xyzs_shape = make_ir_tensor_info((total_samples, 3), 'fp32')
    dirs_type, dirs_shape = make_ir_tensor_info((total_samples, 3), 'fp32')
    dss_type, dss_shape = make_ir_tensor_info((total_samples,), 'fp32')
    z_vals_type, z_vals_shape = make_ir_tensor_info((total_samples,), 'fp32')
    
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


def march_rays_inference_lowering_rule(
    ctx: mlir.LoweringRule,

    # arrays
    rays_o: ir.BlockArgument,
    rays_d: ir.BlockArgument,
    t_starts: ir.BlockArgument,
    t_ends: ir.BlockArgument,
    occupancy_bitfield: ir.BlockArgument,
    next_ray_index_in: ir.BlockArgument,
    terminated: ir.BlockArgument,
    indices_in: ir.BlockArgument,

    # static args
    diagonal_n_steps: int,
    K: int,
    G: int,
    march_steps_cap: int,
    bound: float,
    stepsize_portion: float,
):
    (n_total_rays, _), (n_rays,) = ir.RankedTensorType(rays_o.type).shape, ir.RankedTensorType(terminated.type).shape

    opaque = volrendutils_cuda.make_marching_inference_descriptor(
        n_total_rays,
        n_rays,
        diagonal_n_steps,
        K,
        G,
        march_steps_cap,
        bound,
        stepsize_portion,
    )

    shapes = {
        "in.rays_o": (n_total_rays, 3),
        "in.rays_d": (n_total_rays, 3),
        "in.t_starts": (n_total_rays,),
        "in.t_ends": (n_total_rays,),
        "in.occupancy_bitfield": (K*G*G*G//8,),
        "in.next_ray_index_in": (1,),
        "in.terminated": (n_rays,),
        "in.indices_in": (n_rays,),

        "out.next_ray_index": (1,),
        "out.indices_out": (n_rays,),
        "out.n_samples": (n_rays,),
        "out.t_starts": (n_rays,),
        "out.xyzs": (n_rays, march_steps_cap, 3),
        "out.dss": (n_rays, march_steps_cap),
        "out.z_vals": (n_rays, march_steps_cap),
    }

    out = custom_call(
        b'march_rays_inference',
        result_types=[
            ir.RankedTensorType.get(shapes["out.next_ray_index"], ir.IntegerType.get_unsigned(32)),
            ir.RankedTensorType.get(shapes["out.indices_out"], ir.IntegerType.get_unsigned(32)),
            ir.RankedTensorType.get(shapes["out.n_samples"], ir.IntegerType.get_unsigned(32)),
            ir.RankedTensorType.get(shapes["out.t_starts"], ir.F32Type.get()),
            ir.RankedTensorType.get(shapes["out.xyzs"], ir.F32Type.get()),
            ir.RankedTensorType.get(shapes["out.dss"], ir.F32Type.get()),
            ir.RankedTensorType.get(shapes["out.z_vals"], ir.F32Type.get()),
        ],
        operands=[
            rays_o,
            rays_d,
            t_starts,
            t_ends,
            occupancy_bitfield,
            next_ray_index_in,
            terminated,
            indices_in,
        ],
        backend_config=opaque,
        operand_layouts=_default_layouts(
            shapes["in.rays_o"],
            shapes["in.rays_d"],
            shapes["in.t_starts"],
            shapes["in.t_ends"],
            shapes["in.occupancy_bitfield"],
            shapes["in.next_ray_index_in"],
            shapes["in.terminated"],
            shapes["in.indices_in"],
        ),
        result_layouts=_default_layouts(
            shapes["out.next_ray_index"],
            shapes["out.indices_out"],
            shapes["out.n_samples"],
            shapes["out.t_starts"],
            shapes["out.xyzs"],
            shapes["out.dss"],
            shapes["out.z_vals"],
        ),
    ).results
    return out
