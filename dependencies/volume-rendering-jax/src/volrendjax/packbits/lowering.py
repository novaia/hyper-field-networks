from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call
from .. import volrendutils_cuda

# helper function for mapping given shapes to their default mlir layouts
def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]

def packbits_lowering_rule(
    ctx: mlir.LoweringRule,
    # input array
    density_threshold: ir.Value,
    density_grid: ir.Value,
):
    n_bits = ir.RankedTensorType(density_grid.type).shape[0]
    n_bytes = n_bits // 8

    opaque = volrendutils_cuda.make_packbits_descriptor(n_bytes)

    shapes = {
        "in.density_threshold": (n_bits,),
        "in.density_grid": (n_bits,),

        "out.occupied_mask": (n_bits,),
        "out.occupancy_bitfield": (n_bytes,),
    }

    out = custom_call(
        call_target_name="pack_density_into_bits",
        result_types = [
            ir.RankedTensorType.get(shapes["out.occupied_mask"], ir.IntegerType.get_signless(1)),
            ir.RankedTensorType.get(shapes["out.occupancy_bitfield"], ir.IntegerType.get_unsigned(8)),
        ],
        operands=[
            density_threshold,
            density_grid,
        ],
        backend_config=opaque,
        operand_layouts=default_layouts(
            shapes["in.density_threshold"],
            shapes["in.density_grid"],
        ),
        result_layouts=default_layouts(
            shapes["out.occupied_mask"],
            shapes["out.occupancy_bitfield"],
        ),
    )
    return out
