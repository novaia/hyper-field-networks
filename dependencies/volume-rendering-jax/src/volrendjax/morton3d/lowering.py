from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call
from .. import volrendutils_cuda

# helper function for mapping given shapes to their default mlir layouts
def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]

def morton3d_lowering_rule(
    ctx: mlir.LoweringRule,

    # input array
    xyzs: ir.Value,
):
    length, _ = ir.RankedTensorType(xyzs.type).shape

    opaque = volrendutils_cuda.make_morton3d_descriptor(length)

    shapes = {
        "in.xyzs": (length, 3),

        "out.idcs": (length,),
    }

    return [custom_call(
        call_target_name="morton3d",
        result_types=[
            ir.RankedTensorType.get(shapes["out.idcs"], ir.IntegerType.get_unsigned(32)),
        ],
        operands=[
            xyzs,
        ],
        backend_config=opaque,
        operand_layouts=default_layouts(
            shapes["in.xyzs"],
        ),
        result_layouts=default_layouts(
            shapes["out.idcs"],
        ),
    )]


def morton3d_invert_lowering_rule(
    ctx: mlir.LoweringRule,

    # input array
    idcs: ir.Value,
):
    length, = ir.RankedTensorType(idcs.type).shape

    opaque = volrendutils_cuda.make_morton3d_descriptor(length)

    shapes = {
        "in.idcs": (length,),

        "out.xyzs": (length, 3),
    }

    return [custom_call(
        call_target_name="morton3d_invert",
        result_types=[
            ir.RankedTensorType.get(shapes["out.xyzs"], ir.IntegerType.get_unsigned(32)),
        ],
        operands=[
            idcs,
        ],
        backend_config=opaque,
        operand_layouts=default_layouts(
            shapes["in.idcs"],
        ),
        result_layouts=default_layouts(
            shapes["out.xyzs"],
        ),
    )]
