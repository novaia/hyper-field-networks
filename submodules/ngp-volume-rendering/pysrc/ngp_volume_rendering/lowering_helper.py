from jax.interpreters.mlir import ir

def _default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]

def _get_ir_tensor_info(tensor):
    tensor_type = ir.RankedTensorType(tensor.type)
    tensor_shape = tensor_type.shape
    return tensor_type, tensor_shape

def _make_ir_tensor_info(shape, element_type: str):
    ir_element_type_map = {
        'uint8': ir.IntegerType.get_unsigned(8),
        'uint32': ir.IntegerType.get_unsigned(32),
        'bool': ir.IntegerType.get_signless(1),
        'fp32': ir.F32Type.get()
    }
    assert element_type in ir_element_type_map.keys(), (
        f'Invalid element type {element_type}. Must be one of: {element_type.keys()}'
    )
    return ir.RankedTensorType.get(shape, ir_element_type_map[element_type]), shape
