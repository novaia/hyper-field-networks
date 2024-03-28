import jax
from jax.interpreters import mlir, xla
from jax.lib import xla_client
from ngp_volume_rendering.morton3d import abstract, lowering
from ngp_volume_rendering import cuda_ffi
from functools import partial

# Register GPU XLA custom calls.
for name, value in cuda_ffi.get_morton_3d_registrations().items():
    xla_client.register_custom_call_target(name, value, platform="gpu")

morton3d_p = jax.core.Primitive("morton_3d")
morton3d_p.multiple_results = False
morton3d_p.def_impl(partial(xla.apply_primitive, morton3d_p))
morton3d_p.def_abstract_eval(abstract.morton3d_abstract)

morton3d_invert_p = jax.core.Primitive("morton_3d_invert")
morton3d_invert_p.multiple_results = False
morton3d_invert_p.def_impl(partial(xla.apply_primitive, morton3d_invert_p))
morton3d_invert_p.def_abstract_eval(abstract.morton3d_invert_abstract)

# Register mlir lowering rules.
mlir.register_lowering(
    prim=morton3d_p,
    rule=lowering.morton3d_lowering_rule,
    platform="gpu",
)
mlir.register_lowering(
    prim=morton3d_invert_p,
    rule=lowering.morton3d_invert_lowering_rule,
    platform="gpu",
)
