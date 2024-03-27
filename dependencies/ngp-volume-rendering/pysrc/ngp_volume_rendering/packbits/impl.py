import jax
from jax.interpreters import mlir, xla
from jax.lib import xla_client
from ngp_volume_rendering.packbits import abstract, lowering
from ngp_volume_rendering import cuda_ffi 
from functools import partial

# Register GPU XLA custom calls.
for name, value in cuda_ffi.get_packbits_registrations().items():
    xla_client.register_custom_call_target(name, value, platform="gpu")

packbits_p = jax.core.Primitive("packbits")
packbits_p.multiple_results = True
packbits_p.def_impl(partial(xla.apply_primitive, packbits_p))
packbits_p.def_abstract_eval(abstract.pack_density_into_bits_abstract)

mlir.register_lowering(
    prim=packbits_p,
    rule=lowering.packbits_lowering_rule,
    platform="gpu",
)
