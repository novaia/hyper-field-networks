import jax
from jax.interpreters import mlir, xla
from jax.lib import xla_client
from volrendjax.marching import abstract, lowering
from volrendjax import volrendutils_cuda
from functools import partial

# Register GPU XLA custom calls.
for name, value in volrendutils_cuda.get_marching_registrations().items():
    xla_client.register_custom_call_target(name, value, platform="gpu")

march_rays_p = jax.core.Primitive("march_rays")
march_rays_p.multiple_results = True
march_rays_p.def_impl(partial(xla.apply_primitive, march_rays_p))
march_rays_p.def_abstract_eval(abstract.march_rays_abstract)

march_rays_inference_p = jax.core.Primitive("march_rays_inference")
march_rays_inference_p.multiple_results = True
march_rays_inference_p.def_impl(partial(xla.apply_primitive, march_rays_inference_p))
march_rays_inference_p.def_abstract_eval(abstract.march_rays_inference_abstract)

# Register mlir lowering rules.
mlir.register_lowering(
    prim=march_rays_p,
    rule=lowering.march_rays_lowering_rule,
    platform="gpu",
)
mlir.register_lowering(
    prim=march_rays_inference_p,
    rule=lowering.march_rays_inference_lowering_rule,
    platform="gpu",
)
