#include <cstdint>

#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <serde-helper/serde.h>
#include <stdexcept>

#include "impl/volrend.h"

namespace volrendjax {

template <typename T>
pybind11::bytes to_pybind11_bytes(T const &descriptor) {
    return pybind11::bytes(serialize<T>(descriptor));
}

template <typename T>
pybind11::capsule encapsulate_function(T *fn) {
    return pybind11::capsule(bit_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

// expose gpu function
namespace {

pybind11::dict get_packbits_registrations() {
    pybind11::dict dict;
    dict["pack_density_into_bits"] = encapsulate_function(pack_density_into_bits);
    return dict;
}

pybind11::dict get_marching_registrations() {
    pybind11::dict dict;
    dict["march_rays"] = encapsulate_function(march_rays);
    dict["march_rays_inference"] = encapsulate_function(march_rays_inference);
    return dict;
}

pybind11::dict get_morton3d_registrations() {
    pybind11::dict dict;
    dict["morton3d"] = encapsulate_function(morton3d);
    dict["morton3d_invert"] = encapsulate_function(morton3d_invert);
    return dict;
}

pybind11::dict get_integrating_registrations() {
    pybind11::dict dict;
    dict["integrate_rays"] = encapsulate_function(integrate_rays);
    dict["integrate_rays_backward"] = encapsulate_function(integrate_rays_backward);
    dict["integrate_rays_inference"] = encapsulate_function(integrate_rays_inference);
    return dict;
}

PYBIND11_MODULE(volrendutils_cuda, m) {
    m.def("get_packbits_registrations", &get_packbits_registrations);
    m.def("make_packbits_descriptor",
          [](std::uint32_t const n_bytes) {
            if (n_bytes == 0) {
                throw std::runtime_error("expected n_bytes to be a positive integer, got 0");
            }
            return to_pybind11_bytes(PackbitsDescriptor{
                .n_bytes = n_bytes,
            });
          },
          "Static arguments passed to the `pack_density_into_bits` function.\n\n"
          "Args:\n"
          "    n_bytes: sum of number of byetes of all cascades of occupancy bitfields\n"
          );

    m.def("get_marching_registrations", &get_marching_registrations);
    m.def("make_marching_descriptor",
          [](std::uint32_t const n_rays
             , std::uint32_t const total_samples
             , std::uint32_t const diagonal_n_steps
             , std::uint32_t const K
             , std::uint32_t const G
             , float const bound
             , float const stepsize_portion) {
            if (K == 0) {
                throw std::runtime_error("expected K to be a positive integer, got 0");
            }
            return to_pybind11_bytes(MarchingDescriptor{
                .n_rays = n_rays,
                .total_samples = total_samples,
                .diagonal_n_steps = diagonal_n_steps,
                .K = K,
                .G = G,
                .bound = bound,
                .stepsize_portion = stepsize_portion,
            });
          },
          "Static arguments passed to the `march_rays` function.\n\n"
          "Args:\n"
          "    n_rays: number of input rays\n"
          "    total_samples: number of available slots to write generated samples to, i.e. the\n"
          "                   length of output samples array\n"
          "    diagonal_n_steps: used to calculate the length of a minimal ray marching step\n"
          "    K: total number of cascades of the occupancy bitfield\n"
          "    G: occupancy grid resolution, the paper uses 128 for every cascade\n"
          "    bound: the half length of the longest axis of the scene’s bounding box,\n"
          "           e.g. the `bound` of the bounding box [-1, 1]^3 is 1\n"
          "    stepsize_portion: next step size is calculated as t * stepsize_portion,\n"
          "                      the paper uses 1/256\n"
          );
    m.def("make_marching_inference_descriptor",
          [](std::uint32_t const n_total_rays
             , std::uint32_t const n_rays
             , std::uint32_t const diagonal_n_steps
             , std::uint32_t const K
             , std::uint32_t const G
             , std::uint32_t const march_steps_cap
             , float const bound
             , float const stepsize_portion) {
            if (K == 0) {
                throw std::runtime_error("expected K to be a positive integer, got 0");
            }
            return to_pybind11_bytes(MarchingInferenceDescriptor{
                .n_total_rays = n_total_rays,
                .n_rays = n_rays,
                .diagonal_n_steps = diagonal_n_steps,
                .K = K,
                .G = G,
                .march_steps_cap = march_steps_cap,
                .bound = bound,
                .stepsize_portion = stepsize_portion,
            });
          },
          "Static arguments passed to the `march_rays_inference` function.\n\n"
          "Args:\n"
          "    n_total_rays: total number of rays to march\n"
          "    n_rays: number of rays to march during this iteration\n"
          "    diagonal_n_steps: used to calculate the length of a minimal ray marching step\n"
          "    K: total number of cascades of the occupancy bitfield\n"
          "    G: occupancy grid resolution, the paper uses 128 for every cascade\n"
          "    march_steps_cap: max number of samples to generate for each ray\n"
          "    bound: the half length of the longest axis of the scene’s bounding box,\n"
          "           e.g. the `bound` of the bounding box [-1, 1]^3 is 1\n"
          "    stepsize_portion: next step size is calculated as t * stepsize_portion,\n"
          "                      the paper uses 1/256\n"
          );

    m.def("get_morton3d_registrations", &get_morton3d_registrations);
    m.def(
        "make_morton3d_descriptor",
        [](std::uint32_t const length) {
            return to_pybind11_bytes(Morton3DDescriptor { .length = length });
        },
        "Static arguments passed to the `morton3d` or `morton3d_invert` functions.\n\n"
        "Args:\n"
        "    length: number of entries to process\n"
        "\n"
        "Returns:\n"
        "    Serialized bytes that can be passed as the opaque parameter to `morton3d` or\n"
        "    `morton3d_invert` functions"
    );

    m.def("get_integrating_registrations", &get_integrating_registrations);
    m.def("make_integrating_descriptor",
          [](std::uint32_t const n_rays, std::uint32_t const total_samples) {
            return to_pybind11_bytes(IntegratingDescriptor{
                .n_rays = n_rays,
                .total_samples = total_samples,
            });
          },
          "Static arguments passed to the `integrate_rays` function.\n\n"
          "Args:\n"
          "    n_rays: number of rays\n"
          "    total_samples: sum of number of samples on each ray\n"
          "\n"
          "Returns:\n"
          "    Serialized bytes that can be passed as the opaque parameter to `integrate_rays`\n"
          "    or `integrate_rays_backward`"
          );
    m.def("make_integrating_backward_descriptor",
          [](std::uint32_t const n_rays, std::uint32_t const total_samples, float const near_distance) {
            return to_pybind11_bytes(IntegratingBackwardDescriptor{
                .n_rays = n_rays,
                .total_samples = total_samples,
                .near_distance = near_distance,
            });
          },
          "Static arguments passed to the `integrate_rays_backward` function.\n\n"
          "Args:\n"
          "    n_rays: number of rays\n"
          "    total_samples: sum of number of samples on each ray\n"
          "    near_distance: camera's near distance, samples behind the camera's near plane with\n"
          "                   non-negligible introduce a penalty on their densities\n"
          "\n"
          "Returns:\n"
          "    Serialized bytes that can be passed as the opaque parameter to `integrate_rays`\n"
          "    or `integrate_rays_backward`"
          );
    m.def("make_integrating_inference_descriptor",
          [](std::uint32_t const n_total_rays
             , std::uint32_t const n_rays
             , std::uint32_t const march_steps_cap) {
            return to_pybind11_bytes(IntegratingInferenceDescriptor{
                .n_total_rays = n_total_rays,
                .n_rays = n_rays,
                .march_steps_cap = march_steps_cap,
            });
          },
          "Static arguments passed to the `integrate_rays_inference`\n\n"
          "Args:\n"
          "    n_total_rays: total number of rays to march\n"
          "    n_rays: number of rays to integrate during this iteration\n"
          "    march_steps_cap: see MarchingInferenceDescriptor\n"
          );
};

}  // namespace

}  // namespace volrendjax
