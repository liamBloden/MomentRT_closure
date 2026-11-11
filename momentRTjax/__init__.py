"""momentRTjax: two-moment radiative transfer (M1 and VET closures) in JAX.

Submodules are loaded lazily so that importing the package does NOT import
JAX: `momentRTjax.devices.select_device()` must run before JAX is imported,
otherwise JAX_PLATFORMS / CUDA_VISIBLE_DEVICES are ignored.
"""
import importlib

_EXPORTS = {
    "momentRT": ("M1Config", "advance", "step_m1", "cfl_dt", "divergence_flux",
                 "divergence_flux_generic", "m1_closure", "physical_flux",
<<<<<<< HEAD
                 "source_implicit", "enforce_positivity_and_causality"),
    "vet": ("VETConfig", "fibonacci_directions", "DirectionSet", "formal_solution",
            "eddington_tensor", "step_vet", "advance_vet", "compute_vet"),
=======
                 "ap_speed_factor", "source_implicit",
                 "enforce_positivity_and_causality"),
    "vet": ("VETConfig", "fibonacci_directions", "DirectionSet", "default_dirset",
            "formal_solution", "eddington_tensor", "step_vet", "advance_vet",
            "compute_vet"),
>>>>>>> upstream/main
    "devices": ("select_device",),
    "turbulence": ("gaussian_field", "lognormal_field", "top_percentile_mask",
                   "turbulent_opacity", "random_point_sources", "random_setup",
                   "optical_depth_stats"),
}
_LOOKUP = {name: mod for mod, names in _EXPORTS.items() for name in names}


def __getattr__(name):
    if name in _LOOKUP:
        return getattr(importlib.import_module(f".{_LOOKUP[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
