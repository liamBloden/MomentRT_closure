# turbulence.py
# Synthetic "turbulent" (correlated log-normal) fields and random source
# layouts, ported from Ray-trax/ray_trax/create_turbulent_3D.py and adapted
# to the moment solver's inputs (kappa_a, kappa_t, E_eq, S_E, U0).
#
# Differences from the Ray-trax version:
#   * the Gaussian field is built by filtering real white noise in k-space,
#     so it is exactly real (the original took .real of a non-Hermitian field);
#   * the correlation scale is given as a length in cells (`corr_length`),
#     not as a peak frequency;
#   * a power-law spectrum is available next to the log-Gaussian one;
#   * helpers return fields in the shapes/dtypes the solver expects.
#
# The field generators and random_point_sources are pure JAX and jit-able
# (shape/spectrum args are static). dense_point_sources (data-dependent number
# of stars) and save_setup / load_setup run on the host with NumPy.
#
# Note on `corr_length`: it is the wavelength at the peak of P(k). In 3D the
# number of modes grows as k^2, so the variance sits at shorter wavelengths
# and the structures are much smaller than the name suggests: the
# autocorrelation of the field falls to 0.5 at a lag of ~corr_length / 7.5
# cells (measured at 64^3, both spectra; corr_length = 10 -> ~1.4 cells).

from __future__ import annotations
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

Array = jnp.ndarray


# ---------------------------
# Correlated Gaussian / log-normal fields
# ---------------------------

def _power_spectrum(shape, corr_length: float, spectrum: str, slope: float,
                    sigma_logk: float) -> Array:
    """Isotropic P(k) on the FFT grid of `shape`. k in cycles per cell."""
    ks = [jnp.fft.fftfreq(n) for n in shape]
    grids = jnp.meshgrid(*ks, indexing="ij")
    k = jnp.sqrt(sum(g * g for g in grids))
    k0 = 1.0 / corr_length                       # peak / knee frequency
    if spectrum == "logk":                       # Gaussian bump in log k (Ray-trax default)
        logk = jnp.log(jnp.clip(k, 1e-6))
        P = jnp.exp(-0.5 * ((logk - jnp.log(k0)) / sigma_logk) ** 2)
    elif spectrum == "powerlaw":                 # flat below k0, k^-slope above
        P = jnp.where(k < k0, 1.0, (k / k0) ** (-slope))
    else:
        raise ValueError("spectrum must be 'logk' or 'powerlaw'")
    return P.at[(0,) * len(shape)].set(0.0)      # zero mean


@partial(jax.jit, static_argnames=("shape", "spectrum"))
def gaussian_field(key, shape=(64, 64, 64), corr_length: float = 10.0,
                   spectrum: str = "logk", slope: float = 11.0 / 3.0,
                   sigma_logk: float = 0.5) -> Array:
    """
    Real Gaussian random field with zero mean, unit variance and the chosen
    isotropic spectrum. Periodic by construction (FFT).

    corr_length : dominant structure size in cells (logk: peak wavelength;
                  powerlaw: knee wavelength, white below it).
    slope       : exponent for 'powerlaw' (11/3 = Kolmogorov 3D density-like).
    """
    white = random.normal(key, shape)
    P = _power_spectrum(shape, corr_length, spectrum, slope, sigma_logk)
    g = jnp.fft.ifftn(jnp.fft.fftn(white) * jnp.sqrt(P)).real
    return (g - g.mean()) / g.std()


@partial(jax.jit, static_argnames=("shape", "spectrum"))
def lognormal_field(key, shape=(64, 64, 64), mean: float = 1.0,
                    sigma_g: float = 1.0, corr_length: float = 10.0,
                    spectrum: str = "logk", slope: float = 11.0 / 3.0) -> Array:
    """
    Correlated log-normal field  X = mean * exp(sigma_g * g) / <exp(sigma_g g)>
    with g the unit Gaussian field above. sigma_g sets the contrast
    (sigma_g = 1 -> ~e^±1 around the median, 1.5 -> two decades of range).
    """
    g = gaussian_field(key, shape, corr_length, spectrum, slope)
    x = jnp.exp(sigma_g * g)
    return x * (mean / x.mean())


def top_percentile_mask(field: Array, percentile: float = 99.0) -> Array:
    """Boolean mask of the densest `100 - percentile` % cells (Ray-trax helper)."""
    return field >= jnp.percentile(field, percentile)


# ---------------------------
# Solver-ready fields
# ---------------------------

def turbulent_opacity(key, shape=(64, 64, 64), kappa_mean: float = 0.05,
                      sigma_g: float = 1.0, corr_length: float = 10.0,
                      albedo: float = 0.0, spectrum: str = "logk",
                      slope: float = 11.0 / 3.0):
    """
    Absorption and total opacity fields for the moment solver.
      kappa_tot = log-normal field with mean kappa_mean (per cell length)
      kappa_a   = (1 - albedo) kappa_tot,   kappa_t = kappa_tot
    (kappa_t must be >= kappa_a: it is the total opacity acting on the flux.)
    Returns (kappa_a, kappa_t).
    """
    kappa_tot = lognormal_field(key, shape, kappa_mean, sigma_g, corr_length, spectrum, slope)
    return (1.0 - albedo) * kappa_tot, kappa_tot


def random_point_sources(key, shape=(64, 64, 64), n_sources: int = 3,
                         L_range=(0.3, 2.0), margin: int = 8, cell_volume: float = 1.0):
    """
    n_sources single-cell sources at random positions (>= margin cells from
    the walls) with log-uniform luminosities in L_range.
    Returns S_E (power / volume, shape `shape`), positions (n,3) int array,
    luminosities (n,) array.
    """
    k_pos, k_lum = random.split(key)
    # distinct cells: draw without replacement from the flattened interior box
    inner = tuple(n - 2 * margin for n in shape)
    flat = random.choice(k_pos, int(np.prod(inner)), (n_sources,), replace=False)
    pos = jnp.stack(jnp.unravel_index(flat, inner), axis=-1) + margin
    return _deposit_sources(k_lum, pos, shape, L_range, cell_volume)


def _deposit_sources(key, pos: Array, shape, L_range, cell_volume: float):
    """Log-uniform luminosities in L_range, deposited in the cells `pos` (n,3)."""
    logL = random.uniform(key, (pos.shape[0],),
                          minval=jnp.log(L_range[0]), maxval=jnp.log(L_range[1]))
    L = jnp.exp(logL)
    S_E = jnp.zeros(shape).at[pos[:, 0], pos[:, 1], pos[:, 2]].add(L / cell_volume)
    return S_E, pos, L


def dense_point_sources(key, density: Array, percentile: float = 99.0,
                        L_range=(0.3, 2.0), margin: int = 0, cell_volume: float = 1.0):
    """
    A single-cell source ("star") in every cell at or above `percentile` of
    `density` (99 -> the densest 1 % of the cells), the criterion of Ray-trax
    `process_density_field`. The percentile therefore sets the number of
    stars: (1 - percentile/100) * n_cells. Cells closer than `margin` to a
    wall are skipped. Luminosities are log-uniform in L_range (`key`).
    Same returns as random_point_sources. The number of stars depends on the
    data, so this runs on the host and is not jit-able.
    """
    mask = np.array(top_percentile_mask(density, percentile))
    if mask.all() and percentile > 0:
        raise ValueError("uniform density field: the percentile criterion would make every cell a star")
    if margin > 0:
        inner = np.zeros_like(mask)
        inner[tuple(slice(margin, n - margin) for n in mask.shape)] = True
        mask &= inner
    if not mask.any():
        raise ValueError(f"no cell above percentile {percentile} is at least {margin} cells from the walls")
    pos = jnp.asarray(np.argwhere(mask))
    return _deposit_sources(key, pos, density.shape, L_range, cell_volume)


def random_setup(key, shape=(64, 64, 64), n_sources: int = 3,
                 kappa_mean: float = 0.05, sigma_g: float = 1.0,
                 corr_length: float = 10.0, albedo: float = 0.0,
                 L_range=(0.3, 2.0), margin: int = 8, E_floor: float = 1e-6,
                 spectrum: str = "logk", slope: float = 11.0 / 3.0,
                 cell_volume: float = 1.0, source_percentile: float | None = None):
    """
    One random problem for `advance` / `advance_vet`:
      U0, kappa_a, kappa_t, E_eq, S_E   (+ positions, luminosities, key split)
    Use different `key`s to build a dataset.

    source_percentile : None -> n_sources sources anywhere in the interior
                        (random_point_sources); a number -> a source in every
                        cell above that percentile of the opacity, i.e. of the
                        gas density (dense_point_sources; 99 = top 1 %).
                        n_sources is then ignored and `margin` is applied as
                        given (0 keeps the whole top percentile).
    """
    k_kappa, k_src = random.split(key)
    kappa_a, kappa_t = turbulent_opacity(k_kappa, shape, kappa_mean, sigma_g,
                                         corr_length, albedo, spectrum, slope)
    if source_percentile is None:
        S_E, pos, L = random_point_sources(k_src, shape, n_sources, L_range, margin, cell_volume)
    else:
        S_E, pos, L = dense_point_sources(k_src, kappa_t, source_percentile,
                                          L_range, margin, cell_volume)
    U0 = jnp.concatenate([jnp.full(shape + (1,), E_floor), jnp.zeros(shape + (3,))], -1)
    return dict(U0=U0, kappa_a=kappa_a, kappa_t=kappa_t, E_eq=jnp.zeros(shape),
                S_E=S_E, positions=pos, luminosities=L)


# ---------------------------
# Saving / loading one problem
# ---------------------------

def save_setup(path: str, setup: dict, **meta) -> None:
    """
    Write the fields of one `random_setup` problem to a compressed .npz:
      kappa_a, S_E (float32), positions, luminosities, and the scalars in `meta`.
    kappa_t is stored only if it differs from kappa_a (scattering); E_eq and
    U0 are trivial and rebuilt by load_setup.
    """
    kappa_a, kappa_t = np.asarray(setup["kappa_a"]), np.asarray(setup["kappa_t"])
    arrays = dict(kappa_a=kappa_a.astype(np.float32),
                  S_E=np.asarray(setup["S_E"], dtype=np.float32),
                  positions=np.asarray(setup["positions"], dtype=np.int32),
                  luminosities=np.asarray(setup["luminosities"], dtype=np.float64))
    if not np.array_equal(kappa_a, kappa_t):
        arrays["kappa_t"] = kappa_t.astype(np.float32)
    np.savez_compressed(path, **arrays, **{f"meta_{k}": v for k, v in meta.items()})


def load_setup(path: str, E_floor: float = 1e-6) -> dict:
    """Inverse of save_setup: the dict `random_setup` returns, plus `meta`."""
    ftype = jnp.result_type(float)                   # float64 if x64 is enabled
    with np.load(path) as d:
        kappa_a = jnp.asarray(d["kappa_a"], dtype=ftype)
        kappa_t = jnp.asarray(d["kappa_t"], dtype=ftype) if "kappa_t" in d else kappa_a
        S_E = jnp.asarray(d["S_E"], dtype=ftype)
        pos, L = jnp.asarray(d["positions"]), jnp.asarray(d["luminosities"], dtype=ftype)
        meta = {k[5:]: d[k].item() for k in d.files if k.startswith("meta_")}
    shape = kappa_a.shape
    U0 = jnp.concatenate([jnp.full(shape + (1,), E_floor, dtype=ftype),
                          jnp.zeros(shape + (3,), dtype=ftype)], -1)
    return dict(U0=U0, kappa_a=kappa_a, kappa_t=kappa_t, E_eq=jnp.zeros(shape, dtype=ftype),
                S_E=S_E, positions=pos, luminosities=L, meta=meta)


def optical_depth_stats(kappa_t: Array, h: float = 1.0):
    """Mean optical depth across the box along each axis (diagnostic)."""
    return tuple(float((kappa_t * h).sum(axis=a).mean()) for a in range(3))
