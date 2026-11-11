# momentRT.py
# Minimal 3D gray radiative transfer (two-moment M1 closure) with HLLE,
# implicit absorption/scattering, and explicit volumetric sources.
# JAX implementation (CPU/GPU/TPU). First-order FV; MUSCL can be added later.
#
# Boundary conditions: "outflow" (zero-gradient ghost cells, default) or
# "periodic". Selected through M1Config.bc.

from __future__ import annotations
from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp

Array = jnp.ndarray

# ---------------------------
# M1 closure (Levermore/Minerbo)
# ---------------------------

def m1_closure(E: Array, F: Array, c: float, eps: float = 1e-12):
    """
    Compute Eddington tensor components for the M1 closure.

    Inputs
      E : (...,)     energy density
      F : (...,3)    flux vector
      c : float      radiation speed
      eps : float    small floor for numerical safety

    Returns
      (P_xx, P_yy, P_zz, P_xy, P_xz, P_yz) each shape (...,)
    """
    Fmag = jnp.linalg.norm(F, axis=-1)
    cE   = c * jnp.maximum(E, eps)
    f    = jnp.clip(Fmag / cE, 0.0, 1.0 - 1e-9)

    # Eddington factor chi(f)
    chi = (3.0 + 4.0 * f * f) / (5.0 + 2.0 * jnp.sqrt(jnp.maximum(0.0, 4.0 - 3.0 * f * f)))
    a = 0.5 * (1.0 - chi)
    b = 0.5 * (3.0 * chi - 1.0)

    # Unit direction n
    n = jnp.where((Fmag > 0)[..., None], F / (Fmag[..., None] + eps), jnp.zeros_like(F))
    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]

    Efac_a = E * a
    Efac_b = E * b

    P_xx = Efac_a + Efac_b * nx * nx
    P_yy = Efac_a + Efac_b * ny * ny
    P_zz = Efac_a + Efac_b * nz * nz
    P_xy = Efac_b * nx * ny
    P_xz = Efac_b * nx * nz
    P_yz = Efac_b * ny * nz
    return P_xx, P_yy, P_zz, P_xy, P_xz, P_yz


# ---------------------------
# Physical fluxes per direction
# ---------------------------

def physical_flux(axis: int, E: Array, F: Array, P6, c: float):
    """
    Flux of the conserved state U = [E, Fx, Fy, Fz] along `axis`, given the
<<<<<<< HEAD
    pressure tensor components P6 = (P_xx, P_yy, P_zz, P_xy, P_xz, P_yz):
      axis 0: [ c F_x,  c^2 P_xx,  c^2 P_xy,  c^2 P_xz ]
      axis 1: [ c F_y,  c^2 P_xy,  c^2 P_yy,  c^2 P_yz ]
      axis 2: [ c F_z,  c^2 P_xz,  c^2 P_yz,  c^2 P_zz ]
=======
    pressure tensor components P6 = (P_xx, P_yy, P_zz, P_xy, P_xz, P_yz).
    F is the physical energy flux (|F| <= c E, as assumed by the closure and
    the limiter), so the system is  dE/dt + div F = 0,  dF/dt + c^2 div P = 0:
      axis 0: [ F_x,  c^2 P_xx,  c^2 P_xy,  c^2 P_xz ]
      axis 1: [ F_y,  c^2 P_xy,  c^2 P_yy,  c^2 P_yz ]
      axis 2: [ F_z,  c^2 P_xz,  c^2 P_yz,  c^2 P_zz ]
    Wave speeds are +-c (free streaming), matching the HLLE bound and cfl_dt.
>>>>>>> upstream/main
    """
    P_xx, P_yy, P_zz, P_xy, P_xz, P_yz = P6
    rows = ((P_xx, P_xy, P_xz), (P_xy, P_yy, P_yz), (P_xz, P_yz, P_zz))[axis]
    c2 = c * c
<<<<<<< HEAD
    return jnp.stack([c * F[..., axis], c2 * rows[0], c2 * rows[1], c2 * rows[2]], axis=-1)

def flux_x(E: Array, F: Array, c: float):
    return physical_flux(0, E, F, m1_closure(E, F, c), c)

def flux_y(E: Array, F: Array, c: float):
    return physical_flux(1, E, F, m1_closure(E, F, c), c)

def flux_z(E: Array, F: Array, c: float):
    return physical_flux(2, E, F, m1_closure(E, F, c), c)


# ---------------------------
# HLLE numerical flux (symmetric speeds ±c)
# ---------------------------

def hlle(F_L: Array, F_R: Array, U_L: Array, U_R: Array, smax):
    """HLLE flux with sL=-smax, sR=+smax (smax: scalar or per-face array)."""
    return 0.5 * (F_L + F_R) - 0.5 * smax * (U_R - U_L)


=======
    return jnp.stack([F[..., axis], c2 * rows[0], c2 * rows[1], c2 * rows[2]], axis=-1)


# ---------------------------
# HLLE numerical flux (symmetric speeds ±smax)
# ---------------------------

def hlle(F_L: Array, F_R: Array, U_L: Array, U_R: Array, smax):
    """HLLE flux with sL=-smax, sR=+smax (smax: scalar or per-face array).
    With symmetric speeds this is the Rusanov / local Lax-Friedrichs flux."""
    return 0.5 * (F_L + F_R) - 0.5 * smax * (U_R - U_L)


def ap_speed_factor(kappa_L: Array, kappa_R: Array, h: float) -> Array:
    """
    Asymptotic-preserving scaling of the HLLE dissipation speed at a face,
      eps = 1 / (1 + kappa_t h)^2,   kappa_t = face-averaged total opacity.
    The Rusanov term adds an effective diffusivity smax*h/2 to the energy
    equation. In an optically thick cell (kappa_t h >~ 1) that exceeds the
    physical diffusivity c/(3 kappa_t) and the diffusion limit is lost; with
    this factor the numerical part scales as c/(2 kappa_t^2 h) and vanishes
    relative to the physical one. For kappa_t h << 1 the factor is ~1 and
    the free-streaming scheme is unchanged.
    """
    return 1.0 / (1.0 + 0.5 * (kappa_L + kappa_R) * h) ** 2


>>>>>>> upstream/main
# ---------------------------
# Finite-volume utilities
# ---------------------------

_PAD_MODE = {"periodic": "wrap", "outflow": "edge"}

def pad_ghost(U: Array, axis: int, bc: str) -> Array:
    """Add one ghost cell on each side of `axis` according to the BC."""
    if bc not in _PAD_MODE:
        raise ValueError(f"Unknown bc '{bc}'. Use one of {list(_PAD_MODE)}.")
    pad = [(0, 0)] * U.ndim
    pad[axis] = (1, 1)
    return jnp.pad(U, pad, mode=_PAD_MODE[bc])

def _slice_axis(a: Array, axis: int, start, stop) -> Array:
    idx = [slice(None)] * a.ndim
    idx[axis] = slice(start, stop)
    return a[tuple(idx)]

def divergence_flux_generic(U: Array, aux, closure_fn,
                            dx: float, dy: float, dz: float, c: float,
                            epsE: float = 1e-12, bc: str = "outflow",
<<<<<<< HEAD
                            speed_fn=None):
=======
                            speed_fn=None, kappa: Array | None = None):
>>>>>>> upstream/main
    """
    Compute ∇·F with HLLE in x, y, z for an arbitrary closure.

    U       : (Nx, Ny, Nz, 4)  conserved state [E, Fx, Fy, Fz]
    aux     : (Nx, Ny, Nz, k) or None — extra cell-centred fields the closure
              needs (e.g. the Eddington tensor); padded alongside U.
    closure_fn(E, F, aux) -> (P_xx, P_yy, P_zz, P_xy, P_xz, P_yz)
    speed_fn(axis, E, F, aux) -> max |wave speed| per cell along `axis`
              (None: use c). The HLLE speed at a face is the max of both sides.
<<<<<<< HEAD
=======
    kappa   : (Nx, Ny, Nz) total opacity or None. If given, the HLLE speed is
              multiplied by ap_speed_factor so the scheme keeps the diffusion
              limit in optically thick cells.
>>>>>>> upstream/main

    For each axis, one ghost cell is added on each side (periodic wrap or
    zero-gradient outflow), fluxes are evaluated on all N+1 faces, and the
    divergence is the face-flux difference per cell.
    """
<<<<<<< HEAD
    W = U if aux is None else jnp.concatenate([U, aux], axis=-1)
=======
    parts = [U]
    if aux is not None:
        parts.append(aux)
    if kappa is not None:
        parts.append(kappa[..., None])
    W = jnp.concatenate(parts, axis=-1) if len(parts) > 1 else U
    n_aux = 0 if aux is None else aux.shape[-1]
>>>>>>> upstream/main
    div = jnp.zeros_like(U)
    for axis, h in enumerate((dx, dy, dz)):
        Wp  = pad_ghost(W, axis, bc)                       # N+2 along axis
        W_L = _slice_axis(Wp, axis, 0, -1)                 # N+1 left states
        W_R = _slice_axis(Wp, axis, 1, None)               # N+1 right states
<<<<<<< HEAD
        U_L, U_R = W_L[..., :4], W_R[..., :4]
        A_L = None if aux is None else W_L[..., 4:]
        A_R = None if aux is None else W_R[..., 4:]
        E_L = jnp.maximum(U_L[..., 0], epsE)
        E_R = jnp.maximum(U_R[..., 0], epsE)
        F_L, F_R = U_L[..., 1:4], U_R[..., 1:4]
=======
        A_L = None if aux is None else W_L[..., 4:4 + n_aux]
        A_R = None if aux is None else W_R[..., 4:4 + n_aux]
        E_L = jnp.maximum(W_L[..., 0], epsE)
        E_R = jnp.maximum(W_R[..., 0], epsE)
        F_L, F_R = W_L[..., 1:4], W_R[..., 1:4]
        # floored states, used consistently in the flux and the HLLE jump term
        U_L = jnp.concatenate([E_L[..., None], F_L], axis=-1)
        U_R = jnp.concatenate([E_R[..., None], F_R], axis=-1)
>>>>>>> upstream/main
        if speed_fn is None:
            smax = c
        else:
            smax = jnp.maximum(speed_fn(axis, E_L, F_L, A_L),
                               speed_fn(axis, E_R, F_R, A_R))[..., None]
<<<<<<< HEAD
=======
        if kappa is not None:
            k_L, k_R = W_L[..., 4 + n_aux], W_R[..., 4 + n_aux]
            smax = smax * ap_speed_factor(k_L, k_R, h)[..., None]
>>>>>>> upstream/main
        F_face = hlle(physical_flux(axis, E_L, F_L, closure_fn(E_L, F_L, A_L), c),
                      physical_flux(axis, E_R, F_R, closure_fn(E_R, F_R, A_R), c),
                      U_L, U_R, smax=smax)                 # N+1 faces
        div = div + (_slice_axis(F_face, axis, 1, None)
                     - _slice_axis(F_face, axis, 0, -1)) / h
    return div


def divergence_flux(U: Array, dx: float, dy: float, dz: float, c: float,
<<<<<<< HEAD
                    epsE: float = 1e-12, bc: str = "outflow"):
    """∇·F with the M1 closure (see divergence_flux_generic)."""
    return divergence_flux_generic(U, None, lambda E, F, _: m1_closure(E, F, c),
                                   dx, dy, dz, c, epsE=epsE, bc=bc)
=======
                    epsE: float = 1e-12, bc: str = "outflow",
                    kappa: Array | None = None):
    """∇·F with the M1 closure (see divergence_flux_generic)."""
    return divergence_flux_generic(U, None, lambda E, F, _: m1_closure(E, F, c),
                                   dx, dy, dz, c, epsE=epsE, bc=bc, kappa=kappa)
>>>>>>> upstream/main


# ---------------------------
# Source terms
# ---------------------------

def source_implicit(E: Array, F: Array, dt: float, c: float,
                    kappa_a: Array, kappa_t: Array, E_eq: Array):
    """
    Local implicit update for absorption/emission & transport damping:
      dE/dt =  c kappa_a (E_eq - E)
      dF/dt = -c kappa_t F

    Implicit Euler (algebraic, local):
      E^{n+1} = (E* + dt c kappa_a E_eq) / (1 + dt c kappa_a)
      F^{n+1} = F* / (1 + dt c kappa_t)
    """
    fac_E = 1.0 / (1.0 + dt * c * kappa_a)
    E_new = (E + dt * c * kappa_a * E_eq) * fac_E

    fac_F = 1.0 / (1.0 + dt * c * kappa_t)
    F_new = F * fac_F[..., None]
    return E_new, F_new


def enforce_positivity_and_causality(E: Array, F: Array, c: float, epsE: float = 1e-12):
    """
    Clamp E >= epsE and |F| <= cE. Keeps the M1 closure well-defined.
    """
    Ep = jnp.maximum(E, epsE)
    Fmag = jnp.linalg.norm(F, axis=-1)
    limit = jnp.minimum(1.0, (c * Ep) / (Fmag + 1e-30))
    Fp = F * limit[..., None]
    return Ep, Fp


# ---------------------------
# Timestep control
# ---------------------------

def cfl_dt(dx: float, dy: float, dz: float, c: float, cfl: float = 0.2):
    """
    CFL step for the dimensionally-unsplit first-order scheme with wave
    speed c: stability requires dt * c * (1/dx + 1/dy + 1/dz) <= 1.
    """
    return cfl / (c * (1.0 / dx + 1.0 / dy + 1.0 / dz))


# ---------------------------
# Main integrator
# ---------------------------

@dataclass(frozen=True)  # hashable/static for jit caching
class M1Config:
    c: float = 1.0
    dx: float = 1.0
    dy: float = 1.0
    dz: float = 1.0
    cfl: float = 0.2
    epsE: float = 1e-12
    bc: str = "outflow"   # "outflow" or "periodic"
<<<<<<< HEAD
=======
    ap: bool = True       # asymptotic-preserving HLLE dissipation (ap_speed_factor)
>>>>>>> upstream/main


@partial(jax.jit, static_argnums=(2,))  # cfg is the 3rd positional arg
def step_m1(U: Array, dt: float, cfg: M1Config,
            kappa_a: Array, kappa_t: Array, E_eq: Array,
            S_E: Array, S_F: Array):
    """
    One FV step:
      1) Hyperbolic update (HLLE, BC from cfg.bc)
      2) Positivity/causality limiter
      3) Implicit absorption/scattering
      4) Explicit volumetric sources (S_E, S_F)
      5) Positivity/causality limiter
    """
    # Hyperbolic update
<<<<<<< HEAD
    divU   = divergence_flux(U, cfg.dx, cfg.dy, cfg.dz, cfg.c, epsE=cfg.epsE, bc=cfg.bc)
=======
    divU   = divergence_flux(U, cfg.dx, cfg.dy, cfg.dz, cfg.c, epsE=cfg.epsE, bc=cfg.bc,
                             kappa=kappa_t if cfg.ap else None)
>>>>>>> upstream/main
    U_star = U - dt * divU

    # Pre-source limiter
    E_star = jnp.maximum(U_star[..., 0], cfg.epsE)
    F_star = U_star[..., 1:4]
    E_star, F_star = enforce_positivity_and_causality(E_star, F_star, cfg.c, epsE=cfg.epsE)

    # Implicit local sources (absorption/emission & flux damping)
    E_np1, F_np1 = source_implicit(E_star, F_star, dt, cfg.c, kappa_a, kappa_t, E_eq)

    # Explicit volumetric sources
    E_np1 = E_np1 + dt * S_E
    F_np1 = F_np1 + dt * S_F

    # Post-source limiter
    E_np1, F_np1 = enforce_positivity_and_causality(E_np1, F_np1, cfg.c, epsE=cfg.epsE)

    return jnp.concatenate([E_np1[..., None], F_np1], axis=-1)


def advance(U0: Array, nsteps: int, cfg: M1Config,
            kappa_a: Array, kappa_t: Array, E_eq: Array,
            dt: float | None = None,
            S_E: Array | None = None, S_F: Array | None = None):
    """
    Advance nsteps. If dt is None, uses CFL dt (fixed).
    Optional explicit sources:
      S_E: (Nx,Ny,Nz)    energy source density [power / volume]
      S_F: (Nx,Ny,Nz,3)  momentum source       [flux / time]
    """
    if dt is None:
        dt = cfl_dt(cfg.dx, cfg.dy, cfg.dz, cfg.c, cfg.cfl)

    # Default explicit sources to zeros (avoid None under jit)
    if S_E is None:
        S_E = jnp.zeros_like(U0[..., 0])
    if S_F is None:
        S_F = jnp.zeros_like(U0[..., 1:4])

    def body(U, _):
        U = step_m1(U, dt, cfg, kappa_a, kappa_t, E_eq, S_E, S_F)
        return U, None

    Uf, _ = jax.lax.scan(body, U0, None, length=nsteps)
    return Uf
