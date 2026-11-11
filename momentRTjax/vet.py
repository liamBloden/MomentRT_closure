# vet.py
# Variable Eddington Tensor (VET) closure for the two-moment solver.
#
# The Eddington tensor f_ij = P_ij / E is obtained from a formal solution of
# the transfer equation  n·∇I = -kappa I + j  on a discrete set of directions
# (Fibonacci lattice on the sphere) using first-order short characteristics
# with plane-by-plane upwind sweeps. The moment equations are then advanced
# with P_ij = f_ij E exactly as in the M1 solver (same HLLE flux, same
# implicit sources and limiters), only the closure differs.
#
# Because the tensor only depends on the angular *shape* of the radiation
# field, it is recomputed every `vet_every` steps (VETConfig.vet_every).
#
# Grid must be cubic (dx == dy == dz): the transverse ray shift per plane is
# then at most one cell, which keeps the upwind interpolation a 2-point one.

from __future__ import annotations
from dataclasses import dataclass
<<<<<<< HEAD
from functools import partial
=======
from functools import partial, lru_cache
>>>>>>> upstream/main
import numpy as np
import jax
import jax.numpy as jnp

from .momentRT import (
    Array, M1Config, divergence_flux_generic, source_implicit,
    enforce_positivity_and_causality, cfl_dt,
)


# ---------------------------
# Directions
# ---------------------------

def fibonacci_directions(n: int, symmetric: bool = True) -> np.ndarray:
    """
    ~n unit vectors quasi-uniformly distributed on the sphere.
    symmetric=True: a golden-ratio lattice restricted to the positive octant is
    reflected into all eight octants, so the set is invariant under every axis
    reflection (keeps the Eddington tensor of a symmetric problem symmetric).
    The count is rounded to a multiple of 8.
    """
    def lattice(m):
        i = np.arange(m) + 0.5
        phi = np.arccos(1.0 - 2.0 * i / m)
        theta = np.pi * (1.0 + 5.0 ** 0.5) * i
        return np.stack([np.cos(theta) * np.sin(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(phi)], axis=-1)
    if not symmetric:
        return lattice(n)
    n_oct = max(1, n // 8)
    m = 8 * n_oct
    while True:                     # grow the lattice until the octant holds n_oct points
        d = lattice(m)
        oct_ = d[(d > 0).all(axis=1)]
        if len(oct_) >= n_oct:
<<<<<<< HEAD
            oct_ = oct_[:n_oct]
=======
            # drop the surplus evenly in polar order rather than only near the equator
            keep = np.round(np.linspace(0, len(oct_) - 1, n_oct)).astype(int)
            oct_ = oct_[keep]
>>>>>>> upstream/main
            break
        m += 8
    signs = np.array([[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)])
    return (oct_[None, :, :] * signs[:, None, :]).reshape(-1, 3)


class DirectionSet:
    """
    Directions grouped by (dominant axis, sign of the dominant component).
    Each group is swept along its own axis; grouping is done once in NumPy so
    that every group has a static shape inside jit.
    """
    def __init__(self, n_dirs: int, symmetric: bool = True):
        dirs = fibonacci_directions(n_dirs, symmetric)
        self.n_dirs = len(dirs)
<<<<<<< HEAD
        self.weight = 4.0 * np.pi / self.n_dirs
        dom = np.argmax(np.abs(dirs), axis=1)
        self.groups = []   # list of (axis, sign, (m,3) directions)
=======
        self.weight = quadrature_weights(dirs)          # (n,), sums to 4 pi
        dom = np.argmax(np.abs(dirs), axis=1)
        self.groups = []   # list of (axis, sign, (m,3) directions, (m,) weights)
>>>>>>> upstream/main
        for axis in range(3):
            for sign in (+1, -1):
                sel = (dom == axis) & (np.sign(dirs[:, axis]) == sign)
                if sel.any():
<<<<<<< HEAD
                    self.groups.append((axis, sign, jnp.asarray(dirs[sel])))
=======
                    self.groups.append((axis, sign, jnp.asarray(dirs[sel]),
                                        jnp.asarray(self.weight[sel])))
>>>>>>> upstream/main

    def tree(self):
        """pytree of direction arrays, in group order (static structure)."""
        return tuple(g[2] for g in self.groups)


<<<<<<< HEAD
=======
def quadrature_weights(dirs: np.ndarray) -> np.ndarray:
    """
    Weights closest (least squares) to uniform 4 pi / n that integrate 1 and
    n_i n_j exactly over the sphere:  sum w = 4 pi,  sum w n_i n_j = 4 pi/3 d_ij.
    A uniform-weight Fibonacci set gets the second moments wrong by ~2 % at
    192 directions, which would bias the Eddington tensor of an isotropic
    field away from d_ij / 3.
    """
    n = len(dirs)
    w0 = np.full(n, 4.0 * np.pi / n)
    nx, ny, nz = dirs.T
    A = np.stack([np.ones(n), nx * nx, ny * ny, nz * nz, nx * ny, nx * nz, ny * nz])
    b = np.array([4 * np.pi] + [4 * np.pi / 3] * 3 + [0.0] * 3)
    # minimal-norm correction: w = w0 + A^T lam,  (A A^T) lam = b - A w0  (A is rank 6)
    lam = np.linalg.lstsq(A @ A.T, b - A @ w0, rcond=None)[0]
    w = w0 + A.T @ lam
    assert (w > 0).all(), "quadrature weights must stay positive"
    return w


>>>>>>> upstream/main
# ---------------------------
# Short-characteristics formal solution
# ---------------------------

def _shift1(A: Array, k: int, axis: int, bc: str) -> Array:
    """A shifted so that result[i] = A[i - k] (k = ±1). Wrap for periodic,
    zero incoming intensity for outflow."""
    if bc == "periodic":
        return jnp.roll(A, k, axis=axis)
    pad = [(0, 0)] * A.ndim
    pad[axis] = (1, 0) if k > 0 else (0, 1)
    Ap = jnp.pad(A, pad)
    idx = [slice(None)] * A.ndim
    idx[axis] = slice(0, -1) if k > 0 else slice(1, None)
    return Ap[tuple(idx)]


def _upwind_interp(plane: Array, d: Array, axis: int, bc: str) -> Array:
    """
    Value of `plane` at position (i - d) along `axis`, |d| <= 1, by linear
    interpolation. `plane` is batched over directions on axis 0, so `axis`
    refers to the full array; `d` has shape (n_dir,) and broadcasts.
    """
    ad = jnp.abs(d)[:, None, None]
    nb = jnp.where((d > 0)[:, None, None],
                   _shift1(plane, +1, axis, bc),
                   _shift1(plane, -1, axis, bc))
    return (1.0 - ad) * plane + ad * nb


def _sweep_group(j: Array, kappa: Array, dirs: Array, axis: int, sign: int,
                 h: float, bc: str, n_pass: int):
    """
    Formal solution for all directions in one group.
    j, kappa : (N,N,N) emissivity (per steradian) and opacity
    dirs     : (m,3) unit vectors whose dominant component is `axis` with `sign`
    Returns I of shape (m, N, N, N) in the original axis order.
    """
    # Move the sweep axis to the front, orient so the sweep goes +index.
    order = [axis] + [a for a in range(3) if a != axis]
    jt = jnp.transpose(j, order)
    kt = jnp.transpose(kappa, order)
    if sign < 0:
        jt, kt = jt[::-1], kt[::-1]

    na = jnp.abs(dirs[:, axis])
    # Flipping the sweep axis negates only n_a; transverse components stay.
    nb = dirs[:, order[1]]
    nc = dirs[:, order[2]]
    ds = h / na                         # path length through one plane
    db = nb / na                        # transverse shift per plane (cells)
    dc = nc / na
<<<<<<< HEAD

    tau = kt[None] * ds[:, None, None, None]                  # (m,N,N,N)
    att = jnp.exp(-tau)
    # source integral  ∫ j e^{-τ'} ds'  with constant j, kappa in the cell
    src = jt[None] * jnp.where(tau > 1e-8,
                               -jnp.expm1(-tau) / jnp.maximum(kt[None], 1e-300),
                               ds[:, None, None, None] * (1.0 - 0.5 * tau))
=======
    ds3 = ds[:, None, None]

    def plane_coeffs(k):
        """Attenuation and source integral of plane k for all m directions,
        computed on the fly (keeps memory at O(m N^2) instead of O(m N^3))."""
        tau = kt[k][None] * ds3                               # (m,N,N)
        att = jnp.exp(-tau)
        # source integral  ∫ j e^{-τ'} ds'  with constant j, kappa in the cell
        src = jt[k][None] * jnp.where(tau > 1e-8,
                                      -jnp.expm1(-tau) / jnp.maximum(kt[k][None], 1e-300),
                                      ds3 * (1.0 - 0.5 * tau))
        return att, src
>>>>>>> upstream/main

    def one_pass(I_in):
        def body(I_prev, k):
            up = _upwind_interp(I_prev, db, 1, bc)
            up = _upwind_interp(up, dc, 2, bc)
<<<<<<< HEAD
            I_k = up * att[:, k] + src[:, k]
=======
            att, src = plane_coeffs(k)
            I_k = up * att + src
>>>>>>> upstream/main
            return I_k, I_k
        _, I_planes = jax.lax.scan(body, I_in, jnp.arange(jt.shape[0]))
        return jnp.moveaxis(I_planes, 0, 1)                    # (m,N,N,N)

    m, N = dirs.shape[0], jt.shape[0]
    I_in = jnp.zeros((m,) + jt.shape[1:])
    I = one_pass(I_in)
    if bc == "periodic":                 # feed the outgoing plane back in
        for _ in range(n_pass - 1):
            I = one_pass(I[:, -1])

    if sign < 0:
        I = I[:, ::-1]
    inv = [0] + [1 + order.index(a) for a in range(3)]        # undo transpose
    return jnp.transpose(I, inv)


def formal_solution(j: Array, kappa: Array, dirset: DirectionSet, h: float,
                    c: float, bc: str = "outflow", n_pass: int = 2):
    """
    Angular moments of the formal solution:
      E_f  = (1/c) Σ_d w I_d
      P6_f = (1/c) Σ_d w I_d n_i n_j      (xx, yy, zz, xy, xz, yz)
    """
<<<<<<< HEAD
    w = dirset.weight / c
    E_f = jnp.zeros_like(j)
    P = [jnp.zeros_like(j) for _ in range(6)]
    for (axis, sign, dirs) in dirset.groups:
        I = _sweep_group(j, kappa, dirs, axis, sign, h, bc, n_pass)   # (m,N,N,N)
        nx, ny, nz = (dirs[:, i][:, None, None, None] for i in range(3))
        E_f = E_f + w * jnp.sum(I, axis=0)
        for q, nn in enumerate((nx*nx, ny*ny, nz*nz, nx*ny, nx*nz, ny*nz)):
            P[q] = P[q] + w * jnp.sum(I * nn, axis=0)
=======
    E_f = jnp.zeros_like(j)
    P = [jnp.zeros_like(j) for _ in range(6)]
    for (axis, sign, dirs, wg) in dirset.groups:
        I = _sweep_group(j, kappa, dirs, axis, sign, h, bc, n_pass)   # (m,N,N,N)
        w = (wg / c)[:, None, None, None]
        nx, ny, nz = (dirs[:, i][:, None, None, None] for i in range(3))
        E_f = E_f + jnp.sum(w * I, axis=0)
        for q, nn in enumerate((nx*nx, ny*ny, nz*nz, nx*ny, nx*nz, ny*nz)):
            P[q] = P[q] + jnp.sum(w * I * nn, axis=0)
>>>>>>> upstream/main
    return E_f, tuple(P)


def eddington_tensor(E_f: Array, P6_f, rel_floor: float = 1e-12) -> Array:
    """f_ij = P_ij / E, isotropic (δ_ij/3) where the formal E is negligible.
    Returns (N,N,N,6) in the order xx, yy, zz, xy, xz, yz."""
    ok = E_f > rel_floor * jnp.max(E_f)
    inv = jnp.where(ok, 1.0 / jnp.where(ok, E_f, 1.0), 0.0)
    iso = (1/3, 1/3, 1/3, 0.0, 0.0, 0.0)
    f = [jnp.where(ok, P * inv, iso_q) for P, iso_q in zip(P6_f, iso)]
    return jnp.stack(f, axis=-1)


# ---------------------------
# Moment update with a given Eddington tensor
# ---------------------------

@dataclass(frozen=True)
class VETConfig(M1Config):
    n_dirs: int = 192      # directions in the formal solution (multiple of 8)
    vet_every: int = 50    # recompute the tensor every this many steps
    n_pass: int = 2        # sweeps for periodic BC (incoming = outgoing)
    smooth: int = 1        # 3^3 box-filter passes on the formal moments before P/E (damps ray-effect noise)
    wave_floor: float = 1.0   # HLLE speed = c*sqrt(max(f_aa, wave_floor)); 1.0 -> c


def _vet_closure(E, F, f6):
    return tuple(f6[..., q] * E for q in range(6))


def _vet_speed(cfg: VETConfig):
    """Eigenvalues of the frozen-tensor system along `axis` are ±c sqrt(f_aa)."""
    if cfg.wave_floor >= 1.0:
        return None
    def speed(axis, E, F, f6):
        return cfg.c * jnp.sqrt(jnp.maximum(f6[..., axis], cfg.wave_floor))
    return speed


def box_filter(A: Array, passes: int, bc: str) -> Array:
    """3x3x3 box filter on the first three axes (edge/wrap padding)."""
    mode = "wrap" if bc == "periodic" else "edge"
    for _ in range(passes):
        pad = [(1, 1)] * 3 + [(0, 0)] * (A.ndim - 3)
        Ap = jnp.pad(A, pad, mode=mode)
        out = jnp.zeros_like(A)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    out = out + Ap[i:i+A.shape[0], j:j+A.shape[1], k:k+A.shape[2]]
        A = out / 27.0
    return A


def emissivity(cfg: VETConfig, E: Array, kappa_a: Array, kappa_t: Array,
               E_eq: Array, S_E: Array):
    """
    Isotropic emissivity per steradian seen by the formal solution:
      thermal c kappa_a E_eq, explicit source S_E, and scattering
      c (kappa_t - kappa_a) E redistributing the current field.
    """
    kappa_s = jnp.maximum(kappa_t - kappa_a, 0.0)
    return (cfg.c * kappa_a * E_eq + S_E + cfg.c * kappa_s * E) / (4.0 * jnp.pi)


@partial(jax.jit, static_argnums=(2,))
def step_vet(U: Array, dt: float, cfg: VETConfig, f6: Array,
             kappa_a: Array, kappa_t: Array, E_eq: Array,
             S_E: Array, S_F: Array):
    """One FV step with a frozen Eddington tensor f6 (N,N,N,6)."""
    divU   = divergence_flux_generic(U, f6, _vet_closure,
                                     cfg.dx, cfg.dy, cfg.dz, cfg.c,
                                     epsE=cfg.epsE, bc=cfg.bc,
<<<<<<< HEAD
                                     speed_fn=_vet_speed(cfg))
=======
                                     speed_fn=_vet_speed(cfg),
                                     kappa=kappa_t if cfg.ap else None)
>>>>>>> upstream/main
    U_star = U - dt * divU

    E_star = jnp.maximum(U_star[..., 0], cfg.epsE)
    F_star = U_star[..., 1:4]
    E_star, F_star = enforce_positivity_and_causality(E_star, F_star, cfg.c, epsE=cfg.epsE)

    E_np1, F_np1 = source_implicit(E_star, F_star, dt, cfg.c, kappa_a, kappa_t, E_eq)
    E_np1 = E_np1 + dt * S_E
    F_np1 = F_np1 + dt * S_F

    E_np1, F_np1 = enforce_positivity_and_causality(E_np1, F_np1, cfg.c, epsE=cfg.epsE)
    return jnp.concatenate([E_np1[..., None], F_np1], axis=-1)


@partial(jax.jit, static_argnums=(2, 3))
def compute_vet(U: Array, aux, cfg: VETConfig, dirset: DirectionSet):
    """Eddington tensor (N,N,N,6) for the current state. aux = (kappa_a, kappa_t, E_eq, S_E)."""
    kappa_a, kappa_t, E_eq, S_E = aux
    j = emissivity(cfg, U[..., 0], kappa_a, kappa_t, E_eq, S_E)
    E_f, P6_f = formal_solution(j, kappa_t, dirset, cfg.dx, cfg.c, cfg.bc, cfg.n_pass)
    if cfg.smooth > 0:
        E_f = box_filter(E_f, cfg.smooth, cfg.bc)
        P6_f = tuple(box_filter(P, cfg.smooth, cfg.bc) for P in P6_f)
    return eddington_tensor(E_f, P6_f)


<<<<<<< HEAD
=======
@partial(jax.jit, static_argnums=(3,))
def _advance_frozen(U: Array, f6: Array, dt: float, cfg: VETConfig,
                    kappa_a: Array, kappa_t: Array, E_eq: Array,
                    S_E: Array, S_F: Array):
    """cfg.vet_every steps with a frozen tensor. Module-level and jitted so the
    scan is compiled once per (shape, cfg), not once per tensor rebuild."""
    def body(U, _):
        return step_vet(U, dt, cfg, f6, kappa_a, kappa_t, E_eq, S_E, S_F), None
    return jax.lax.scan(body, U, None, length=cfg.vet_every)[0]


@lru_cache(maxsize=None)
def default_dirset(n_dirs: int) -> DirectionSet:
    """One DirectionSet per n_dirs, reused across calls so that compute_vet
    (which takes it as a static argument) is not retraced every call."""
    return DirectionSet(n_dirs)


>>>>>>> upstream/main
def advance_vet(U0: Array, nsteps: int, cfg: VETConfig,
                kappa_a: Array, kappa_t: Array, E_eq: Array,
                dt: float | None = None,
                S_E: Array | None = None, S_F: Array | None = None,
                dirset: DirectionSet | None = None,
                return_tensor: bool = False):
    """
    Advance with the VET closure. The tensor is rebuilt every cfg.vet_every
    steps; nsteps is rounded up to a multiple of cfg.vet_every.
    """
    assert cfg.dx == cfg.dy == cfg.dz, "VET sweeps assume a cubic grid"
    if dt is None:
        dt = cfl_dt(cfg.dx, cfg.dy, cfg.dz, cfg.c, cfg.cfl)
    if S_E is None:
        S_E = jnp.zeros_like(U0[..., 0])
    if S_F is None:
        S_F = jnp.zeros_like(U0[..., 1:4])
    if dirset is None:
<<<<<<< HEAD
        dirset = DirectionSet(cfg.n_dirs)
=======
        dirset = default_dirset(cfg.n_dirs)
>>>>>>> upstream/main

    n_outer = -(-nsteps // cfg.vet_every)
    aux = (kappa_a, kappa_t, E_eq, S_E)

<<<<<<< HEAD
    def inner(U, f6):
        def body(U, _):
            return step_vet(U, dt, cfg, f6, kappa_a, kappa_t, E_eq, S_E, S_F), None
        return jax.lax.scan(body, U, None, length=cfg.vet_every)[0]

=======
>>>>>>> upstream/main
    U = U0
    f6 = None
    for _ in range(n_outer):
        f6 = compute_vet(U, aux, cfg, dirset)
<<<<<<< HEAD
        U = inner(U, f6)
=======
        U = _advance_frozen(U, f6, dt, cfg, kappa_a, kappa_t, E_eq, S_E, S_F)
>>>>>>> upstream/main
    return (U, f6) if return_tensor else U
