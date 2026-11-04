# main_test_debug.py
# Manual stepping of the M1 solver with a NaN guard, to catch the first bad
# step. Drives the source through the equilibrium field E_eq (emission
# c*kappa_a*E_eq) rather than the explicit S_E path of main_benchmark.py.
#
# Run:  python main_test_debug.py              (GPU via autocvd)
#       M1_DEVICE=cpu python main_test_debug.py (CPU)
import os
from momentRTjax.devices import select_device
select_device()                       # must precede the jax import

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
# Uncomment to debug without JIT:
# jax.config.update("jax_disable_jit", True)

from momentRTjax.momentRT import (
    M1Config, step_m1, cfl_dt, enforce_positivity_and_causality,
)


def step_m1_safe(U, dt, cfg, kappa_a, kappa_t, E_eq, S_E, S_F):
    """Compiled step followed by an extra positivity/causality clamp."""
    U = step_m1(U, dt, cfg, kappa_a, kappa_t, E_eq, S_E, S_F)
    E = jnp.maximum(U[..., 0], cfg.epsE)
    F = U[..., 1:4]
    E, F = enforce_positivity_and_causality(E, F, cfg.c, epsE=cfg.epsE)
    return jnp.concatenate([E[..., None], F], axis=-1)


# --------------------------
# Problem setup
# --------------------------
Nx, Ny, Nz = 64, 64, 64
dx = dy = dz = 1.0
c = 1.0
cfg = M1Config(c=c, dx=dx, dy=dy, dz=dz, cfl=0.2, epsE=1e-12, bc="outflow")

kappa_a_val = 0.05    # must be > 0 when the source is expressed through E_eq
kappa_s_val = 0.0
kappa_t_val = kappa_a_val + kappa_s_val   # total opacity on the flux (see main_benchmark.py)
assert kappa_a_val > 0.0, "kappa_a_val must be > 0 when using E_eq as source."

kappa_a = jnp.ones((Nx, Ny, Nz)) * kappa_a_val
kappa_t = jnp.ones((Nx, Ny, Nz)) * kappa_t_val

L = 1e-1
cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
cell_vol = dx * dy * dz
E_eq_center = L / (c * kappa_a_val * cell_vol)   # emission c*kappa_a*E_eq = L/V
E_eq = jnp.zeros((Nx, Ny, Nz)).at[cx, cy, cz].set(E_eq_center)

# No explicit volumetric sources in this script
S_E = jnp.zeros((Nx, Ny, Nz))
S_F = jnp.zeros((Nx, Ny, Nz, 3))

E0 = jnp.ones((Nx, Ny, Nz)) * 1e-6
F0 = jnp.zeros((Nx, Ny, Nz, 3))
U = jnp.concatenate([E0[..., None], F0], axis=-1)

dt = cfl_dt(dx, dy, dz, c, cfg.cfl)
nsteps = int(os.environ.get("NSTEPS", 400))
print(f"[debug] dt = {dt:.4f}, nsteps = {nsteps}")

# --------------------------
# Manual stepping with NaN guard
# --------------------------
for n in range(nsteps):
    U = step_m1_safe(U, dt, cfg, kappa_a, kappa_t, E_eq, S_E, S_F)
    if bool(jnp.isnan(U).any() | jnp.isinf(U).any()):
        Ecur = U[..., 0]; Fcur = U[..., 1:4]
        print(f"[NaN] at step {n}")
        print("  E range:", float(jnp.nanmin(Ecur)), float(jnp.nanmax(Ecur)))
        print("  |F| max:", float(jnp.nanmax(jnp.linalg.norm(Fcur, axis=-1))))
        print("  E_eq_center:", float(E_eq_center))
        print("  kappa_a_val:", kappa_a_val, "kappa_t_val:", kappa_t_val, "dt:", dt)
        raise SystemExit(1)

print("OK: no NaNs after", nsteps, "steps.")

# --------------------------
# Analytic reference and metrics
# --------------------------
x = (jnp.arange(Nx) - cx) * dx
y = (jnp.arange(Ny) - cy) * dy
z = (jnp.arange(Nz) - cz) * dz
X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
r = jnp.sqrt(X*X + Y*Y + Z*Z)
eps_r = 1e-6

E_num, F_num = U[..., 0], U[..., 1:4]

E_an = (L / (4.0 * jnp.pi * c)) * jnp.exp(-kappa_a_val * r) / jnp.maximum(r*r, eps_r)
ux = X / jnp.maximum(r, eps_r)
uy = Y / jnp.maximum(r, eps_r)
uz = Z / jnp.maximum(r, eps_r)
F_an_vec = jnp.stack([c * E_an * ux, c * E_an * uy, c * E_an * uz], axis=-1)

# Only cells the light front has reached (minus a margin) are in steady state.
r_reached = max(0.0, c * dt * nsteps - 4.0 * dx)
mask = (r >= 2.0 * dx) & (r <= min(r_reached, 0.45 * Nx * dx))

rel_err_E = jnp.abs((E_num - E_an) / (E_an + 1e-12))
mean_rel_err_E = float(jnp.mean(rel_err_E[mask]))
Fmag = jnp.linalg.norm(F_num, axis=-1)
Fhat_num = jnp.where((Fmag > 0)[..., None], F_num / (Fmag[..., None] + 1e-30), 0.0)
Fhat_an  = jnp.stack([ux, uy, uz], axis=-1)
cos_sim = jnp.clip(jnp.sum(Fhat_num * Fhat_an, axis=-1), -1.0, 1.0)
mean_cos = float(jnp.mean(cos_sim[mask]))
print(f"[M1] steady-state shell: 2 <= r <= {float(min(r_reached, 0.45*Nx*dx)):.1f} ({int(mask.sum())} cells)")
print(f"[M1] mean relative error E (masked): {mean_rel_err_E:.3e}")
print(f"[M1] mean cosine similarity(F, r̂) (masked): {mean_cos:.3f}")

# --------------------------
# Plots (XY mid-slice)
# --------------------------
outdir = os.environ.get("SAVE_DIR", "plots_m1_debug")
os.makedirs(outdir, exist_ok=True)
mid = cz
E_an_plot = E_an.at[cx, cy, cz].set(E_num[cx, cy, cz])

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(np.log10(np.asarray(E_num[:, :, mid]) + 1e-12), origin='lower', cmap='inferno')
plt.title("log10 E (numeric)"); plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(np.log10(np.asarray(E_an_plot[:, :, mid]) + 1e-12), origin='lower', cmap='inferno')
plt.title("log10 E (analytic)"); plt.colorbar()
plt.subplot(1, 3, 3)
res = np.log10(np.asarray(rel_err_E[:, :, mid]) + 1e-12)
plt.imshow(res, origin='lower', cmap='magma', vmin=-3, vmax=0); plt.title("log10 |rel. err E|"); plt.colorbar()
plt.tight_layout(); plt.savefig(os.path.join(outdir, "E_xy.png"), dpi=150); plt.close()

Fmag_num = jnp.linalg.norm(F_num, axis=-1)
Fmag_an  = jnp.linalg.norm(F_an_vec, axis=-1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(np.log10(np.asarray(Fmag_num[:, :, mid]) + 1e-12), origin='lower', cmap='viridis')
plt.title("log10 |F| (numeric)"); plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(np.log10(np.asarray(Fmag_an[:, :, mid]) + 1e-12), origin='lower', cmap='viridis')
plt.title("log10 |F| (analytic)"); plt.colorbar()
plt.subplot(1, 3, 3)
resF = np.log10(np.asarray(jnp.abs((Fmag_num - Fmag_an) / (Fmag_an + 1e-12))[:, :, mid]) + 1e-12)
plt.imshow(resF, origin='lower', cmap='magma', vmin=-3, vmax=0); plt.title("log10 |rel. err |F||"); plt.colorbar()
plt.tight_layout(); plt.savefig(os.path.join(outdir, "F_xy.png"), dpi=150); plt.close()

print(f"Saved plots to: {outdir}")
