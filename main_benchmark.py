# main_benchmark.py
# Point-source benchmark for the M1 solver: compare the steady state against
# the free-streaming solution with absorption,
#   E(r) = L / (4 pi c r^2) exp(-kappa_a r),  F = c E r_hat.
#
# Run:  python main_benchmark.py            (GPU via autocvd, waits for a free card)
#       M1_DEVICE=cpu python main_benchmark.py   (CPU)
import os
from momentRTjax.devices import select_device
select_device()                       # must precede the jax import

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from momentRTjax.momentRT import M1Config, advance, cfl_dt


def main():
    # --------------------------
    # Grid / physical parameters
    # --------------------------
    Nx, Ny, Nz = 64, 64, 64
    dx = dy = dz = 1.0
    c = 1.0

    # Outflow BC: the analytic reference is an infinite-domain solution, so
    # periodic wrap-around would pollute the outer cells.
    cfg = M1Config(c=c, dx=dx, dy=dy, dz=dz, cfl=0.2, epsE=1e-12, bc="outflow")

    # --------------------------
    # Medium and source settings
    # --------------------------
    # kappa_t is the TOTAL (transport) opacity acting on the flux. With no
    # scattering it must equal kappa_a: only then is the free-streaming
    # reference (f = 1, E ~ exp(-kappa_a r)/r^2) an exact M1 steady state.
    # A kappa_t < kappa_a is unphysical and leaves a few-percent bias in E.
    kappa_a_val = 0.05
    kappa_s_val = 0.0
    kappa_t_val = kappa_a_val + kappa_s_val
    L = 1.0                # luminosity injected in the central cell

    # --------------------------
    # Initial condition
    # --------------------------
    E0 = jnp.ones((Nx, Ny, Nz)) * 1e-6
    F0 = jnp.zeros((Nx, Ny, Nz, 3))
    U0 = jnp.concatenate([E0[..., None], F0], axis=-1)

    kappa_a = jnp.ones((Nx, Ny, Nz)) * kappa_a_val
    kappa_t = jnp.ones((Nx, Ny, Nz)) * kappa_t_val
    E_eq    = jnp.zeros((Nx, Ny, Nz))

    cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
    cell_vol = dx * dy * dz
    S_E = jnp.zeros((Nx, Ny, Nz)).at[cx, cy, cz].set(L / cell_vol)
    S_F = jnp.zeros((Nx, Ny, Nz, 3))

    # --------------------------
    # Time step & integration: run until light has crossed the box twice so
    # the whole domain is in steady state.
    # --------------------------
    dt = cfl_dt(dx, dy, dz, c, cfg.cfl)
    t_end = 2.0 * max(Nx * dx, Ny * dy, Nz * dz) / c
    nsteps = int(np.ceil(t_end / dt))
    print(f"[M1] dt = {dt:.4f}, nsteps = {nsteps}, t_end = {nsteps*dt:.1f}")

    U = advance(U0, nsteps=nsteps, cfg=cfg,
                kappa_a=kappa_a, kappa_t=kappa_t, E_eq=E_eq,
                dt=dt, S_E=S_E, S_F=S_F)
    U.block_until_ready()

    if bool(jnp.isnan(U).any() | jnp.isinf(U).any()):
        raise RuntimeError("Solution contains NaN/Inf. Try reducing cfl, lowering L, "
                           "or increasing kappa_t_val slightly.")

    E_num, F_num = U[..., 0], U[..., 1:4]

    # --------------------------
    # Analytic reference
    # --------------------------
    x = (jnp.arange(Nx) - cx) * dx
    y = (jnp.arange(Ny) - cy) * dy
    z = (jnp.arange(Nz) - cz) * dz
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    r = jnp.sqrt(X*X + Y*Y + Z*Z)
    eps_r = 1e-6

    E_an = (L / (4.0 * jnp.pi * c)) * jnp.exp(-kappa_a_val * r) / jnp.maximum(r*r, eps_r)
    ux = X / jnp.maximum(r, eps_r)
    uy = Y / jnp.maximum(r, eps_r)
    uz = Z / jnp.maximum(r, eps_r)
    F_an = c * E_an
    F_an_vec = jnp.stack([F_an * ux, F_an * uy, F_an * uz], axis=-1)

    # --------------------------
    # Metrics on a spherical shell r_in <= r <= r_out.
    # r_in skips the source cell and its immediate neighbours (finite-cell
    # source + numerical diffusion); r_out stays inside the inscribed sphere.
    # --------------------------
    r_in  = 2.0 * dx
    r_out = 0.45 * Nx * dx
    mask = (r >= r_in) & (r <= r_out)

    rel_err_E = jnp.abs((E_num - E_an) / (E_an + 1e-12))
    mean_rel_err_E = float(jnp.mean(rel_err_E[mask]))
    med_rel_err_E  = float(jnp.median(rel_err_E[mask]))

    Fmag_num = jnp.linalg.norm(F_num, axis=-1)
    Fmag_an  = jnp.linalg.norm(F_an_vec, axis=-1)
    rel_err_F = jnp.abs((Fmag_num - Fmag_an) / (Fmag_an + 1e-12))
    mean_rel_err_F = float(jnp.mean(rel_err_F[mask]))

    Fhat_num = jnp.where((Fmag_num > 0)[..., None], F_num / (Fmag_num[..., None] + 1e-30), 0.0)
    Fhat_an  = jnp.stack([ux, uy, uz], axis=-1)
    cos_sim = jnp.clip(jnp.sum(Fhat_num * Fhat_an, axis=-1), -1.0, 1.0)
    mean_cos = float(jnp.mean(cos_sim[mask]))

    # Energy budget: in steady state the absorbed power must equal L minus
    # what leaves through the boundary (small here, tau_edge ~ 1.6).
    absorbed = float(jnp.sum(c * kappa_a * E_num) * cell_vol)

    print(f"[M1] shell {r_in:.0f} <= r <= {r_out:.1f}: {int(mask.sum())} cells")
    print(f"[M1] mean rel. error E   : {mean_rel_err_E:.3e}  (median {med_rel_err_E:.3e})")
    print(f"[M1] mean rel. error |F| : {mean_rel_err_F:.3e}")
    print(f"[M1] mean cosine(F, r_hat): {mean_cos:.4f}")
    print(f"[M1] absorbed power / L  : {absorbed / L:.4f}  (1 - exp(-kappa r_box/2) = {1-np.exp(-kappa_a_val*Nx*dx/2):.4f})")

    # --------------------------
    # Plots
    # --------------------------
    outdir = os.environ.get("SAVE_DIR", "plots_m1_benchmark")
    os.makedirs(outdir, exist_ok=True)
    mid = cz

    # Patch the singular centre cell of the reference for plotting only.
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
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "E_slices_xy.png"), dpi=150); plt.close()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(np.log10(np.asarray(Fmag_num[:, :, mid]) + 1e-12), origin='lower', cmap='viridis')
    plt.title("log10 |F| (numeric)"); plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(np.log10(np.asarray(Fmag_an[:, :, mid]) + 1e-12), origin='lower', cmap='viridis')
    plt.title("log10 |F| (analytic)"); plt.colorbar()
    plt.subplot(1, 3, 3)
    resF = np.log10(np.asarray(rel_err_F[:, :, mid]) + 1e-12)
    plt.imshow(resF, origin='lower', cmap='magma', vmin=-3, vmax=0); plt.title("log10 |rel. err |F||"); plt.colorbar()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "Fmag_slices_xy.png"), dpi=150); plt.close()

    ang_deg = np.degrees(np.arccos(np.clip(np.asarray(cos_sim[:, :, mid]), -1.0, 1.0)))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(np.asarray(cos_sim[:, :, mid]), origin='lower', vmin=0, vmax=1, cmap='plasma')
    plt.title("cosine(F_num, r̂)"); plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(ang_deg, origin='lower', vmin=0, vmax=10, cmap='magma')
    plt.title("angle error (deg)"); plt.colorbar()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "Fdir_metrics_xy.png"), dpi=150); plt.close()

    # Radial profiles: on-axis line and spherical-shell average
    r_line = (np.arange(cx+1, Nx) - cx) * dx
    E_line_num = np.asarray(E_num[cx+1:, cy, cz])
    E_line_an  = np.asarray(E_an [cx+1:, cy, cz])

    r_np = np.asarray(r); E_np = np.asarray(E_num); Ean_np = np.asarray(E_an)
    bins = np.arange(0.5, 0.5 * Nx * dx, 1.0)
    idx = np.digitize(r_np.ravel(), bins)
    r_shell = 0.5 * (bins[:-1] + bins[1:])
    E_shell = np.array([E_np.ravel()[idx == k].mean() for k in range(1, len(bins))])
    Ean_shell = np.array([Ean_np.ravel()[idx == k].mean() for k in range(1, len(bins))])

    plt.figure(figsize=(6, 4))
    plt.loglog(r_line, E_line_num, 'o', ms=3, label='numeric (x-axis)')
    plt.loglog(r_shell, E_shell, 's', ms=3, label='numeric (shell avg)')
    plt.loglog(r_line, E_line_an,  '-', lw=2, label='analytic')
    plt.xlabel("r (cells)"); plt.ylabel("E")
    plt.title("Radial profile of E")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "E_radial_profile.png"), dpi=150); plt.close()

    print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main()
