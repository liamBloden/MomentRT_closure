# example_turbulent.py
# Point sources in a turbulent (correlated log-normal) opacity field:
# M1 vs VET. No analytic reference here, so the script reports the energy
# budget of each run and the difference between the closures.
#
# Run:  python example_turbulent.py               (GPU via autocvd)
#       M1_DEVICE=cpu python example_turbulent.py  (CPU, ~4 min at 64^3)
# Env:  N (grid, 64), N_SOURCES (3), SIGMA_G (1.0), CORR_LENGTH (10),
#       KAPPA_MEAN (0.05), SEED (0), SAVE_DIR (plots_turbulent)
import os
from momentRTjax.devices import select_device
select_device()

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from momentRTjax.momentRT import M1Config, advance, cfl_dt
from momentRTjax.vet import VETConfig, advance_vet
from momentRTjax.turbulence import random_setup, optical_depth_stats


def main():
    N = int(os.environ.get("N", 64))
    n_src = int(os.environ.get("N_SOURCES", 3))
    sigma_g = float(os.environ.get("SIGMA_G", 1.0))
    corr = float(os.environ.get("CORR_LENGTH", 10))
    kmean = float(os.environ.get("KAPPA_MEAN", 0.05))
    seed = int(os.environ.get("SEED", 0))
    outdir = os.environ.get("SAVE_DIR", "plots_turbulent")
    os.makedirs(outdir, exist_ok=True)
    h, c = 1.0, 1.0

    setup = random_setup(jax.random.PRNGKey(seed), (N, N, N), n_sources=n_src,
                         kappa_mean=kmean, sigma_g=sigma_g, corr_length=corr)
    kappa_a, kappa_t, S_E, U0 = setup["kappa_a"], setup["kappa_t"], setup["S_E"], setup["U0"]
    print(f"[turb] N={N}, sources at {np.asarray(setup['positions']).tolist()}, "
          f"L={np.round(np.asarray(setup['luminosities']), 3).tolist()}")
    print(f"[turb] kappa range {float(kappa_t.min()):.3e} .. {float(kappa_t.max()):.3e}, "
          f"mean box optical depth per axis {np.round(optical_depth_stats(kappa_t, h), 2)}")

    dt = cfl_dt(h, h, h, c, 0.2)
    nsteps = int(np.ceil(2.0 * N * h / c / dt))
    cfg_m1 = M1Config(c=c, dx=h, dy=h, dz=h, cfl=0.2, bc="outflow")
    cfg_vet = VETConfig(c=c, dx=h, dy=h, dz=h, cfl=0.2, bc="outflow",
                        n_dirs=192, vet_every=nsteps, smooth=1)

    U_m1 = advance(U0, nsteps, cfg_m1, kappa_a, kappa_t, setup["E_eq"], dt=dt, S_E=S_E)
    U_m1.block_until_ready()
    U_vet, f6 = advance_vet(U0, nsteps, cfg_vet, kappa_a, kappa_t, setup["E_eq"],
                            dt=dt, S_E=S_E, return_tensor=True)
    U_vet.block_until_ready()

    L_tot = float(S_E.sum() * h**3)
    print(f"\n[turb] steps={nsteps}, L_tot={L_tot:.3f}")
    print(f"  {'closure':8s} {'absorbed/L':>11s} {'max E':>10s} {'<|F|/cE>':>9s}  finite")
    E = {}
    for tag, U in (("M1", U_m1), ("VET", U_vet)):
        E[tag] = U[..., 0]; F = U[..., 1:4]
        absorbed = float((c * kappa_a * E[tag]).sum() * h**3) / L_tot
        fbar = float((jnp.linalg.norm(F, axis=-1) / (c * E[tag])).mean())
        print(f"  {tag:8s} {absorbed:11.4f} {float(E[tag].max()):10.3e} {fbar:9.3f}  {bool(jnp.isfinite(U).all())}")
    rel = jnp.abs(E["VET"] - E["M1"]) / (0.5 * (E["VET"] + E["M1"]))
    print(f"  mean |E_VET - E_M1| / E = {float(rel.mean()):.3e}  (median {float(jnp.median(rel)):.3e})")

    # ---- plots: mid-plane of the first source ----
    z0 = int(setup["positions"][0, 2])
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    im = ax[0][0].imshow(np.log10(np.asarray(kappa_t[:, :, z0])), origin="lower", cmap="cividis")
    ax[0][0].set_title("log10 kappa"); plt.colorbar(im, ax=ax[0][0])
    lo = np.log10(float(min(E["M1"][:, :, z0].min(), E["VET"][:, :, z0].min())))
    hi = np.log10(float(np.percentile(np.asarray(E["M1"][:, :, z0]), 99.5)))
    for a, tag in zip(ax[0][1:], ("M1", "VET")):
        im = a.imshow(np.log10(np.asarray(E[tag][:, :, z0])), origin="lower", cmap="inferno", vmin=lo, vmax=hi)
        a.set_title(f"log10 E — {tag}"); plt.colorbar(im, ax=a)
    im = ax[1][0].imshow(np.asarray(rel[:, :, z0]), origin="lower", cmap="magma", vmin=0, vmax=1)
    ax[1][0].set_title("|E_VET - E_M1| / E"); plt.colorbar(im, ax=ax[1][0])
    im = ax[1][1].imshow(np.asarray(f6[:, :, z0, 0]), origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax[1][1].set_title("VET f_xx"); plt.colorbar(im, ax=ax[1][1])
    fm1 = jnp.linalg.norm(U_m1[..., 1:4], axis=-1) / (c * E["M1"])
    im = ax[1][2].imshow(np.asarray(fm1[:, :, z0]), origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax[1][2].set_title("M1 |F|/cE"); plt.colorbar(im, ax=ax[1][2])
    for a in ax.ravel():
        a.scatter(setup["positions"][:, 1], setup["positions"][:, 0], s=12, c="cyan", marker="x")
    fig.suptitle(f"turbulent opacity, {n_src} sources, z = {z0}"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"turbulent_seed{seed}_xy.png"), dpi=150); plt.close(fig)
    print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main()
