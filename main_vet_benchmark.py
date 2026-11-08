# main_vet_benchmark.py
# M1 vs Variable Eddington Tensor (VET) closure on two steady-state problems
# with exact solutions (pure absorber, free streaming):
#   A) one point source at the centre
#   B) two point sources on the x axis — the exact field is the superposition,
#      which M1 cannot represent where the beams cross.
#   C) five point sources of different luminosity scattered through the box.
#
# Run:  python main_vet_benchmark.py              (GPU via autocvd)
#       M1_DEVICE=cpu python main_vet_benchmark.py (CPU, ~3 min)
# Env: N_DIRS (default 192, rounded to a multiple of 8), VET_EVERY (0 = compute the tensor once), SAVE_DIR
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
from momentRTjax.vet import VETConfig, advance_vet, DirectionSet


def analytic(L_list, pos_list, X, Y, Z, kappa, c):
    E = jnp.zeros_like(X); F = jnp.zeros(X.shape + (3,))
    for L, (px, py, pz) in zip(L_list, pos_list):
        dx_, dy_, dz_ = X - px, Y - py, Z - pz
        r = jnp.sqrt(dx_**2 + dy_**2 + dz_**2)
        r2 = jnp.maximum(r * r, 1e-6)
        Ei = L * jnp.exp(-kappa * r) / (4.0 * jnp.pi * c * r2)
        E = E + Ei
        F = F + c * Ei[..., None] * jnp.stack([dx_, dy_, dz_], -1) / jnp.maximum(r, 1e-6)[..., None]
    return E, F


def run_case(name, L_list, pos_list, N, h, c, kappa, t_end, n_dirs, vet_every):
    dt = cfl_dt(h, h, h, c, 0.2)
    nsteps = int(np.ceil(t_end / dt))
    if vet_every <= 0:          # no scattering: tensor independent of E -> compute once
        vet_every = nsteps
    nsteps = -(-nsteps // vet_every) * vet_every
    cfg_m1  = M1Config(c=c, dx=h, dy=h, dz=h, cfl=0.2, bc="outflow")
    cfg_vet = VETConfig(c=c, dx=h, dy=h, dz=h, cfl=0.2, bc="outflow",
                        n_dirs=n_dirs, vet_every=vet_every)

    U0 = jnp.concatenate([jnp.full((N, N, N, 1), 1e-6), jnp.zeros((N, N, N, 3))], -1)
    kappa_a = jnp.full((N, N, N), kappa)
    kappa_t = kappa_a                              # pure absorber
    E_eq = jnp.zeros((N, N, N))
    S_E = jnp.zeros((N, N, N))
    for L, p in zip(L_list, pos_list):
        S_E = S_E.at[p].add(L / h**3)

    U_m1 = advance(U0, nsteps, cfg_m1, kappa_a, kappa_t, E_eq, dt=dt, S_E=S_E)
    U_m1.block_until_ready()
    U_vet, f6 = advance_vet(U0, nsteps, cfg_vet, kappa_a, kappa_t, E_eq, dt=dt, S_E=S_E,
                            dirset=DirectionSet(n_dirs), return_tensor=True)
    U_vet.block_until_ready()
    for tag, U in (("M1", U_m1), ("VET", U_vet)):
        if not bool(jnp.isfinite(U).all()):
            raise RuntimeError(f"{name}/{tag}: NaN/Inf in solution")

    x = (jnp.arange(N) - N // 2) * h
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    pos_phys = [((p[0] - N//2) * h, (p[1] - N//2) * h, (p[2] - N//2) * h) for p in pos_list]
    E_an, F_an = analytic(L_list, pos_phys, X, Y, Z, kappa, c)

    # mask: >= 2 cells from every source, inside the inscribed sphere
    r0 = jnp.sqrt(X**2 + Y**2 + Z**2)
    mask = r0 <= 0.45 * N * h
    for (px, py, pz) in pos_phys:
        mask = mask & (jnp.sqrt((X-px)**2 + (Y-py)**2 + (Z-pz)**2) >= 2.0 * h)

    out = {}
    for tag, U in (("M1", U_m1), ("VET", U_vet)):
        E, F = U[..., 0], U[..., 1:4]
        relE = jnp.abs(E - E_an) / E_an
        Fm, Fam = jnp.linalg.norm(F, axis=-1), jnp.linalg.norm(F_an, axis=-1)
        # vector flux error normalised by c E_an (|F_an| vanishes where beams cancel)
        relF = jnp.linalg.norm(F - F_an, axis=-1) / (c * E_an)
        cos = jnp.sum(F * F_an, -1) / (Fm * Fam + 1e-30)
        out[tag] = dict(E=E, relE=relE, relF=relF,
                        meanE=float(relE[mask].mean()), medE=float(jnp.median(relE[mask])),
                        maxE=float(relE[mask].max()), meanF=float(relF[mask].mean()),
                        cos=float(cos[mask].mean()))
    print(f"\n[{name}]  N={N}, steps={nsteps}, dt={dt:.4f}, n_dirs={n_dirs}, {int(mask.sum())} cells in mask")
    print(f"  {'closure':8s} {'mean|dE|/E':>11s} {'median':>8s} {'max':>8s} {'mean|dF|/cE':>12s} {'cos(F,F_an)':>12s}")
    for tag in ("M1", "VET"):
        o = out[tag]
        print(f"  {tag:8s} {o['meanE']:11.3e} {o['medE']:8.3e} {o['maxE']:8.3e} {o['meanF']:12.3e} {o['cos']:12.4f}")
    return out, E_an, f6, mask


def plot_case(name, out, E_an, f6, N, outdir):
    mid = N // 2
    lo = np.log10(float(E_an[:, :, mid].min()))
    hi = np.log10(float(np.percentile(np.asarray(E_an[:, :, mid]), 99.5)))
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    for a, (title, E) in zip(ax[0], (("analytic", E_an), ("M1", out["M1"]["E"]), ("VET", out["VET"]["E"]))):
        im = a.imshow(np.log10(np.asarray(E[:, :, mid])), origin="lower", cmap="inferno", vmin=lo, vmax=hi)
        a.set_title(f"log10 E — {title}"); plt.colorbar(im, ax=a)
    for a, tag in zip(ax[1][:2], ("M1", "VET")):
        im = a.imshow(np.log10(np.asarray(out[tag]["relE"][:, :, mid]) + 1e-12), origin="lower",
                      cmap="magma", vmin=-3, vmax=0)
        a.set_title(f"log10 |rel. err E| — {tag}"); plt.colorbar(im, ax=a)
    im = ax[1][2].imshow(np.asarray(f6[:, :, mid, 0]), origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax[1][2].set_title("VET f_xx"); plt.colorbar(im, ax=ax[1][2])
    fig.suptitle(f"{name} (z = mid-plane)"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}_E_xy.png"), dpi=150); plt.close(fig)

    # profile along the x axis through the sources
    xs = np.arange(N) - mid
    plt.figure(figsize=(7, 4))
    plt.semilogy(xs, np.asarray(E_an[:, mid, mid]), "k-", lw=2, label="analytic")
    plt.semilogy(xs, np.asarray(out["M1"]["E"][:, mid, mid]), "o", ms=3, label="M1")
    plt.semilogy(xs, np.asarray(out["VET"]["E"][:, mid, mid]), "s", ms=3, label="VET")
    plt.xlabel("x (cells)"); plt.ylabel("E"); plt.title(f"{name}: E along the x axis")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_profile_x.png"), dpi=150); plt.close()


def main():
    N, h, c, kappa = 64, 1.0, 1.0, 0.05
    t_end = 2.0 * N * h / c
    n_dirs = int(os.environ.get("N_DIRS", 192))
    vet_every = int(os.environ.get("VET_EVERY", 0))   # 0 = once per run
    outdir = os.environ.get("SAVE_DIR", "plots_vet_benchmark")
    os.makedirs(outdir, exist_ok=True)
    mid = N // 2

    cases = [
        ("single_source", [1.0], [(mid, mid, mid)]),
        ("two_sources",   [1.0, 1.0], [(mid - 12, mid, mid), (mid + 12, mid, mid)]),
        ("five_sources",  [1.0, 0.5, 2.0, 0.3, 1.0],
                          [(mid, mid, mid), (mid - 14, mid + 6, mid - 4), (mid + 10, mid + 12, mid + 8),
                           (mid + 6, mid - 15, mid - 10), (mid - 9, mid - 8, mid + 13)]),
    ]
    for name, Ls, ps in cases:
        out, E_an, f6, _ = run_case(name, Ls, ps, N, h, c, kappa, t_end, n_dirs, vet_every)
        plot_case(name, out, E_an, f6, N, outdir)
    print(f"\nSaved plots to: {outdir}")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main()
