# generate_fields.py
# Build a set of random problems for the moment solver: an absorption field
# (turbulent log-normal opacity) and an emission field (stars in the densest
# gas). No solver is run here.
#
# Stars follow a density criterion, as in Ray-trax `process_density_field`:
# every cell above a percentile of the opacity field is a star (99 = the
# densest 1 % of the cells). The percentile sets the number of stars,
# (1 - percentile/100) * N^3, and is one of the per-sample parameters.
#
# All options live in an options file (fields_options.toml, documented there).
# Sample i is fully determined by its seed (seed0 + i) and that file. Drawn
# from the seed, per sample: percentile, kappa_mean, sigma_g, corr_length (each
# a [LO, HI] range, or a single number to fix it) and one luminosity per star.
#
# Output (option `out`):
#   sample_<seed>.npz   kappa_a, S_E (float32), positions, luminosities, parameters;
#                       read back with momentRTjax.turbulence.load_setup
#   manifest.json       options + one row per sample (parameters, star count, optical depths)
#   preview.png         opacity slices of the first `preview` samples with their star cells
#
# Run:  python generate_fields.py                   (reads fields_options.toml next to this script)
#       python generate_fields.py my_options.toml   (another options file)
# Runs on the CPU by default (a 64^3 sample takes a fraction of a second);
# M1_DEVICE=gpu picks a GPU through autocvd like the other scripts.
import os
os.environ.setdefault("M1_DEVICE", "cpu")
from momentRTjax.devices import select_device
select_device()

import argparse
import json
import tomllib
from types import SimpleNamespace
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # also pins the random streams (x32 draws differ)
from jax import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from momentRTjax.turbulence import random_setup, save_setup


DEFAULT_OPTIONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fields_options.toml")
RANGES = ("percentile", "kappa_mean", "sigma_g", "corr_length", "L_range")   # [LO, HI] or one number
OPTIONS = set(RANGES) | {"out", "n", "seed0", "N", "preview", "overwrite", "margin", "spectrum", "albedo"}


def load_options(path):
    """Read the options file: every option must be present, none unknown."""
    with open(path, "rb") as f:
        opts = tomllib.load(f)
    missing, unknown = sorted(OPTIONS - opts.keys()), sorted(opts.keys() - OPTIONS)
    if missing or unknown:
        raise SystemExit(f"{path}: missing options {missing}, unknown options {unknown}")
    for k in RANGES:
        v = opts[k] if isinstance(opts[k], list) else [opts[k], opts[k]]
        if len(v) != 2:
            raise SystemExit(f"{path}: {k} must be one number or [LO, HI]")
        opts[k] = (float(v[0]), float(v[1]))
    if not 0.0 <= opts["percentile"][0] <= opts["percentile"][1] < 100.0:
        raise SystemExit(f"{path}: percentile needs 0 <= LO <= HI < 100")
    if opts["spectrum"] not in ("logk", "powerlaw"):
        raise SystemExit(f"{path}: spectrum must be 'logk' or 'powerlaw'")
    return SimpleNamespace(**opts)


def draw_params(key, a):
    """Per-sample generator parameters, drawn from the sample's key."""
    k_pct, k_kappa, k_sigma, k_corr = random.split(key, 4)

    def uniform(k, lo, hi, log=False):
        if lo == hi:                     # fixed parameter: return it exactly
            return lo
        u = float(random.uniform(k))
        return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))) if log else lo + u * (hi - lo)

    p_lo, p_hi = a.percentile            # drawn through the star fraction 100 - percentile
    return dict(percentile=p_lo if p_lo == p_hi
                else 100.0 - uniform(k_pct, 100.0 - p_lo, 100.0 - p_hi, log=True),
                kappa_mean=uniform(k_kappa, *a.kappa_mean, log=True),
                sigma_g=uniform(k_sigma, *a.sigma_g),
                corr_length=uniform(k_corr, *a.corr_length, log=True))


def plot_preview(panels, path):
    """log10 opacity in the z-plane holding the most stars of each sample, its star cells marked."""
    ncol = min(4, len(panels))
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.9 * nrow), squeeze=False)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    for ax, (row, kappa_slice, stars, z0) in zip(axes.ravel(), panels):
        im = ax.imshow(np.log10(kappa_slice), origin="lower", cmap="cividis")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.scatter(stars[:, 1], stars[:, 0], s=16, c="white", edgecolors="black", linewidths=0.6,
                   label="star cell")
        ax.set_title(f"seed {row['seed']}:  kappa={row['kappa_mean']:.3f}  sigma={row['sigma_g']:.2f}  "
                     f"corr={row['corr_length']:.0f}\npercentile {row['percentile']:.2f}: "
                     f"{row['n_stars']} stars ({len(stars)} at z={z0})", fontsize=9)
        ax.set_xlabel("y (cells)"); ax.set_ylabel("x (cells)")
    axes[0][0].legend(loc="upper left", fontsize=7, framealpha=0.85)
    fig.suptitle("log10 opacity per cell in the z-plane with the most stars; "
                 "stars = cells above the sample's opacity percentile")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    cli = argparse.ArgumentParser(description="Random opacity + star fields for the moment solver.")
    cli.add_argument("options", nargs="?", default=DEFAULT_OPTIONS,
                     help="options file (default: fields_options.toml next to this script)")
    options_path = cli.parse_args().options
    a = load_options(options_path)
    print(f"[fields] options from {options_path}")
    manifest_path = os.path.join(a.out, "manifest.json")
    if os.path.exists(manifest_path) and not a.overwrite:
        raise SystemExit(f"{manifest_path} exists: change `out` or set overwrite = true in the options file")
    os.makedirs(a.out, exist_ok=True)
    shape = (a.N, a.N, a.N)

    rows, panels = [], []
    for i in range(a.n):
        seed = a.seed0 + i
        k_par, k_setup = random.split(random.PRNGKey(seed))
        par = draw_params(k_par, a)
        setup = random_setup(k_setup, shape, albedo=a.albedo, L_range=a.L_range,
                             margin=a.margin, spectrum=a.spectrum,
                             source_percentile=par["percentile"], kappa_mean=par["kappa_mean"],
                             sigma_g=par["sigma_g"], corr_length=par["corr_length"])
        name = f"sample_{seed:06d}.npz"
        save_setup(os.path.join(a.out, name), setup, seed=seed, albedo=a.albedo, **par)

        kappa = np.asarray(setup["kappa_t"])
        pos = np.asarray(setup["positions"])
        ksrc = kappa[pos[:, 0], pos[:, 1], pos[:, 2]]           # optical depth of the star cells (h = 1)
        row = dict(file=name, seed=seed, **par,
                   tau_box=par["kappa_mean"] * a.N,             # mean optical depth across the box
                   n_stars=len(pos), L_tot=float(np.asarray(setup["luminosities"]).sum()),
                   kappa_min=float(kappa.min()), kappa_max=float(kappa.max()),
                   kappa_star_min=float(ksrc.min()), kappa_star_median=float(np.median(ksrc)))
        rows.append(row)
        if i < a.preview:
            z0 = int(np.bincount(pos[:, 2], minlength=a.N).argmax())   # z-plane holding the most stars
            panels.append((row, kappa[:, :, z0], pos[pos[:, 2] == z0], z0))
        if (i + 1) % 50 == 0 or i + 1 == a.n:
            print(f"[fields] {i + 1}/{a.n}")

    with open(manifest_path, "w") as f:
        json.dump(dict(options=vars(a), h=1.0, samples=rows), f, indent=1)
    if panels:
        plot_preview(panels, os.path.join(a.out, "preview.png"))

    col = lambda k: np.array([r[k] for r in rows])
    print(f"[fields] {a.n} samples ({a.N}^3) in {a.out}")
    print(f"  percentile             : {col('percentile').min():.3f} .. {col('percentile').max():.3f}")
    print(f"  stars per sample       : {col('n_stars').min()} .. {col('n_stars').max()} "
          f"(median {int(np.median(col('n_stars')))})")
    print(f"  box optical depth      : {col('tau_box').min():.2f} .. {col('tau_box').max():.2f} "
          f"(median {np.median(col('tau_box')):.2f})")
    print(f"  optical depth of a star cell (per-sample median): {col('kappa_star_median').min():.3f} .. "
          f"{col('kappa_star_median').max():.3f}")


if __name__ == "__main__":
    main()
