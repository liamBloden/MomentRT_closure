# Moment_based — M1 radiative transfer in JAX

Minimal 3D gray two-moment (M1) radiative transfer solver.

- `momentRTjax/momentRT.py` — solver: first-order finite volume, HLLE fluxes
  (wave speeds ±c), Levermore M1 closure, implicit absorption/flux damping,
  explicit volumetric sources, positivity/causality limiter.
  Boundary conditions: `M1Config(bc="outflow")` (default) or `"periodic"`.
- `momentRTjax/vet.py` — Variable Eddington Tensor closure. A short-characteristics
  formal solution (plane-by-plane upwind sweeps over an octant-symmetric
  Fibonacci direction set, `VETConfig.n_dirs`, default 192) gives
  `f_ij = P_ij / E`; the moment equations are then advanced with `P = f E`
  using the same HLLE flux, sources and limiters as M1. The tensor is rebuilt
  every `VETConfig.vet_every` steps (pure absorber + fixed sources: once is enough).
- `momentRTjax/turbulence.py` — correlated log-normal ("turbulent") fields,
  ported from Ray-trax `create_turbulent_3D.py`: `gaussian_field`,
  `lognormal_field` (log-k bump or power-law spectrum, correlation length in
  cells), `turbulent_opacity` → (kappa_a, kappa_t) with optional albedo,
  `random_point_sources`, and `random_setup(key, ...)` returning everything
  `advance` / `advance_vet` need. Intended for dataset generation.
- `momentRTjax/devices.py` — `select_device()`: picks a free GPU with autocvd
  (default mode, waits until a card is completely free). Call it before
  importing JAX. `M1_DEVICE=cpu` forces the CPU.
- `main_benchmark.py` — point source in a 64³ box vs. the analytic
  free-streaming solution `E = L exp(-κ_a r) / (4π c r²)`. Prints shell-averaged
  relative errors and writes plots to `plots_m1_benchmark/` (override with `SAVE_DIR`).
- `main_vet_benchmark.py` — M1 vs VET on (A) one point source, (B) two point
  sources on the x axis and (C) five sources of different luminosity. The exact
  field is the superposition of `L e^{-κr}/(4π c r²)`. Prints an error table
  per closure, writes `plots_vet_benchmark/`. Env: `N_DIRS`, `VET_EVERY`
  (0 = once), `SAVE_DIR`.
- `example_turbulent.py` — point sources in a turbulent opacity field, M1 vs
  VET (no analytic reference: reports energy budget and the M1–VET
  difference). Env: `N`, `N_SOURCES`, `SIGMA_G`, `CORR_LENGTH`, `KAPPA_MEAN`,
  `SEED`, `SAVE_DIR`. Writes `plots_turbulent/`.
- `main_test_debug.py` — manual stepping with a NaN guard; the source is driven
  through `E_eq` instead of `S_E`. `NSTEPS` env var sets the step count.

```bash
source /export/scratch/lbranca/ray_trax_env/bin/activate
python main_benchmark.py                 # GPU (autocvd)
M1_DEVICE=cpu python main_benchmark.py   # CPU, ~15 s
M1_DEVICE=cpu python main_vet_benchmark.py   # CPU, a few minutes
```

Notes
<<<<<<< HEAD
- Stability: `dt · c · (1/dx + 1/dy + 1/dz) ≤ 1`; `cfl_dt` uses `cfl=0.2` of that.
=======
- Variables: `U = [E, Fx, Fy, Fz]` with `F` the *physical* energy flux
  (`|F| ≤ cE`), so the system is `∂E/∂t + ∇·F = 0`, `∂F/∂t + c²∇·P = 0` with
  wave speeds ±c. (An earlier version used `c·F` in the energy equation, which
  is only consistent at `c = 1`.)
- Scheme: first-order unsplit finite volume, Rusanov/HLLE flux with speeds ±c,
  forward Euler, implicit (exact, local) absorption and flux damping, explicit
  volumetric sources; Lie splitting between the parts.
- Stability: `dt · c · (1/dx + 1/dy + 1/dz) ≤ 1`; `cfl_dt` uses `cfl=0.2` of that.
- Diffusion limit (`M1Config.ap`, default on): the Rusanov dissipation `c h/2`
  would swamp the physical diffusivity `c/(3κ_t)` in optically thick cells, so
  the HLLE speed is scaled by `1/(1 + κ_t h)²` at each face (`ap_speed_factor`).
  1D scattering slab (κ_a = 0.01), steady-state decay length vs. diffusion theory:
  `κ_t h = 2`: 4.7 (theory 4.1, 8.2 without); `κ_t h = 5`: 2.8 (theory 2.6, 7.5
  without). Free-streaming accuracy is unchanged (slightly better at κh ≪ 1).
>>>>>>> upstream/main
- `kappa_t` is the total opacity on the flux (absorption + scattering). Set
  `kappa_t = kappa_a` for a pure absorber; a smaller value biases E by a few %.
- VET: the grid must be cubic (`dx == dy == dz`). The formal solution is first
  order and shows ray effects between discrete directions; the *ratio* `P/E` is
<<<<<<< HEAD
  much less affected than `E_f` itself (tensor error ~3 % at 192 directions).
- Remaining M1 error (~2.5 % in E) is first-order numerical diffusion plus the
  four-lobed grid anisotropy of the closure; MUSCL reconstruction is the next step.

## Results (64³, κ = 0.05, outflow, first order, 2026-10-15, CPU)

Mean relative error of E on the shell 2 ≤ r ≤ 0.45 N around every source:

| case          | M1     | VET (192 dirs, smooth=1) |
|---------------|--------|--------------------------|
| single source | 2.5 %  | 7.7 %                    |
| two sources   | 19.2 % | 10.5 %                   |
| five sources  | 26.8 % | 8.6 %                    |
=======
  much less affected than `E_f` itself. The quadrature weights are corrected
  (least-squares, `quadrature_weights`) so that `∫dΩ` and `∫n_i n_j dΩ` are
  exact: an isotropic field gives `f_ij = δ_ij/3` to 1e-6 (uniform weights were
  off by ~2 % at 192 directions). Pass one `DirectionSet` (or rely on the cached
  `default_dirset`) across calls so `compute_vet` is not retraced.
- Remaining M1 error (~2.5 % in E) is first-order numerical diffusion plus the
  four-lobed grid anisotropy of the closure; MUSCL reconstruction is the next step.

## Results (64³, κ = 0.05, outflow, first order, 2026-11-04, CPU)

Mean relative error of E on the shell 2 ≤ r ≤ 0.45 N around every source
(`ap=True`, corrected quadrature weights):

| case          | M1     | VET (192 dirs, smooth=1) |
|---------------|--------|--------------------------|
| single source | 2.3 %  | 7.5 %                    |
| two sources   | 20.1 % | 11.3 %                   |
| five sources  | 28.2 % | 8.7 %                    |

(Previous table, 2026-10-15, uniform weights and no AP scaling: M1 2.5 / 19.2 /
26.8 %, VET 7.7 / 10.5 / 8.6 %. The AP scaling removes ~9 % of the numerical
dissipation at κh = 0.05, which was partly smearing the closure errors in the
multi-source cases; set `ap=False` to reproduce the old numbers. The corrected
weights improve VET by 0.3–0.9 points.)
>>>>>>> upstream/main

M1 is exact for one point source by construction, so it wins there; its error
in the multi-source cases is the well-known beam-merging failure (bright band
on the midplane between two sources). The VET error is dominated by ray effects
of the first-order formal solution (cell-scale noise in `f_ij`, which the flux
equation amplifies at large r). At 48³ the single-source VET error goes
14 % → 6.6 % with one smoothing pass, 5.0 % with 384 directions and two passes.
Lowering the HLLE speeds to the tensor eigenvalues (`VETConfig.wave_floor < 1`)
made things worse and is off by default.

## Important: device selection
`momentRTjax/__init__.py` is lazy on purpose. JAX reads `JAX_PLATFORMS` at
import, so `select_device()` must run before anything imports `jax`; it now
raises if `jax` is already imported. (A first version imported JAX from the
package `__init__`, which silently put "CPU" runs on GPU 0.)
