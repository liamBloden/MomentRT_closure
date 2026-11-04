# Student project: a neural closure for two-moment radiative transfer

Goal in one sentence: replace the closure of the two-moment (E, F) radiative
transfer solver in this folder with a learned Eddington tensor that costs no
more than the M1 closure per time step, but reaches the accuracy of the
Variable Eddington Tensor (VET) method on multi-source problems.

## 1. Where you start

`momentRTjax/` is a small, fully differentiable JAX solver for the gray
two-moment equations

    dE/dt + div F        = -c kappa_a (E - E_eq) + S_E
    dF/dt + c^2 div P    = -c kappa_t F + S_F,        P_ij = f_ij E

on a cubic grid: first-order finite volume, HLLE fluxes, implicit local
sources, outflow or periodic boundaries. Two closures for the Eddington
tensor f_ij are implemented and benchmarked:

| closure | where | what it does | cost / step (64^3, CPU) | multi-source error |
|---|---|---|---|---|
| M1  | `momentRT.py::m1_closure` | algebraic: f_ij from the local ratio f = |F|/(cE) and direction F/|F| (Levermore) | 35 ms | 19 % (2 sources), 27 % (5 sources) |
| VET | `vet.py` | formal solution of the transfer equation on ~192 directions, f_ij = P_ij/E; rebuilt every k steps | 64 ms + 1.1 s per tensor build | 10 % / 9 % |

M1 is exact for a single point source and fails when beams cross (it merges
them). VET resolves crossing beams but costs an angular formal solution.
Read `README.md` for the results table and run `main_vet_benchmark.py` to
reproduce it (`M1_DEVICE=cpu` works, ~5 min).

Key files:
- `momentRT.py` — moment solver. `divergence_flux_generic(U, aux, closure_fn, ...)`
  already accepts any closure: `closure_fn(E, F, aux) -> (P_xx, P_yy, P_zz, P_xy, P_xz, P_yz)`.
  A neural closure plugs in here; `aux` can carry extra per-cell inputs.
- `vet.py` — formal solution (`formal_solution`, `eddington_tensor`) = your
  source of training labels.
- `main_vet_benchmark.py` — analytic multi-source tests = your validation.
- `devices.py` — call `select_device()` before importing JAX. Lab rule: GPUs
  only via autocvd (default mode), one job per GPU, never share a card.

## 2. The problem, precisely

The closure maps local information to a symmetric 3x3 tensor with

    trace f = 1,   f positive semi-definite,   |F| <= c E   (realizability).

M1 uses only (E, F) at the cell. That information is not enough to tell
"one beam with f = 0.7" from "two opposite beams with f = 0.7 each": any
local closure of (E, F) alone has this degeneracy, which is exactly why M1
merges beams. A neural closure therefore must see more than the cell:

- a spatial neighbourhood of (E, F), e.g. a 3^3 or 5^3 stencil, and/or
- the local optical depth kappa * dx and the source field S_E, and/or
- the previous tensor (memory), updated cheaply each step.

The hypothesis to test: a small neighbourhood of (E, F, kappa, S_E) contains
enough information to reconstruct the VET tensor to a few percent, while a
convolutional evaluation of that stencil costs about one M1 closure.

## 3. Suggested plan

1. **Reproduce.** Run both benchmarks, understand `physical_flux`, the HLLE
   flux, and how `vet.py` builds f_ij. Confirm the numbers in README.
2. **Build a dataset.** Random configurations: 1-8 point sources with random
   positions and luminosities, random uniform or lognormal kappa fields
   (see `../Ray-trax/ray_trax/create_turbulent_3D.py`), 48^3 or 64^3 grids.
   For each: run the VET solver to steady state, store (E, F, kappa, S_E)
   and the VET tensor f_ij as label. Also store intermediate (non steady)
   snapshots so the closure sees transients. Use the exact analytic tensor
   (r-hat r-hat superposition weighted by each source's E) as a cleaner label
   where it exists; the formal solution has ray-effect noise.
3. **Model.** A small 3D CNN (2-3 layers, 3^3 kernels, 16-32 channels)
   mapping the stencil to the 6 tensor components. Enforce the constraints
   by construction: output a symmetric matrix A, take f = A A^T / tr(A A^T)
   (PSD and trace 1 for free). Make it rotation/reflection equivariant, or
   at least train with the 48 cubic symmetries as augmentation; the physics
   is isotropic and the benchmark cases are symmetric.
   Normalize inputs: E in log, F/(cE) (bounded), kappa dx, S_E/(cE).
4. **Train offline first** (supervised on the labels), then **evaluate online**:
   plug the network into `divergence_flux_generic` and run the multi-source
   benchmark. Offline accuracy does not guarantee a stable solver.
5. **Stabilize online.** Expect instabilities. Tools: a-posteriori (solver in
   the loop) fine-tuning through JAX autodiff over a few unrolled steps,
   blending f = (1-w) f_M1 + w f_NN near the source cells, and the existing
   positivity/causality limiter.
6. **Cost target.** Time `step` with the NN closure against M1 at 64^3 on a
   GPU (`jax.jit`, batch the stencil evaluation as a conv). Report cost per
   step and accuracy in the same table as README. Goal: <= M1 cost,
   <= VET error on 2 and 5 sources, and no regression on 1 source.

## 4. Things that will bite

- JAX reads `JAX_PLATFORMS` at import: `select_device()` must be the first
  thing in every script, before any module that imports jax.
- kappa_t is the total opacity on the flux; set kappa_t = kappa_a (+ kappa_s).
- Use outflow boundaries when comparing to infinite-domain analytic solutions.
- The solver is first order; both closures lose a few percent to numerical
  diffusion regardless. Do not chase the last percent with the network.
  Implementing MUSCL reconstruction in `divergence_flux_generic` is a
  worthwhile side task and helps every closure.
- VET labels are noisy (ray effects). Smooth them (`VETConfig.smooth`) or use
  384 directions when generating data; check label quality on the single
  source case where the exact tensor is known.

## 5. Deliverables

- Dataset generator script and a trained closure (`momentRTjax/nn_closure.py`).
- One figure: error vs. cost for M1, VET, NN on the three benchmark cases.
- A short report: what information the closure needs (ablation over stencil
  size and inputs), and where it still fails.

## 6. Reading

- Levermore 1984, J. Quant. Spectrosc. Radiat. Transfer 31, 149 — M1 closure.
- Stone, Mihalas & Norman 1992, ApJS 80, 819 — VET method.
- Jiang, Stone & Davis 2012, ApJS 199, 14 — VET in a modern FV code, HLLE for moments.
- Huang, Tong, Ma 2022 (and related "neural closure for moment equations" work) —
  realizability-preserving learned closures for kinetic / RT moment systems.
- JAX docs: `jax.lax.scan`, `jax.grad` through a solver loop.
