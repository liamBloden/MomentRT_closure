"""GPU selection helper. Call `select_device()` BEFORE importing jax.

Lab policy: GPUs are picked with autocvd in its default mode, which blocks
until a card is completely free (never share a GPU with another process).
Set M1_DEVICE=cpu to run on the CPU instead (small tests, no GPU needed).

JAX reads JAX_PLATFORMS when it is imported, so this function refuses to run
if jax is already in sys.modules: a silent fallback would land the job on
GPU 0 regardless of who is using it.
"""
import os
import sys


def select_device(num_gpus: int = 1) -> None:
    if "jax" in sys.modules:
        raise RuntimeError("select_device() must be called before `import jax` "
                           "(import momentRTjax.devices first, then jax).")
    if os.environ.get("M1_DEVICE", "").lower() == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        print("[devices] M1_DEVICE=cpu -> running on CPU")
        return
    from autocvd import autocvd
    gpus = autocvd(num_gpus=num_gpus)   # default mode: waits for a fully free GPU
    print(f"[devices] autocvd selected GPU(s): {gpus}")
