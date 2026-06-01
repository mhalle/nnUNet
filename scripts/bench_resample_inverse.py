"""Device-resident forward + K-channel inverse (probabilities) benchmark.

Times the compute-only path (tensor already on device, as in real inference)
and the inverse resample where scipy loops per channel in Python. The inverse
uses channel-chunking so large K (e.g. 117 TS classes) stays within memory.
"""
import time
import numpy as np
import torch

from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
from nnunetv2.preprocessing.resampling.resample_gpu_aa import (
    resample_aa_torch, _separable_resample, _best_device)

DEV = _best_device()


def _sync(d=DEV):
    if d.type == "mps": torch.mps.synchronize()
    elif d.type == "cuda": torch.cuda.synchronize()


def t_dev(fn, *a, reps=3, **kw):
    fn(*a, **kw); _sync()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter(); fn(*a, **kw); _sync()
        best = min(best, time.perf_counter() - t0)
    return best


def forward_device_resident():
    print(f"\n=== FORWARD, device-resident tensor (no transfer), device={DEV} ===")
    for shp, new in [((1, 512, 512, 165), (227, 227, 220)),
                     ((1, 768, 768, 709), (333, 333, 473))]:
        x = torch.randn(*shp, device=DEV)
        dt = t_dev(lambda z: _separable_resample(z, new, 1.1), x)
        print(f"  {shp[1:]} -> {new}   AA-{DEV.type} compute  {dt*1000:8.1f} ms")


def inverse_probabilities():
    print(f"\n=== INVERSE (probabilities), channel_chunk=8, device={DEV} ===")
    for K, mg, ag in [(20, (96, 96, 96), (160, 160, 160)),
                      (50, (80, 80, 96), (128, 128, 160)),
                      (117, (112, 112, 128), (192, 192, 224))]:
        logits = np.random.randn(K, *mg).astype(np.float32)
        t0 = time.perf_counter()
        resample_data_or_seg_to_shape(logits, ag, [1.5]*3, [1.0]*3, is_seg=False, order=1)
        dt_s = time.perf_counter() - t0
        lt = torch.as_tensor(logits).to(DEV)
        dt_a = t_dev(lambda z: resample_aa_torch(z, ag, device=DEV, channel_chunk=8), lt, reps=2)
        outsz = K * float(np.prod(ag)) * 4 / 1e6
        print(f"  K={K:3d}  {mg} -> {ag}  (out {outsz:.0f} MB)")
        print(f"      scipy order-1 per-chan (CPU)  {dt_s*1000:8.1f} ms")
        print(f"      AA resample ({DEV.type})           {dt_a*1000:8.1f} ms   "
              f"speedup {dt_s/dt_a:5.1f}x")


if __name__ == "__main__":
    forward_device_resident()
    inverse_probabilities()
