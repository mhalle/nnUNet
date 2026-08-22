"""Benchmark resample_data_or_seg_to_shape_gpu against nnU-Net's default resampler and the torch one.

    python nnunetv2/tests/benchmark_resample_gpu.py [path/to/image.nii.gz] [target_mm]

Without arguments a synthetic 512 x 512 x 300 volume at 0.7 x 0.7 x 1.5 mm is used and resampled to
target_mm (default 1.5) isotropic. Reports wall time (best of 3, after a warm-up) for

  * resample_data_or_seg_to_shape       (scipy / skimage, CPU, the default)
  * resample_torch_fornnunet            (F.interpolate trilinear; GPU incl. MPS on torch >= 2.11)
  * resample_data_or_seg_to_shape_gpu   (this module, default = same results as the scipy path)
  * resample_data_or_seg_to_shape_gpu   with anti_alias=True

and the max absolute difference of each result to the scipy default. The last section shows
the anti-aliasing policy on white noise decimated 8x (how much aliased variance survives).
"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape, resample_data_or_seg_to_shape
from nnunetv2.preprocessing.resampling.resample_gpu import _best_device, resample_data_or_seg_to_shape_gpu
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet

DEV = _best_device()


def _sync():
    if DEV.type == "mps":
        torch.mps.synchronize()
    elif DEV.type == "cuda":
        torch.cuda.synchronize()


def time_call(fn, *a, reps=3, **kw):
    out = fn(*a, **kw)
    _sync()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        _sync()
        best = min(best, time.perf_counter() - t0)
    return best, out


def load_volume(path):
    import SimpleITK as sitk
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)[None]          # (1, z, y, x)
    spacing = tuple(reversed(img.GetSpacing()))                          # (z, y, x)
    return arr, spacing


def synthetic_volume():
    rng = np.random.default_rng(0)
    low = rng.standard_normal((1, 38, 64, 64)).astype(np.float32) * 300.0
    vol = F.interpolate(torch.as_tensor(low)[None], size=(300, 512, 512), mode="trilinear", align_corners=False)[0].numpy()
    vol += rng.standard_normal(vol.shape).astype(np.float32) * 20.0       # noise: realistic high-frequency content
    return vol, (1.5, 0.7, 0.7)


def report(name, dt, out, ref=None):
    out = out.cpu().numpy() if isinstance(out, torch.Tensor) else out
    line = f"  {name:<44} {dt * 1e3:9.1f} ms"
    if ref is not None:
        line += f"   max|d| vs scipy = {float(np.abs(out.astype(np.float64) - ref).max()):9.4f}"
    print(line)


def main():
    if len(sys.argv) > 1:
        data, spacing = load_volume(sys.argv[1])
        src = sys.argv[1]
    else:
        data, spacing = synthetic_volume()
        src = "synthetic"
    target_mm = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    new_spacing = (target_mm,) * 3
    new_shape = tuple(int(s) for s in compute_new_shape(data.shape[1:], spacing, new_spacing))
    print(f"device={DEV}  input={src} {data.shape[1:]} @ {tuple(round(s, 3) for s in spacing)} mm -> {new_shape} @ {target_mm} mm")

    dt_s, ref = time_call(resample_data_or_seg_to_shape, data, new_shape, spacing, new_spacing, is_seg=False, order=3, order_z=0, force_separate_z=None)
    ref = ref.astype(np.float64)
    report("scipy default (CPU)", dt_s, ref)
    dt_t, out_t = time_call(resample_torch_fornnunet, data, new_shape, spacing, new_spacing, is_seg=False)
    report("resample_torch_fornnunet", dt_t, out_t, ref)
    dt_g, out_g = time_call(resample_data_or_seg_to_shape_gpu, data, new_shape, spacing, new_spacing, is_seg=False, order=3, order_z=0, force_separate_z=None, device=DEV)
    report(f"resample_data_or_seg_to_shape_gpu ({DEV.type})", dt_g, out_g, ref)
    dt_a, out_a = time_call(resample_data_or_seg_to_shape_gpu, data, new_shape, spacing, new_spacing, is_seg=False, device=DEV, anti_alias=True)
    report(f"  ... anti_alias=True ({DEV.type})", dt_a, out_a, ref)
    print(f"  speed-up vs scipy: torch {dt_s / dt_t:5.1f}x   gpu {dt_s / dt_g:5.1f}x")

    print("\nanti-aliasing: white noise (std 100) decimated 8x in y and z; lower output std = less aliased variance")
    d = (np.random.default_rng(0).standard_normal((1, 64, 64, 64)) * 100.0).astype(np.float32)
    out_plain = resample_data_or_seg_to_shape_gpu(d, (64, 8, 8), device=DEV)
    out_aa = resample_data_or_seg_to_shape_gpu(d, (64, 8, 8), device=DEV, anti_alias=True)
    out_tri = F.interpolate(torch.as_tensor(d)[None], (64, 8, 8), mode="trilinear", antialias=False)[0].numpy()
    print(f"  trilinear (F.interpolate)   : std = {float(out_tri.std()):6.1f}")
    print(f"  default (skimage semantics) : std = {float(out_plain.std()):6.1f}")
    print(f"  anti_alias=True             : std = {float(out_aa.std()):6.1f}")


if __name__ == "__main__":
    main()
