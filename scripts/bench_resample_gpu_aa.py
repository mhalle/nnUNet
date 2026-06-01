"""Benchmark + correctness: resample_aa_torch vs upstream resamplers.

Compares the new GPU anti-aliased resampler against:
  * scipy default  (resample_data_or_seg_to_shape, order=3, anti_aliasing=False)
  * torch          (resample_torch_fornnunet, F.interpolate trilinear)

on (a) wall-time, (b) numeric agreement on an upsample (linear) case, and
(c) aliasing on a synthetic high-frequency pattern when downsampling.

Usage:
  uv run python scripts/bench_resample_gpu_aa.py [path_to_nii] [target_mm]
"""
import sys
import time

import numpy as np
import torch

from nnunetv2.preprocessing.resampling.default_resampling import (
    resample_data_or_seg_to_shape, compute_new_shape)
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet
from nnunetv2.preprocessing.resampling.resample_gpu_aa import resample_aa_torch, _best_device

DEV = _best_device()


def _sync():
    if DEV.type == "mps":
        torch.mps.synchronize()
    elif DEV.type == "cuda":
        torch.cuda.synchronize()


def time_call(fn, *a, reps=3, **kw):
    fn(*a, **kw); _sync()                       # warmup
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(*a, **kw); _sync()
        best = min(best, time.perf_counter() - t0)
    return best, out


def load_volume(path):
    import SimpleITK as sitk
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)   # (z, y, x)
    spacing_zyx = tuple(reversed(img.GetSpacing()))
    return arr[None], np.array(spacing_zyx)                # (1, z, y, x)


def report(name, dt, out, ref=None):
    o = out[0] if hasattr(out, "ndim") else out
    if isinstance(o, torch.Tensor):
        o = o.detach().cpu().numpy()
    line = f"  {name:22s} {dt*1000:8.1f} ms   range[{o.min():8.1f},{o.max():8.1f}]"
    if ref is not None:
        r = ref[0]
        if isinstance(r, torch.Tensor):
            r = r.detach().cpu().numpy()
        line += f"   vs-ref max|Δ|={np.abs(o - r).max():.3g}  rmse={np.sqrt(((o-r)**2).mean()):.3g}"
    print(line)
    return o


def bench_real(path, target_mm):
    print(f"\n=== REAL VOLUME: {path} ===")
    data, spacing = load_volume(path)
    new_spacing = np.array([float(target_mm)] * 3)
    new_shape = compute_new_shape(data.shape[1:], spacing, new_spacing)
    f = np.array(data.shape[1:]) / new_shape
    print(f"  shape {tuple(data.shape[1:])} @ {spacing.round(3)} mm "
          f"-> {tuple(int(s) for s in new_shape)} @ {new_spacing} mm   per-axis factor {f.round(2)}")
    print(f"  device={DEV}")

    dt_s, out_s = time_call(resample_data_or_seg_to_shape, data, new_shape, spacing, new_spacing,
                            is_seg=False, order=3)
    ref = report("scipy order-3 (CPU)", dt_s, out_s)

    # torch trilinear3d is NOT implemented on MPS (only CPU); run it on CPU.
    dt_t, out_t = time_call(resample_torch_fornnunet, data, new_shape, spacing, new_spacing,
                            is_seg=False, device=torch.device("cpu"))
    report("torch trilinear (CPU*)", dt_t, out_t, ref=(ref,))

    dt_a, out_a = time_call(resample_aa_torch, data, new_shape, spacing, new_spacing,
                            is_seg=False, device=DEV)
    report(f"AA cubic/linear ({DEV.type.upper()})", dt_a, out_a, ref=(ref,))

    print(f"  * torch trilinear3d unsupported on MPS — CPU is its only device for 3D")
    print(f"  speedup vs scipy:  torch-cpu {dt_s/dt_t:5.1f}x   AA-{DEV.type} {dt_s/dt_a:5.1f}x")


def aliasing_test():
    """High-frequency checkerboard along x; downsample 4x. A non-anti-aliased
    resampler aliases the pattern into spurious low-frequency structure (high
    output variance); a proper AA prefilter averages it toward the mean."""
    print("\n=== ALIASING TEST: 4x downsample of a 1-voxel checkerboard ===")
    N = 256
    z = np.arange(N)
    patt = ((z[:, None, None] // 1) % 2).astype(np.float32) * 1000.0   # stripes along x-axis-0
    data = np.broadcast_to(patt, (N, 32, 32)).copy()[None]            # (1, 256, 32, 32)
    new_shape = (64, 32, 32)                                          # 4x down on axis 0

    _, out_t = time_call(resample_torch_fornnunet, data, new_shape, [1,1,1], [4,1,1],
                         is_seg=False, device=torch.device("cpu"), reps=1)
    _, out_a = time_call(resample_aa_torch, data, new_shape, [1,1,1], [4,1,1],
                         is_seg=False, device=DEV, reps=1)
    ot = out_t[0].cpu().numpy() if isinstance(out_t, torch.Tensor) else out_t[0]
    oa = out_a[0].cpu().numpy() if isinstance(out_a, torch.Tensor) else out_a[0]
    print(f"  ideal (band-limited) output = flat 500.  Lower std = better AA.")
    print(f"  torch trilinear : mean={ot.mean():7.1f}  std={ot.std():7.1f}")
    print(f"  AA cubic        : mean={oa.mean():7.1f}  std={oa.std():7.1f}")


def upsample_agreement():
    """On a pure 2x upsample, ours (linear) should closely match torch trilinear."""
    print("\n=== UPSAMPLE AGREEMENT (2x up, linear regime) ===")
    rng = np.random.default_rng(0)
    data = rng.standard_normal((1, 40, 50, 60)).astype(np.float32) * 100
    new_shape = (80, 100, 120)
    _, out_t = time_call(resample_torch_fornnunet, data, new_shape, [2,2,2], [1,1,1],
                         is_seg=False, device=torch.device("cpu"), reps=1)
    _, out_a = time_call(resample_aa_torch, data, new_shape, [2,2,2], [1,1,1],
                         is_seg=False, device=DEV, reps=1)
    ot = out_t[0].cpu().numpy() if isinstance(out_t, torch.Tensor) else out_t[0]
    oa = out_a[0].cpu().numpy() if isinstance(out_a, torch.Tensor) else out_a[0]
    print(f"  max|Δ| vs torch trilinear = {np.abs(ot-oa).max():.4g}   rmse = {np.sqrt(((ot-oa)**2).mean()):.4g}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    target_mm = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    upsample_agreement()
    aliasing_test()
    if path:
        bench_real(path, target_mm)
