"""Self-checking correctness tests for resample_aa_torch.

Run: uv run python scripts/test_resample_gpu_aa.py
(No pytest dependency - plain asserts, exits nonzero on failure.)
"""
import numpy as np
import torch
import torch.nn.functional as F

from nnunetv2.preprocessing.resampling.resample_gpu_aa import resample_aa_torch, scipy_zoom_torch, _best_device
from nnunetv2.preprocessing.resampling.utils import recursive_find_resampling_fn_by_name

DEV = _best_device()
PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def test_noop():
    d = np.random.randn(1, 16, 16, 16).astype(np.float32)
    o = resample_aa_torch(d, (16, 16, 16), device=DEV)
    check("identity shape is a no-op (returns input)", o is d)


def test_upsample_matches_trilinear():
    rng = np.random.default_rng(0)
    d = (rng.standard_normal((1, 30, 40, 50)) * 100).astype(np.float32)
    new = (60, 80, 100)
    ours = resample_aa_torch(d, new, device=DEV)
    ref = F.interpolate(torch.as_tensor(d)[None], new, mode="trilinear",
                        align_corners=False)[0].numpy()
    md = np.abs(ours - ref).max()
    check("upsample linear == torch trilinear", md < 1e-3, f"max|delta|={md:.2e}")


def test_downsample_antialiases():
    # High-frequency zero-mean noise along axis 0, downsampled 8x. A proper AA
    # prefilter averages ~8 source voxels -> output variance collapses toward 0.
    # A 2-tap kernel (trilinear) averages only 2 -> retains far more variance.
    # Random content avoids the phase-alignment luck that can make a periodic
    # stripe look band-limited under point interpolation.
    rng = np.random.default_rng(1)
    N = 512
    noise = rng.standard_normal(N).astype(np.float32) * 100.0
    d = np.broadcast_to(noise[:, None, None], (N, 8, 8)).copy()[None]
    o = resample_aa_torch(d, (64, 8, 8), device=DEV)                       # 8x down -> cubic AA
    ot = F.interpolate(torch.as_tensor(d)[None].float(), (64, 8, 8),
                       mode="trilinear", align_corners=False)[0].numpy()   # 2-tap, no AA
    check("AA prefilter collapses high-freq variance", o.std() < 40,
          f"AA std={o.std():.1f} (input std=100)")
    check("AA suppresses aliasing far better than trilinear", o.std() < 0.6 * ot.std(),
          f"AA std={o.std():.1f}  vs trilinear std={ot.std():.1f}")


def test_cubic_no_overshoot():
    # sharp step edge; cubic rings past range unless clamped.
    d = np.zeros((1, 8, 200, 8), np.float32)
    d[:, :, 100:, :] = 1000.0
    o = resample_aa_torch(d, (8, 80, 8), device=DEV)  # downsample axis 1 -> cubic
    check("cubic output clamped to input range (no ring)",
          o.min() >= -1e-3 and o.max() <= 1000 + 1e-3,
          f"range[{o.min():.2f},{o.max():.2f}]")


def test_seg_label_preserving():
    s = np.zeros((1, 64, 64, 64), np.int16)
    s[0, 10:50, 10:50, 10:50] = 7
    s[0, 20:30, 20:30, 20:30] = 3
    o = resample_aa_torch(s, (32, 32, 32), is_seg=True, device=DEV)
    labs = set(np.unique(o).tolist())
    check("seg resample emits only input labels", labs.issubset({0, 3, 7}), f"labels={sorted(labs)}")
    check("seg output shape correct", o.shape == (1, 32, 32, 32))
    check("seg dtype integer", o.dtype == np.int16)


def test_container_contract():
    d = np.random.randn(1, 20, 20, 20).astype(np.float32)
    on = resample_aa_torch(d, (10, 10, 10), device=DEV)
    check("numpy in -> numpy out, dtype preserved", isinstance(on, np.ndarray) and on.dtype == np.float32)
    t = torch.randn(1, 20, 20, 20)
    ot = resample_aa_torch(t, (10, 10, 10), device=DEV)
    check("tensor in -> tensor back on original device", isinstance(ot, torch.Tensor) and ot.device == t.device)


def test_channel_chunk_equiv():
    d = np.random.randn(40, 24, 24, 24).astype(np.float32)
    a = resample_aa_torch(d, (32, 32, 32), device=DEV, channel_chunk=8)
    b = resample_aa_torch(d, (32, 32, 32), device=DEV, channel_chunk=64)
    md = np.abs(a - b).max()
    check("channel-chunked == single-shot", md < 1e-4, f"max|delta|={md:.2e}")


def test_discoverable():
    fn = recursive_find_resampling_fn_by_name("resample_aa_torch")
    check("discoverable by plans name", fn.__name__ == "resample_aa_torch")



def _rand_vol(shape, seed=0, dtype=np.float64, scale=1000.0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(shape) * scale
    v[:, shape[1] // 3:2 * shape[1] // 3] += 800.0      # structure: exercises prefilter + boundaries
    return v.astype(dtype)


def test_scipy_parity_float64_cpu():
    from scipy import ndimage
    vol = _rand_vol((1, 40, 52, 36))
    for new_shape in [(17, 23, 61), (80, 26, 36), (40, 52, 36)]:
        for order in (0, 1, 3):
            for mode in ("nearest", "mirror"):
                ref = ndimage.zoom(vol[0], np.array(new_shape) / np.array(vol.shape[1:]), order=order, mode=mode)
                out = resample_aa_torch(vol, new_shape, device="cpu", convention="scipy", order=order, mode=mode)[0]
                err = float(np.abs(out - ref).max())
                check(f"scipy parity f64 cpu {new_shape} order={order} mode={mode}", err < 1e-6, f"max|d|={err:.2e}")


def test_scipy_parity_float32_gpu():
    from scipy import ndimage
    if DEV.type == "cpu":
        print("  (no GPU; skipping float32 GPU parity)"); return
    vol = _rand_vol((1, 48, 60, 44))
    for new_shape in [(21, 27, 70), (48, 60, 44)]:
        for order in (1, 3):
            ref = ndimage.zoom(vol[0], np.array(new_shape) / np.array(vol.shape[1:]), order=order, mode="nearest")
            out = resample_aa_torch(vol.astype(np.float32), new_shape, device=DEV, convention="scipy", order=order)[0]
            err = float(np.abs(out.astype(np.float64) - ref).max())
            check(f"scipy parity f32 {DEV.type} {new_shape} order={order}", err < 2e-2, f"max|d|={err:.2e} (HU-scale values)")


def test_scipy_parity_int16_rounding():
    from scipy import ndimage
    vol = _rand_vol((1, 30, 34, 28)).astype(np.int16)
    ref = ndimage.zoom(vol[0], np.array((13, 61, 28)) / np.array(vol.shape[1:]), order=3, mode="nearest")
    out = resample_aa_torch(vol, (13, 61, 28), device="cpu", convention="scipy", order=3)[0]
    same = float((out == ref).mean())
    check("scipy parity int16 (scipy rounding)", out.dtype == np.int16 and same > 0.999, f"dtype={out.dtype} identical={same:.5f}")


def test_scipy_seg_nearest_exact():
    from scipy import ndimage
    rng = np.random.default_rng(1)
    seg = rng.integers(0, 120, size=(1, 31, 29, 35)).astype(np.int16)
    for new_shape in [(14, 67, 35), (62, 58, 70)]:
        ref = ndimage.zoom(seg[0], np.array(new_shape) / np.array(seg.shape[1:]), order=0, mode="nearest")
        out = resample_aa_torch(seg, new_shape, is_seg=True, device="cpu", convention="scipy", order=0)[0]
        check(f"scipy seg NN exact {new_shape} cpu", np.array_equal(out.astype(np.int64), ref.astype(np.int64)), "label mismatch")
        if DEV.type != "cpu":
            outg = resample_aa_torch(seg, new_shape, is_seg=True, device=DEV, convention="scipy", order=0)[0]
            check(f"scipy seg NN exact {new_shape} {DEV.type}", np.array_equal(outg.astype(np.int64), ref.astype(np.int64)), "label mismatch")


def test_scipy_zoom_torch_wrapper():
    from scipy import ndimage
    vol = _rand_vol((1, 20, 24, 22))
    ref = ndimage.zoom(vol[0], np.array((9, 11, 44)) / np.array(vol.shape[1:]), order=3, mode="nearest")
    out = scipy_zoom_torch(vol, (9, 11, 44), order=3, mode="nearest", device="cpu")[0]
    check("scipy_zoom_torch wrapper", float(np.abs(out - ref).max()) < 1e-6, "")


def test_grid_default_unchanged():
    vol = _rand_vol((1, 40, 52, 36)).astype(np.float32)
    a = resample_aa_torch(vol, (17, 23, 61), device="cpu")[0]
    b = resample_aa_torch(vol, (17, 23, 61), device="cpu", convention="grid")[0]
    check("grid default unchanged", np.array_equal(a, b), "")


if __name__ == "__main__":
    print(f"device={DEV}")
    for t in [test_noop, test_upsample_matches_trilinear, test_downsample_antialiases,
              test_cubic_no_overshoot, test_seg_label_preserving, test_container_contract,
              test_channel_chunk_equiv, test_discoverable,
              test_scipy_parity_float64_cpu, test_scipy_parity_float32_gpu, test_scipy_parity_int16_rounding,
              test_scipy_seg_nearest_exact, test_scipy_zoom_torch_wrapper, test_grid_default_unchanged]:
        t()
    n_ok, n = sum(PASS), len(PASS)
    print(f"\n{n_ok}/{n} checks passed")
    raise SystemExit(0 if n_ok == n else 1)
