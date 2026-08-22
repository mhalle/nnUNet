"""Self-checking correctness tests for resample_data_or_seg_to_shape_gpu.

Run: uv run python scripts/test_resample_gpu.py
(No pytest dependency - plain asserts, exits nonzero on failure.)
"""
import numpy as np
import torch
import torch.nn.functional as F

from nnunetv2.preprocessing.resampling.resample_gpu import resample_data_or_seg_to_shape_gpu, scipy_zoom_torch, skimage_resize_torch, _best_device
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
from nnunetv2.preprocessing.resampling.utils import recursive_find_resampling_fn_by_name

DEV = _best_device()
PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def test_noop():
    d = np.random.randn(1, 16, 16, 16).astype(np.float32)
    o = resample_data_or_seg_to_shape_gpu(d, (16, 16, 16), device=DEV)
    check("identity shape is a no-op (returns input)", o is d)


def test_upsample_matches_trilinear():
    rng = np.random.default_rng(0)
    d = (rng.standard_normal((1, 30, 40, 50)) * 100).astype(np.float32)
    new = (60, 80, 100)
    ours = resample_data_or_seg_to_shape_gpu(d, new, device=DEV, anti_alias=True)
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
    o = resample_data_or_seg_to_shape_gpu(d, (64, 8, 8), device=DEV, anti_alias=True)                       # 8x down -> cubic AA
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
    o = resample_data_or_seg_to_shape_gpu(d, (8, 80, 8), device=DEV, anti_alias=True)  # downsample axis 1 -> cubic
    check("cubic output clamped to input range (no ring)",
          o.min() >= -1e-3 and o.max() <= 1000 + 1e-3,
          f"range[{o.min():.2f},{o.max():.2f}]")


def test_seg_label_preserving():
    s = np.zeros((1, 64, 64, 64), np.int16)
    s[0, 10:50, 10:50, 10:50] = 7
    s[0, 20:30, 20:30, 20:30] = 3
    o = resample_data_or_seg_to_shape_gpu(s, (32, 32, 32), is_seg=True, device=DEV, anti_alias=True)
    labs = set(np.unique(o).tolist())
    check("seg resample emits only input labels", labs.issubset({0, 3, 7}), f"labels={sorted(labs)}")
    check("seg output shape correct", o.shape == (1, 32, 32, 32))
    check("seg dtype integer", o.dtype == np.int16)


def test_container_contract():
    d = np.random.randn(1, 20, 20, 20).astype(np.float32)
    on = resample_data_or_seg_to_shape_gpu(d, (10, 10, 10), device=DEV, anti_alias=True)
    check("numpy in -> numpy out, dtype preserved", isinstance(on, np.ndarray) and on.dtype == np.float32)
    t = torch.randn(1, 20, 20, 20)
    ot = resample_data_or_seg_to_shape_gpu(t, (10, 10, 10), device=DEV, anti_alias=True)
    check("tensor in -> tensor back on original device", isinstance(ot, torch.Tensor) and ot.device == t.device)


def test_channel_chunk_equiv():
    d = np.random.randn(40, 24, 24, 24).astype(np.float32)
    a = resample_data_or_seg_to_shape_gpu(d, (32, 32, 32), device=DEV, channel_chunk=8, anti_alias=True)
    b = resample_data_or_seg_to_shape_gpu(d, (32, 32, 32), device=DEV, channel_chunk=64, anti_alias=True)
    md = np.abs(a - b).max()
    check("channel-chunked == single-shot", md < 1e-4, f"max|delta|={md:.2e}")


def test_discoverable():
    fn = recursive_find_resampling_fn_by_name("resample_data_or_seg_to_shape_gpu")
    check("discoverable by plans name", fn.__name__ == "resample_data_or_seg_to_shape_gpu")



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
                out = resample_data_or_seg_to_shape_gpu(vol, new_shape, device="cpu", convention="corner", order=order, mode=mode)[0]
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
            out = resample_data_or_seg_to_shape_gpu(vol.astype(np.float32), new_shape, device=DEV, convention="corner", order=order)[0]
            err = float(np.abs(out.astype(np.float64) - ref).max())
            check(f"scipy parity f32 {DEV.type} {new_shape} order={order}", err < 2e-2, f"max|d|={err:.2e} (HU-scale values)")


def test_scipy_parity_int16_rounding():
    from scipy import ndimage
    vol = _rand_vol((1, 30, 34, 28)).astype(np.int16)
    ref = ndimage.zoom(vol[0], np.array((13, 61, 28)) / np.array(vol.shape[1:]), order=3, mode="nearest")
    out = resample_data_or_seg_to_shape_gpu(vol, (13, 61, 28), device="cpu", convention="corner", order=3)[0]
    same = float((out == ref).mean())
    check("scipy parity int16 (scipy rounding)", out.dtype == np.int16 and same > 0.999, f"dtype={out.dtype} identical={same:.5f}")


def test_scipy_seg_nearest_exact():
    from scipy import ndimage
    rng = np.random.default_rng(1)
    seg = rng.integers(0, 120, size=(1, 31, 29, 35)).astype(np.int16)
    for new_shape in [(14, 67, 35), (62, 58, 70)]:
        ref = ndimage.zoom(seg[0], np.array(new_shape) / np.array(seg.shape[1:]), order=0, mode="nearest")
        out = resample_data_or_seg_to_shape_gpu(seg, new_shape, is_seg=True, device="cpu", convention="corner", order=0)[0]
        check(f"scipy seg NN exact {new_shape} cpu", np.array_equal(out.astype(np.int64), ref.astype(np.int64)), "label mismatch")
        if DEV.type != "cpu":
            outg = resample_data_or_seg_to_shape_gpu(seg, new_shape, is_seg=True, device=DEV, convention="corner", order=0)[0]
            check(f"scipy seg NN exact {new_shape} {DEV.type}", np.array_equal(outg.astype(np.int64), ref.astype(np.int64)), "label mismatch")


def test_scipy_zoom_torch_wrapper():
    from scipy import ndimage
    vol = _rand_vol((1, 20, 24, 22))
    ref = ndimage.zoom(vol[0], np.array((9, 11, 44)) / np.array(vol.shape[1:]), order=3, mode="nearest")
    out = scipy_zoom_torch(vol, (9, 11, 44), order=3, mode="nearest", device="cpu")[0]
    check("scipy_zoom_torch wrapper", float(np.abs(out - ref).max()) < 1e-6, "")


def test_center_parity_nnunet():
    """Defaults == nnU-Net's resample_data_or_seg_to_shape == skimage.resize(mode='edge', anti_aliasing=False)."""
    vol = _rand_vol((2, 40, 52, 36))
    sp = (1.0, 1.0, 1.0)
    for new_shape in [(17, 23, 61), (80, 26, 36), (40, 52, 36)]:
        for order in (0, 1, 3):
            ref = resample_data_or_seg_to_shape(vol, new_shape, sp, sp, is_seg=False, order=order, order_z=0, force_separate_z=False)
            out = resample_data_or_seg_to_shape_gpu(vol, new_shape, device="cpu", order=order)          # all defaults
            err = float(np.abs(out - ref).max())
            check(f"nnU-Net parity f64 cpu {new_shape} order={order}", err < 1e-6, f"max|d|={err:.2e}")
        if DEV.type != "cpu":
            ref = resample_data_or_seg_to_shape(vol, new_shape, sp, sp, is_seg=False, order=3, order_z=0, force_separate_z=False)
            out = resample_data_or_seg_to_shape_gpu(vol.astype(np.float32), new_shape, device=DEV, order=3)
            err = float(np.abs(out.astype(np.float64) - ref).max())
            check(f"nnU-Net parity f32 {DEV.type} {new_shape} order=3", err < 2e-2, f"max|d|={err:.2e}")
    from skimage.transform import resize
    ref = resize(vol[0], (17, 23, 61), order=3, mode="edge", anti_aliasing=False, preserve_range=True)
    out = skimage_resize_torch(vol, (17, 23, 61), order=3, device="cpu")[0]
    check("skimage_resize_torch wrapper", float(np.abs(out - ref).max()) < 1e-6, "")


def test_center_seg_parity_nnunet():
    """is_seg=True defaults == nnU-Net's label rule (threshold 0.5, paint ascending)."""
    rng = np.random.default_rng(2)
    seg = np.zeros((1, 31, 29, 35), dtype=np.int16)
    for L in range(1, 9):                      # overlapping boxes painted in order -> blobby labels
        z, y, x = rng.integers(0, 20, 3); seg[0, z:z + 12, y:y + 11, x:x + 13] = L
    sp = (1.0, 1.0, 1.0)
    for new_shape in [(14, 67, 35), (62, 58, 70)]:
        for order in (0, 1):
            ref = resample_data_or_seg_to_shape(seg, new_shape, sp, sp, is_seg=True, order=order, order_z=0, force_separate_z=False)
            out = resample_data_or_seg_to_shape_gpu(seg, new_shape, is_seg=True, device="cpu", order=order)
            same = float((out.astype(np.int64) == ref.astype(np.int64)).mean())
            check(f"nnU-Net seg parity {new_shape} order={order}", same == 1.0, f"identical={same:.6f}")
            if DEV.type != "cpu":
                outg = resample_data_or_seg_to_shape_gpu(seg, new_shape, is_seg=True, device=DEV, order=order)
                same = float((outg.astype(np.int64) == ref.astype(np.int64)).mean())
                check(f"nnU-Net seg parity {new_shape} order={order} {DEV.type}", same > 0.9999, f"identical={same:.6f}")


def test_aa_policy_is_opt_in():
    vol = _rand_vol((1, 40, 52, 36)).astype(np.float32)
    a = resample_data_or_seg_to_shape_gpu(vol, (17, 23, 61), device="cpu")[0]
    b = resample_data_or_seg_to_shape_gpu(vol, (17, 23, 61), device="cpu", anti_alias=True)[0]
    check("anti_alias=True differs from the exact default on downsampling", not np.array_equal(a, b), "")


def test_separate_z_parity_nnunet_data():
    """Anisotropic spacing -> upstream's separate-z path; ours must match it exactly."""
    vol = _rand_vol((2, 40, 44, 12))
    cases = [  # (current_spacing, new_spacing, force) -> (in-plane down, z up 5x), axis 0 variant, no-sep tie, forced
        ((0.7, 0.7, 5.0), (1.0, 1.0, 1.0), None),
        ((5.0, 0.7, 0.7), (1.5, 1.5, 1.5), None),
        ((0.24, 1.25, 1.25), (1.0, 1.0, 1.0), None),     # two low-res axes -> upstream does NOT separate
        ((0.7, 0.7, 5.0), (1.0, 1.0, 1.0), True),
        ((0.7, 0.7, 5.0), (1.0, 1.0, 1.0), False),
    ]
    for cur, new, force in cases:
        new_shape = tuple(int(round(s * c / n)) for s, c, n in zip(vol.shape[1:], cur, new))
        for order, order_z in ((3, 0), (1, 0), (3, 1), (3, 3)):
            ref = resample_data_or_seg_to_shape(vol, new_shape, cur, new, is_seg=False, order=order, order_z=order_z, force_separate_z=force)
            out = resample_data_or_seg_to_shape_gpu(vol, new_shape, cur, new, device="cpu", order=order, order_z=order_z, force_separate_z=force)
            err = float(np.abs(out - ref).max())
            check(f"separate-z data {cur}->{new} force={force} order={order} order_z={order_z}", err < 1e-6, f"max|d|={err:.2e}")
    if DEV.type != "cpu":
        cur, new = (0.7, 0.7, 5.0), (1.0, 1.0, 1.0)
        new_shape = tuple(int(round(s * c / n)) for s, c, n in zip(vol.shape[1:], cur, new))
        ref = resample_data_or_seg_to_shape(vol, new_shape, cur, new, is_seg=False, order=3, order_z=0, force_separate_z=None)
        out = resample_data_or_seg_to_shape_gpu(vol.astype(np.float32), new_shape, cur, new, device=DEV, order=3, order_z=0, force_separate_z=None)
        err = float(np.abs(out.astype(np.float64) - ref).max())
        check(f"separate-z data f32 {DEV.type}", err < 2e-2, f"max|d|={err:.2e}")


def test_separate_z_parity_nnunet_seg():
    rng = np.random.default_rng(3)
    seg = np.zeros((1, 40, 44, 12), dtype=np.int16)
    for L in range(1, 9):
        z, y, x = rng.integers(0, 24, 2).tolist() + [int(rng.integers(0, 6))]
        seg[0, z:z + 14, y:y + 13, x:x + 5] = L
    cur, new = (0.7, 0.7, 5.0), (1.0, 1.0, 1.0)
    new_shape = tuple(int(round(s * c / n)) for s, c, n in zip(seg.shape[1:], cur, new))
    for order, order_z, force in ((1, 0, None), (0, 0, None), (1, 1, True), (3, 3, True), (1, 0, False)):
        ref = resample_data_or_seg_to_shape(seg, new_shape, cur, new, is_seg=True, order=order, order_z=order_z, force_separate_z=force)
        out = resample_data_or_seg_to_shape_gpu(seg, new_shape, cur, new, is_seg=True, device="cpu", order=order, order_z=order_z, force_separate_z=force)
        same = float((out.astype(np.int64) == ref.astype(np.int64)).mean())
        check(f"separate-z seg order={order} order_z={order_z} force={force}", same == 1.0, f"identical={same:.6f}")
        if DEV.type != "cpu":
            outg = resample_data_or_seg_to_shape_gpu(seg, new_shape, cur, new, is_seg=True, device=DEV, order=order, order_z=order_z, force_separate_z=force)
            same = float((outg.astype(np.int64) == ref.astype(np.int64)).mean())
            check(f"separate-z seg order={order} order_z={order_z} force={force} {DEV.type}", same > 0.9999, f"identical={same:.6f}")


def test_seg_threshold_exact_half():
    """2x downsampling with order 1 produces indicator values of exactly 0.5: the >= rule must match."""
    rng = np.random.default_rng(4)
    seg = rng.integers(0, 5, size=(1, 32, 30, 28)).astype(np.int16)
    sp = (1.0, 1.0, 1.0)
    ref = resample_data_or_seg_to_shape(seg, (16, 15, 14), sp, sp, is_seg=True, order=1, order_z=0, force_separate_z=False)
    out = resample_data_or_seg_to_shape_gpu(seg, (16, 15, 14), sp, sp, is_seg=True, device="cpu", order=1, force_separate_z=False)
    same = float((out.astype(np.int64) == ref.astype(np.int64)).mean())
    check("seg >= 0.5 rule at exact halves (2x down, order 1)", same == 1.0, f"identical={same:.6f}")


def _smooth_field(shape, seed=5):
    """Low-frequency random field with a constant offset so cubic ringing stays inside the value range."""
    rng = np.random.default_rng(seed)
    low = tuple(max(2, s // 6) for s in shape)
    from scipy.ndimage import zoom as _zoom
    f = _zoom(rng.standard_normal(low), tuple(s / l for s, l in zip(shape, low)), order=3)[:shape[0], :shape[1]]
    return (f * 200.0 + 1000.0).astype(np.float64)


def test_aa_matches_torch_antialias_downsample():
    """anti_alias=True == F.interpolate(mode='bicubic', antialias=True): same PIL-style filter
    (Catmull-Rom scaled by the factor, half-pixel centers, taps renormalized at the borders)."""
    img = _smooth_field((96, 80))
    vol = img[None, None]                                  # (c=1, x=1, y=96, z=80): x is identity
    ours = resample_data_or_seg_to_shape_gpu(vol, (1, 42, 35), device="cpu", anti_alias=True)[0, 0]
    ref = F.interpolate(torch.as_tensor(img)[None, None], size=(42, 35), mode="bicubic", antialias=True, align_corners=False)[0, 0].numpy()
    interior = (ours > img.min() + 1e-9) & (ours < img.max() - 1e-9)   # where our range clip cannot have acted
    err = float(np.abs(ours - ref)[interior].max())
    check("AA downsample == torch bicubic antialias", err < 1e-6 and interior.mean() > 0.9, f"max|d|={err:.2e} on {interior.mean()*100:.0f}% of voxels")
    if DEV.type != "cpu":
        oursg = resample_data_or_seg_to_shape_gpu(vol.astype(np.float32), (1, 42, 35), device=DEV, anti_alias=True)[0, 0]
        err = float(np.abs(oursg.astype(np.float64) - ref)[interior].max())
        check(f"AA downsample == torch bicubic antialias ({DEV.type} f32)", err < 2e-2, f"max|d|={err:.2e}")


def test_aa_matches_torch_antialias_upsample():
    """Upsampling under the AA policy is linear: == F.interpolate(mode='bilinear', antialias=True)."""
    img = _smooth_field((40, 36), seed=6)
    vol = img[None, None]
    ours = resample_data_or_seg_to_shape_gpu(vol, (1, 68, 61), device="cpu", anti_alias=True)[0, 0]
    ref = F.interpolate(torch.as_tensor(img)[None, None], size=(68, 61), mode="bilinear", antialias=True, align_corners=False)[0, 0].numpy()
    err = float(np.abs(ours - ref).max())
    check("AA upsample == torch bilinear antialias", err < 1e-6, f"max|d|={err:.2e}")


def test_aa_3d_is_separable_product():
    """The 3-D AA resample equals the 2-D torch reference applied per slice followed by the 1-D z filter
    (separability), checked against torch's antialias on each axis pair."""
    rng = np.random.default_rng(7)
    vol = np.stack([_smooth_field((64, 56), seed=10 + i) for i in range(30)], axis=0)[None]   # (1, 30, 64, 56)
    full = resample_data_or_seg_to_shape_gpu(vol, (13, 28, 24), device="cpu", anti_alias=True)[0]
    # y,z first via torch (per x-slice), then x via torch on (z... ) -> use ours for the last axis instead:
    step = F.interpolate(torch.as_tensor(vol[0])[:, None], size=(28, 24), mode="bicubic", antialias=True, align_corners=False)[:, 0].numpy()
    # (30, 28, 24) -> resample axis 0 with the same 1-D filter via F.interpolate on a transposed view
    step_t = torch.as_tensor(step).permute(1, 2, 0).reshape(1, 1, 28 * 24, 30)
    z = F.interpolate(step_t, size=(28 * 24, 13), mode="bicubic", antialias=True, align_corners=False)[0, 0].reshape(28, 24, 13).permute(2, 0, 1).numpy()
    interior = (full > vol.min() + 1e-9) & (full < vol.max() - 1e-9)
    err = float(np.abs(full - z)[interior].max())
    check("AA 3-D == separable torch antialias", err < 1e-6 and interior.mean() > 0.9, f"max|d|={err:.2e} on {interior.mean()*100:.0f}%")


if __name__ == "__main__":
    print(f"device={DEV}")
    for t in [test_noop, test_upsample_matches_trilinear, test_downsample_antialiases,
              test_cubic_no_overshoot, test_seg_label_preserving, test_container_contract,
              test_channel_chunk_equiv, test_discoverable,
              test_scipy_parity_float64_cpu, test_scipy_parity_float32_gpu, test_scipy_parity_int16_rounding,
              test_scipy_seg_nearest_exact, test_scipy_zoom_torch_wrapper,
              test_center_parity_nnunet, test_center_seg_parity_nnunet, test_aa_policy_is_opt_in,
              test_separate_z_parity_nnunet_data, test_separate_z_parity_nnunet_seg, test_seg_threshold_exact_half,
              test_aa_matches_torch_antialias_downsample, test_aa_matches_torch_antialias_upsample, test_aa_3d_is_separable_product]:
        t()
    n_ok, n = sum(PASS), len(PASS)
    print(f"\n{n_ok}/{n} checks passed")
    raise SystemExit(0 if n_ok == n else 1)
