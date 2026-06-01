"""Self-checking correctness tests for resample_aa_torch.

Run: uv run python scripts/test_resample_gpu_aa.py
(No pytest dependency — plain asserts, exits nonzero on failure.)
"""
import numpy as np
import torch
import torch.nn.functional as F

from nnunetv2.preprocessing.resampling.resample_gpu_aa import resample_aa_torch, _best_device
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
    check("upsample linear == torch trilinear", md < 1e-3, f"max|Δ|={md:.2e}")


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
    check("channel-chunked == single-shot", md < 1e-4, f"max|Δ|={md:.2e}")


def test_discoverable():
    fn = recursive_find_resampling_fn_by_name("resample_aa_torch")
    check("discoverable by plans name", fn.__name__ == "resample_aa_torch")


if __name__ == "__main__":
    print(f"device={DEV}")
    for t in [test_noop, test_upsample_matches_trilinear, test_downsample_antialiases,
              test_cubic_no_overshoot, test_seg_label_preserving, test_container_contract,
              test_channel_chunk_equiv, test_discoverable]:
        t()
    n_ok, n = sum(PASS), len(PASS)
    print(f"\n{n_ok}/{n} checks passed")
    raise SystemExit(0 if n_ok == n else 1)
