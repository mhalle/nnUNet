# GPU anti-aliased resampling (`resample_aa_torch`)

A drop-in resampling function for nnU-Net that runs on **MPS / CUDA / CPU** and
**anti-aliases when downsampling** — two things the existing resamplers do not
both do.

Module: `nnunetv2/preprocessing/resampling/resample_gpu_aa.py`

## Why

nnU-Net has two resamplers today, and neither is a fast *and* correct GPU
downsampler:

| resampler | device | anti-aliased downsample? |
|---|---|---|
| `resample_data_or_seg_to_shape` (scipy, **default**) | CPU only | no (`anti_aliasing=False`, order-3 spline) |
| `resample_torch_fornnunet` (`F.interpolate`) | CPU only for 3-D* | no |
| **`resample_aa_torch` (this)** | **MPS / CUDA / CPU** | **yes** |

\* `aten::upsample_trilinear3d` is **not implemented for MPS** in current
PyTorch, so the torch resampler silently cannot use Apple-Silicon GPUs for 3-D
volumes (it errors unless `PYTORCH_ENABLE_MPS_FALLBACK=1` pushes it back to CPU).
`F.interpolate` also offers no anti-aliasing for `trilinear` — the `antialias`
flag is implemented only for the 2-D `bilinear`/`bicubic` modes.

Downsampling without a band-limiting prefilter aliases thin / high-contrast
structure: a 2-tap (trilinear) or order-3 spline kernel samples far fewer than
the `f` source voxels that map onto one output voxel when shrinking by `f×`.

## How it works

Per-axis separable resample, policy keyed on the factor `f = n_in / n_out`:

* **`f > aa_threshold` (downsampling):** factor-scaled **Catmull-Rom cubic** —
  kernel support stretched by `f` so it averages the whole output-voxel
  footprint (a real anti-aliasing prefilter). Output is clipped to the input
  value range to remove cubic ringing while keeping interior sharpness.
* **`f <= aa_threshold` (upsampling / near-identity):** **linear** — monotone, so
  it never invents out-of-range values (cubic's negative lobes ring across
  high-contrast edges, e.g. between the sparse slices of thick-slice CT).

This per-axis decision generalizes nnU-Net's `do_separate_z` special case: an
anisotropic through-plane axis being upsampled automatically gets the
non-ringing linear path while the in-plane axes being downsampled get
anti-aliased cubic — no explicit separate-z branch.

Each axis is applied as a **dense banded matmul** (`(n_out, n_in)` weight matrix).
On MPS/CUDA the GEMM path is far better optimized than a per-tap `index_select`
gather (~3× faster on MPS). Multi-channel inputs (K-class probabilities at
export) are processed in **channel chunks** (`channel_chunk=8`) so peak memory
stays chunk-sized instead of materializing the full `(K, *new_shape)` volume.

Sampling is half-pixel-centered (`align_corners=False`), matching skimage
`resize` and `F.interpolate`, so it is a faithful drop-in.

## Use it for inference (no retraining)

The resampler is selected by name from the model's `plans.json`. To run an
existing pretrained model with GPU AA resampling, point the resampling
functions at this one and pass a device:

```jsonc
// in plans.json, per configuration:
"resampling_fn_data":          "resample_aa_torch",
"resampling_fn_data_kwargs":   {"device": "mps"},
"resampling_fn_seg":           "resample_aa_torch",
"resampling_fn_seg_kwargs":    {"is_seg": true, "device": "mps"},
"resampling_fn_probabilities": "resample_aa_torch",
"resampling_fn_probabilities_kwargs": {"device": "mps", "channel_chunk": 8}
```

`device` accepts `"mps"`, `"cuda"`, `"cpu"`, or omit it to auto-pick the best
available. Extra upstream plans kwargs (`order`, `force_separate_z`, …) are
accepted and ignored.

## Benchmarks (M2, 16 GB, MPS)

`scripts/bench_resample_gpu_aa.py` (forward + quality) and
`scripts/bench_resample_inverse.py` (device-resident + K-channel):

| path | scipy default (CPU) | `resample_aa_torch` (MPS) | speedup |
|---|---|---|---|
| forward `512²×165 → 227²×220` | 3.8 s | 95 ms | **40×** |
| forward `768²×709 → 333²×473` | 27.4 s | 1.1 s | **24×** |
| inverse K=20  `96³ → 160³` | 2.6 s | 79 ms | **33×** |
| inverse K=50  `→ 128²×160` | 4.2 s | 118 ms | **36×** |
| inverse K=117 `112²×128 → 192²×224` | 32.2 s | 0.87 s | **37×** |

Quality / correctness (`scripts/test_resample_gpu_aa.py`, 12 checks):

* upsample (linear) matches `F.interpolate` trilinear to `max|Δ| ≈ 3e-5`;
* 8× downsample of zero-mean high-frequency noise: AA output std **24.5** vs
  trilinear **55.4** (input std 100) — the prefilter collapses aliased variance
  a 2-tap kernel retains;
* cubic output stays within the input value range (no ringing overshoot);
* seg path is label-preserving; channel-chunked == single-shot.

> Note: `F.interpolate(trilinear)` on **CPU** is itself very fast (~60 ms on the
> chest volume) — but it has no anti-aliasing and cannot run on MPS for 3-D. The
> value of `resample_aa_torch` is anti-aliased downsampling that runs on the GPU,
> at 24–40× the nnU-Net scipy default that ships in stock plans.
