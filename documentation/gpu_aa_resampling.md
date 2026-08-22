# GPU anti-aliased resampling (`resample_aa_torch`)

A drop-in resampling function for nnU-Net that runs on **MPS / CUDA / CPU** and
**anti-aliases when downsampling** - two things the existing resamplers do not
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
`F.interpolate` also offers no anti-aliasing for `trilinear` - the `antialias`
flag is implemented only for the 2-D `bilinear`/`bicubic` modes.

Downsampling without a band-limiting prefilter aliases thin / high-contrast
structure: a 2-tap (trilinear) or order-3 spline kernel samples far fewer than
the `f` source voxels that map onto one output voxel when shrinking by `fx`.

## How it works

Per-axis separable resample, policy keyed on the factor `f = n_in / n_out`:

* **`f > aa_threshold` (downsampling):** factor-scaled **Catmull-Rom cubic** -
  kernel support stretched by `f` so it averages the whole output-voxel
  footprint (a real anti-aliasing prefilter). Output is clipped to the input
  value range to remove cubic ringing while keeping interior sharpness.
* **`f <= aa_threshold` (upsampling / near-identity):** **linear** - monotone, so
  it never invents out-of-range values (cubic's negative lobes ring across
  high-contrast edges, e.g. between the sparse slices of thick-slice CT).

This per-axis decision generalizes nnU-Net's `do_separate_z` special case: an
anisotropic through-plane axis being upsampled automatically gets the
non-ringing linear path while the in-plane axes being downsampled get
anti-aliased cubic - no explicit separate-z branch.

Each axis is applied as a **dense banded matmul** (`(n_out, n_in)` weight matrix).
On MPS/CUDA the GEMM path is far better optimized than a per-tap `index_select`
gather (~3x faster on MPS). Multi-channel inputs (K-class probabilities at
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
available. Extra upstream plans kwargs (`order`, `force_separate_z`, ...) are
accepted and ignored.

## Benchmarks (M2, 16 GB, MPS)

`scripts/bench_resample_gpu_aa.py` (forward + quality) and
`scripts/bench_resample_inverse.py` (device-resident + K-channel):

| path | scipy default (CPU) | `resample_aa_torch` (MPS) | speedup |
|---|---|---|---|
| forward `512^2x165 -> 227^2x220` | 3.8 s | 95 ms | **40x** |
| forward `768^2x709 -> 333^2x473` | 27.4 s | 1.1 s | **24x** |
| inverse K=20  `96^3 -> 160^3` | 2.6 s | 79 ms | **33x** |
| inverse K=50  `-> 128^2x160` | 4.2 s | 118 ms | **36x** |
| inverse K=117 `112^2x128 -> 192^2x224` | 32.2 s | 0.87 s | **37x** |

Quality / correctness (`scripts/test_resample_gpu_aa.py`, 12 checks):

* upsample (linear) matches `F.interpolate` trilinear to `max|delta| ~ 3e-5`;
* 8x downsample of zero-mean high-frequency noise: AA output std **24.5** vs
  trilinear **55.4** (input std 100) - the prefilter collapses aliased variance
  a 2-tap kernel retains;
* cubic output stays within the input value range (no ringing overshoot);
* seg path is label-preserving; channel-chunked == single-shot.

> Note: `F.interpolate(trilinear)` on **CPU** is itself very fast (~60 ms on the
> chest volume) - but it has no anti-aliasing and cannot run on MPS for 3-D. The
> value of `resample_aa_torch` is anti-aliased downsampling that runs on the GPU,
> at 24-40x the nnU-Net scipy default that ships in stock plans.

## Sampling conventions: `convention="center"` vs `convention="corner"`

Two resamplers can agree on "linear" or "cubic" and still disagree about *where* the
output samples sit. `resample_aa_torch` names the two conventions by where the value
sits in its voxel:

| `convention` | model | output sample `j` reads input coordinate | anti-aliasing | matches |
|---|---|---|---|---|
| `"center"` (default) | **voxel-center**: value at the center of a cell; cells tile the field of view | `(j + 0.5) * n_in/n_out - 0.5` (half-pixel, `align_corners=False`) | Catmull-Rom scaled by the factor when downsampling by > `aa_threshold` | skimage `resize`, `F.interpolate`, ITK, nnU-Net's own `resample_data_or_seg_to_shape` |
| `"corner"` | **voxel-corner point grid**: values are points at `i * spacing`; the rescale preserves the *span of the points* | `j * (n_in - 1) / (n_out - 1)` (`align_corners=True`, `grid_mode=False`) | none; `order` / `mode` honored | `scipy.ndimage.zoom`, i.e. TotalSegmentator's `change_spacing` |

A corner-sampled grid that preserved the *cell* extent instead (`j * n_in/n_out`, the
spacing-exact convention some fused kernels use) differs from `"corner"` by a factor
`(n-1)/n` per axis - 0.46 % here, up to ~2 native voxels at the far edge of a 768-voxel
axis. That is a third convention, not a variant of the second; it is not offered here.

`"corner"` is implemented by **probing `ndimage.zoom` itself**: the resampler is linear and
separable, so zooming an identity matrix along one axis yields that axis's exact
`(n_out x n_in)` operator - spline prefilter, boundary `mode` and coordinate map included -
which is then applied with the same per-axis matmul as the anti-aliased path. CPU float64
results equal scipy's to float precision; MPS/CUDA (float32) to ~1e-4 relative. Integer
inputs are rounded half-away-from-zero on output, as scipy does. `is_seg=True, order=0` is
the exact nearest-neighbor gather of `zoom(order=0)`.

Why this matters (measured on a chest CT with the TotalSegmentator 3 mm model,
2026-08-22): the two conventions place the 3 mm grid ~0.4 voxel apart, and because the
label upsample snaps every boundary to that grid, the convention alone moved mean Dice
against TS stock from 0.995 to 0.888 (46/110 labels below 0.9). Anti-aliasing on top of it
changed the mean only to 0.875 - but erased two sub-centimeter structures entirely
(gallbladder, adrenal), because the model was trained on scipy's aliased-but-sharp inputs.
Hence: **match the host pipeline's convention exactly, and treat anti-aliasing as opt-in
for models trained with it.** `convention="corner", order=3, mode="nearest"` reproduces
`change_spacing(order=3)` and reaches Dice 0.9966 against stock, at the re-run floor (0.9987).

```python
from nnunetv2.preprocessing.resampling.resample_gpu_aa import resample_aa_torch, scipy_zoom_torch

img3mm  = scipy_zoom_torch(data[None], new_shape, order=3, mode="nearest", device="mps")[0]   # == ndimage.zoom
labels  = scipy_zoom_torch(seg[None], native_shape, order=0, is_seg=True, device="mps")[0]   # == zoom(order=0)
aa_img  = resample_aa_torch(data[None], new_shape, device="mps")[0]                           # grid + AA (opt-in)
```
