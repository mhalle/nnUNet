"""GPU anti-aliased separable resampling for nnU-Net (Torch, MPS/CUDA/CPU).

Drop-in replacement for ``resample_data_or_seg_to_shape`` /
``resample_torch_fornnunet`` that addresses two gaps in the existing
resamplers:

1. **Anti-aliasing on downsampling.** The scipy default
   (``resample_data_or_seg``) sets ``anti_aliasing=False`` and relies on an
   order-3 spline that does not band-limit; the torch path uses
   ``F.interpolate(mode='trilinear', antialias=False)`` and *cannot* set
   ``antialias=True`` at all (Torch supports the anti-alias flag only for the
   2-D ``bilinear``/``bicubic`` modes, never for 3-D ``trilinear``). Either way,
   shrinking a volume by a large factor undersamples — a 2-tap/order-3 kernel
   sees far fewer than the ``f`` source voxels that map onto one output voxel,
   so thin / high-contrast structure aliases.

2. **GPU execution.** scipy resampling is CPU-bound and is the dominant cost of
   nnU-Net preprocessing on Apple Silicon / single-GPU boxes. Everything here is
   plain Torch ops (per-axis matmul), so it runs on ``mps`` / ``cuda``.

Per-axis policy, keyed on the resample factor ``f = n_in / n_out`` for that axis:

* ``f > aa_threshold`` (downsampling): factor-scaled Catmull-Rom cubic. The
  kernel support is stretched by ``f`` so it averages the whole output-voxel
  footprint — i.e. a genuine anti-aliasing prefilter, not point interpolation.
* ``f <= aa_threshold`` (upsampling / near-identity): linear. Anti-aliasing
  does not apply when upsampling, and cubic's negative lobes ring/overshoot at
  high-contrast edges (e.g. inventing haloes between the sparse slices of
  thick-slice CT when the through-plane axis is upsampled). Linear is monotone,
  so it never invents values outside the data.

This per-axis decision generalizes nnU-Net's ``do_separate_z`` special-case: an
anisotropic through-plane axis that is being upsampled automatically gets the
linear (non-ringing) treatment, while the in-plane axes that are being
downsampled get the anti-aliased cubic — no explicit separate-z branch needed.

Half-pixel-center sampling (``align_corners=False``), matching skimage ``resize``
and ``F.interpolate``, so it is a faithful drop-in. Catmull-Rom's negative lobes
can ring slightly past the data range at sharp edges; the cubic output is
clipped to the input's value range to remove that overshoot while keeping
interior sharpness (linear axes never ring, so the clip only ever touches cubic
ones).
"""
from typing import Union, Tuple, List, Optional

import numpy as np
import torch

from nnunetv2.configuration import ANISO_THRESHOLD


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _catmull_rom(x: torch.Tensor) -> torch.Tensor:
    """Catmull-Rom cubic (a = -0.5), support radius 2 in kernel units."""
    ax = x.abs()
    ax2 = ax * ax
    ax3 = ax2 * ax
    w = torch.zeros_like(ax)
    m1 = ax < 1.0
    m2 = (ax >= 1.0) & (ax < 2.0)
    w = torch.where(m1, 1.5 * ax3 - 2.5 * ax2 + 1.0, w)
    w = torch.where(m2, -0.5 * ax3 + 2.5 * ax2 - 4.0 * ax + 2.0, w)
    return w


def _axis_weights(
    n_in: int,
    n_out: int,
    aa_threshold: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
):
    """Dense ``(n_out, n_in)`` resample matrix + cubic flag for one axis.

    Returns ``(w, cubic)``, or ``None`` when ``n_in == n_out`` (identity — caller
    skips the axis). The matrix is banded (only ~``n_taps`` nonzeros per row) but
    applied as a dense matmul: on MPS/CUDA the GEMM path is far better optimized
    than a per-tap ``index_select`` gather (measured ~3× faster on MPS), so the
    wasted FLOPs are more than repaid. Rows are normalized so weights sum to 1,
    which makes boundary tap loss behave like edge replication.
    """
    if n_in == n_out:
        return None

    f = n_in / n_out                       # > 1 downsample, < 1 upsample
    cubic = f > aa_threshold
    scale = f if cubic else 1.0

    # Output-voxel centers in source-index space (half-pixel / align_corners=False).
    j = torch.arange(n_out, device=device, dtype=dtype)
    c = (j + 0.5) * f - 0.5                 # (n_out,)
    k = torch.arange(n_in, device=device, dtype=dtype)  # (n_in,)

    x = (k[None, :] - c[:, None]) / scale   # (n_out, n_in)
    w = _catmull_rom(x) if cubic else torch.clamp(1.0 - x.abs(), min=0.0)
    w = w / torch.clamp(w.sum(dim=1, keepdim=True), min=1e-8)
    return w.to(dtype), cubic


def _resample_axis(x: torch.Tensor, axis: int, w: torch.Tensor) -> torch.Tensor:
    """Apply ``(n_out, n_in)`` matrix ``w`` along ``axis`` of ``x`` via matmul."""
    x = x.movedim(axis, -1)                 # (..., n_in)
    shp = x.shape
    x2 = x.reshape(-1, shp[-1])             # (M, n_in)
    o = x2 @ w.t()                          # (M, n_out)
    o = o.reshape(*shp[:-1], w.shape[0])
    return o.movedim(-1, axis)


def _separable_resample(
    data: torch.Tensor,                     # (C, X, Y, Z) float
    new_shape: Tuple[int, int, int],
    aa_threshold: float,
    clamp_range: bool = True,
) -> torch.Tensor:
    device = data.device
    out = data
    did_cubic = False
    for sp_axis in range(3):                # spatial axes -> tensor dims 1,2,3
        n_in = out.shape[sp_axis + 1]
        n_out = int(new_shape[sp_axis])
        res = _axis_weights(n_in, n_out, aa_threshold, device, out.dtype)
        if res is None:
            continue
        w, cubic = res
        if cubic:
            did_cubic = True
        out = _resample_axis(out, sp_axis + 1, w)
    if clamp_range and did_cubic:
        # Clip cubic ringing to the input's value range (interior sharpness kept).
        lo = data.amin()
        hi = data.amax()
        out = torch.clamp(out, lo, hi)
    return out


def resample_aa_torch(
    data: Union[torch.Tensor, np.ndarray],
    new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray] = None,
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray] = None,
    is_seg: bool = False,
    device: Union[torch.device, str, None] = None,
    aa_threshold: float = 1.1,
    seg_resample_chunk_labels: int = 64,
    channel_chunk: int = 8,
    **kwargs,
) -> Union[torch.Tensor, np.ndarray]:
    """GPU anti-aliased separable resample. ``data`` must be ``(c, x, y, z)``.

    Signature-compatible with nnU-Net's ``resampling_fn_data`` /
    ``resampling_fn_seg`` / ``resampling_fn_probabilities`` (extra plans kwargs
    such as ``force_separate_z`` / ``order`` are accepted and ignored — the
    per-axis policy subsumes them). ``current_spacing`` / ``new_spacing`` are
    accepted for interface parity but the resample is driven purely by
    ``new_shape`` (as nnU-Net always passes an explicit target shape).

    Returns the same container type it was given (numpy in → numpy out; tensor
    in → tensor on its original device).
    """
    assert data.ndim == 4, "data must be (c, x, y, z)"
    new_shape = tuple(int(i) for i in new_shape)

    input_was_numpy = isinstance(data, np.ndarray)
    if input_was_numpy:
        orig_device = torch.device("cpu")
    else:
        orig_device = data.device

    if all(int(s) == int(n) for s, n in zip(data.shape[1:], new_shape)):
        return data  # no-op, same as upstream

    if device is None:
        device = _best_device()
    device = torch.device(device)

    with torch.no_grad():
        if input_was_numpy:
            t = torch.as_tensor(np.ascontiguousarray(data)).to(device)
        else:
            t = data.to(device)

        if is_seg:
            result = _resample_seg(t, new_shape, aa_threshold, seg_resample_chunk_labels)
        elif t.shape[0] > channel_chunk:
            # Multi-channel (e.g. K-class probabilities/logits at export): the
            # per-axis matmul materializes a full (C, *new_shape) intermediate
            # per axis, which blows up memory for large C (e.g. K=117 at full
            # acquisition resolution). Resample in channel chunks into a
            # preallocated output so peak memory stays ~chunk-sized.
            C = t.shape[0]
            result = torch.empty((C, *new_shape), dtype=torch.float32, device=device)
            for s in range(0, C, channel_chunk):
                e = min(s + channel_chunk, C)
                result[s:e] = _separable_resample(t[s:e].float(), new_shape, aa_threshold)
        else:
            result = _separable_resample(t.float(), new_shape, aa_threshold)

        if input_was_numpy:
            result = result.cpu().numpy().astype(data.dtype, copy=False)
        else:
            result = result.to(orig_device)
    return result


def _resample_seg(
    data: torch.Tensor,                     # (C, X, Y, Z), integer labels
    new_shape: Tuple[int, int, int],
    aa_threshold: float,
    chunk_labels: int,
) -> torch.Tensor:
    """Label-preserving resample: one-hot -> separable resample -> argmax.

    Anti-aliased per-axis blending of the per-label indicator volumes, then the
    label whose (soft) indicator is largest wins. Processes unique labels in
    chunks to bound memory; assumes a single channel (C == 1), as nnU-Net seg.
    """
    assert data.shape[0] == 1, "seg resampling expects a single channel"
    unique = torch.unique(data)
    result_dtype = torch.int16 if int(unique.max()) > 127 else torch.int8
    best_val = None
    best_lab = None
    for start in range(0, len(unique), chunk_labels):
        labs = unique[start:start + chunk_labels]
        onehot = (data[0][None] == labs.view(-1, 1, 1, 1)).float()  # (L, X, Y, Z)
        soft = _separable_resample(onehot, new_shape, aa_threshold, clamp_range=False)
        chunk_best, chunk_idx = soft.max(dim=0)                     # (X', Y', Z')
        chunk_lab = labs[chunk_idx]
        if best_val is None:
            best_val, best_lab = chunk_best, chunk_lab
        else:
            take = chunk_best > best_val
            best_val = torch.where(take, chunk_best, best_val)
            best_lab = torch.where(take, chunk_lab, best_lab)
    return best_lab.to(result_dtype)[None]   # (1, X', Y', Z')


if __name__ == "__main__":
    # quick smoke test
    x = torch.zeros(1, 64, 128, 128)
    x[:, 16:48, 32:96, 32:96] = 1000.0
    out = resample_aa_torch(x, (32, 64, 64), device="cpu")
    print("data:", x.shape, "->", out.shape, "range", float(out.min()), float(out.max()))
    seg = torch.zeros(1, 64, 128, 128, dtype=torch.int16)
    seg[:, 16:48, 32:96, 32:96] = 3
    outs = resample_aa_torch(seg, (32, 64, 64), is_seg=True, device="cpu")
    print("seg:", seg.shape, "->", outs.shape, "labels", torch.unique(outs).tolist())
