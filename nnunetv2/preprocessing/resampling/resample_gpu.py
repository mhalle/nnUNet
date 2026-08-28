"""GPU implementation of nnU-Net's default resampler.

``resample_data_or_seg_to_shape_gpu`` is ``resample_data_or_seg_to_shape`` on MPS / CUDA / CPU:
same sampling convention, spline prefilter, per-channel clip, label rules and separate-z policy,
verified against the original to float precision (see ``nnunetv2/tests/test_resample_gpu.py``).
It is selected by name from ``plans.json`` like any other resampling function (planners in
``experiment_planning/experiment_planners/resampling/resample_with_gpu.py``), and it is
plain torch, so it runs on Apple Silicon where neither scipy nor ``F.interpolate``'s 3-D
trilinear path can use the GPU.

Each axis is applied as a dense ``(n_out, n_in)`` matrix via matmul. The exact operators are
obtained by probing ``scipy.ndimage.zoom`` with an identity matrix (the resampler is linear
and separable, so the probe captures prefilter, boundary mode and coordinate map without
re-implementing any of them); ``anti_alias=True`` swaps in a PIL-style anti-aliased policy
(Catmull-Rom scaled by the factor), which is opt-in because models trained with the default
resampler are not trained on anti-aliased inputs. See ``documentation/gpu_resampling.md``.
"""
from typing import Union, Tuple, List

import functools
import numpy as np
import torch



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

    Returns ``(w, cubic)``, or ``None`` when ``n_in == n_out`` (identity - caller
    skips the axis). The matrix is banded (only ~``n_taps`` nonzeros per row) but
    applied as a dense matmul: on MPS/CUDA the GEMM path is far better optimized
    than a per-tap ``index_select`` gather (measured ~3x faster on MPS), so the
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


@functools.lru_cache(maxsize=128)
def _scipy_axis_matrix(n_in: int, n_out: int, order: int, mode: str, grid_mode: bool = False) -> np.ndarray:
    """Exact 1-D operator of ``scipy.ndimage.zoom`` along one axis as an ``(n_out, n_in)``
    float64 matrix.

    ``zoom`` is linear and separable, so zooming an identity matrix along one axis
    yields the operator itself - spline prefilter, boundary ``mode`` and the
    corner-aligned coordinate map (``j * (n_in-1) / (n_out-1)``, ``grid_mode=False``)
    included, for any ``order``. Nothing of scipy's boundary handling is re-implemented,
    hence nothing to get subtly wrong; cached per (n_in, n_out, order, mode).
    """
    from scipy import ndimage
    if n_in == n_out:
        return np.eye(n_in)
    probe = ndimage.zoom(np.eye(n_in, dtype=np.float64), (1.0, n_out / n_in), order=order, mode=mode, grid_mode=grid_mode)
    if probe.shape != (n_in, n_out):
        raise RuntimeError(f"scipy zoom probe produced {probe.shape}, expected {(n_in, n_out)}")
    return np.ascontiguousarray(probe.T)


@functools.lru_cache(maxsize=128)
def _scipy_nn_index(n_in: int, n_out: int, mode: str, grid_mode: bool = False) -> np.ndarray:
    """Input index chosen by ``scipy.ndimage.zoom(order=0)`` for each output index (exact)."""
    w = _scipy_axis_matrix(n_in, n_out, 0, mode, grid_mode)          # (n_out, n_in), one 1 per row
    return np.ascontiguousarray(w.argmax(axis=1))


def _axis_operator(n_in, n_out, convention, order, mode, aa_threshold, device, dtype, anti_alias=False):
    """``(w, cubic)`` for one axis, or ``None`` for identity.

    * ``"corner"``: exact ``scipy.ndimage.zoom`` operator (voxel-corner point grid).
    * ``"center"``, ``anti_alias=False``: exact ``zoom(grid_mode=True)`` operator (voxel-center),
      i.e. what skimage ``resize`` and nnU-Net's ``resample_data_or_seg_to_shape`` compute.
    * ``"center"``, ``anti_alias=True``: the anti-aliased policy of :func:`_axis_weights`.
    """
    if n_in == n_out:
        return None
    if convention == "corner":
        w = torch.as_tensor(_scipy_axis_matrix(int(n_in), int(n_out), int(order), str(mode), False), device=device, dtype=dtype)
        return w, order >= 2
    if convention != "center":
        raise ValueError(f"unknown convention {convention!r}; expected 'center' or 'corner'")
    if not anti_alias:
        w = torch.as_tensor(_scipy_axis_matrix(int(n_in), int(n_out), int(order), str(mode), True), device=device, dtype=dtype)
        return w, order >= 2
    return _axis_weights(n_in, n_out, aa_threshold, device, dtype)


def _resample_axis(x: torch.Tensor, axis: int, w: torch.Tensor) -> torch.Tensor:
    """Apply ``(n_out, n_in)`` matrix ``w`` along ``axis`` of ``x`` via matmul.

    History worth keeping: an earlier streaming variant of this contracted over a two-column
    band rather than the whole axis, and on MPS those very small matmuls returned values
    wrong by ~2 absolute against CPU and einsum. It was worked around with einsum at the
    time. It does not reproduce on torch 2.13 - matmul there is bit-identical to CPU and to
    einsum for n_in in {2, 3, 4, 8, 33, 167} - so no workaround is carried here. Noted so
    that a discrepancy on an older torch is recognizable, and so the workaround is not
    reintroduced for a defect that is gone.
    """
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
    convention: str = "center",
    order: int = 3,
    mode: str = "nearest",
    anti_alias: bool = False,
) -> torch.Tensor:
    device = data.device
    out = data
    did_cubic = False
    for sp_axis in range(3):                # spatial axes -> tensor dims 1,2,3
        n_in = out.shape[sp_axis + 1]
        n_out = int(new_shape[sp_axis])
        res = _axis_operator(n_in, n_out, convention, order, mode, aa_threshold, device, out.dtype, anti_alias)
        if res is None:
            continue
        w, cubic = res
        if cubic:
            did_cubic = True
        out = _resample_axis(out, sp_axis + 1, w)
    # scipy.ndimage.zoom never clips ("corner"). skimage.resize clips the output to the input's
    # value range per call (clip=True), which nnU-Net inherits per channel; the anti-aliased
    # policy clips cubic ringing for the same reason. Per channel, so channel chunking is exact.
    if clamp_range and did_cubic and convention == "center":
        lo = data.amin(dim=(1, 2, 3), keepdim=True)
        hi = data.amax(dim=(1, 2, 3), keepdim=True)
        out = torch.maximum(torch.minimum(out, hi), lo)
    return out


def _center_axis_matrix_t(n_in, n_out, order, mode, device, dtype):
    return torch.as_tensor(_scipy_axis_matrix(int(n_in), int(n_out), int(order), str(mode), True), device=device, dtype=dtype)


def _nn_gather_center(t: torch.Tensor, dim: int, n_out: int, mode: str) -> torch.Tensor:
    """Exact nearest-neighbor gather of zoom(order=0, grid_mode=True) along tensor dim ``dim``."""
    n_in = t.shape[dim]
    if n_in == n_out:
        return t
    idx = torch.as_tensor(_scipy_nn_index(int(n_in), int(n_out), str(mode), True), device=t.device)
    return t.index_select(dim, idx)


def _separate_z_data(t: torch.Tensor, new_shape, axis: int, order: int, order_z: int, mode: str = "nearest") -> torch.Tensor:
    """nnU-Net's separate-z data path, exactly: each slice along ``axis`` is skimage-resized
    in-plane with ``order`` (clipped to that slice's input range, as resize(clip=True) does per
    2-D call), then the low-resolution axis is resampled with ``order_z`` at half-pixel
    positions (map_coordinates, mode='nearest'), with no further clip."""
    out = t
    inplane = [sp for sp in range(3) if sp != axis]
    for sp in inplane:
        n_in, n_out = out.shape[sp + 1], int(new_shape[sp])
        if n_in != n_out:
            out = _resample_axis(out, sp + 1, _center_axis_matrix_t(n_in, n_out, order, mode, t.device, t.dtype))
    dims = tuple(sp + 1 for sp in inplane)
    lo = t.amin(dim=dims, keepdim=True)
    hi = t.amax(dim=dims, keepdim=True)
    out = torch.maximum(torch.minimum(out, hi), lo)
    n_in, n_out = out.shape[axis + 1], int(new_shape[axis])
    if n_in != n_out:
        if order_z == 0:
            out = _nn_gather_center(out, axis + 1, n_out, mode)
        else:
            out = _resample_axis(out, axis + 1, _center_axis_matrix_t(n_in, n_out, order_z, mode, t.device, t.dtype))
    return out


def _separate_z_seg(t: torch.Tensor, new_shape, axis: int, order: int, order_z: int, chunk_labels: int, mode: str = "nearest") -> torch.Tensor:
    """nnU-Net's separate-z label path, exactly: in-plane ``resize_segmentation`` per slice
    (order 0: nearest; else per-label indicator resized with ``order``, painted where ``>= 0.5``
    in ascending label order), then along ``axis`` with ``order_z`` (0: nearest gather of the
    label values; else per-label indicator, painted where ``round(v) > 0.5`` i.e. ``v > 0.5``)."""
    assert t.shape[0] == 1, "seg resampling expects a single channel"
    inplane = [sp for sp in range(3) if sp != axis]
    mid_shape = list(new_shape)
    mid_shape[axis] = t.shape[axis + 1]
    unique = torch.unique(t)
    result_dtype = torch.int16 if int(unique.max()) > 127 else torch.int8
    if order == 0:
        cur = t
        for sp in inplane:
            cur = _nn_gather_center(cur, sp + 1, int(new_shape[sp]), mode)
        cur = cur.to(result_dtype)
    else:
        cur = torch.zeros((1, *mid_shape), dtype=result_dtype, device=t.device)
        for start in range(0, len(unique), chunk_labels):
            labs = unique[start:start + chunk_labels]
            soft = (t[0][None] == labs.view(-1, 1, 1, 1)).float()
            for sp in inplane:
                n_in, n_out = soft.shape[sp + 1], int(new_shape[sp])
                if n_in != n_out:
                    soft = _resample_axis(soft, sp + 1, _center_axis_matrix_t(n_in, n_out, order, mode, t.device, soft.dtype))
            for i in range(len(labs)):
                cur[0][soft[i] >= 0.5] = labs[i].to(result_dtype)
    n_in, n_out = cur.shape[axis + 1], int(new_shape[axis])
    if n_in == n_out:
        return cur
    if order_z == 0:
        return _nn_gather_center(cur, axis + 1, n_out, mode)
    unique2 = torch.unique(cur)
    out = torch.zeros((1, *new_shape), dtype=result_dtype, device=t.device)
    for start in range(0, len(unique2), chunk_labels):
        labs = unique2[start:start + chunk_labels]
        soft = (cur[0][None] == labs.view(-1, 1, 1, 1)).float()
        soft = _resample_axis(soft, axis + 1, _center_axis_matrix_t(n_in, n_out, order_z, mode, t.device, soft.dtype))
        for i in range(len(labs)):
            out[0][soft[i] > 0.5] = labs[i].to(result_dtype)
    return out


def resample_data_or_seg_to_shape_gpu(
    data: Union[torch.Tensor, np.ndarray],
    new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray] = None,
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray] = None,
    is_seg: bool = False,
    device: Union[torch.device, str, None] = None,
    aa_threshold: float = 1.1,
    seg_resample_chunk_labels: int = 64,
    channel_chunk: int = 8,
    convention: str = "center",
    order: int = 3,
    mode: str = "nearest",
    anti_alias: bool = False,
    order_z: int = 0,
    force_separate_z: Union[bool, None] = False,
    separate_z_anisotropy_threshold: float = None,
    **kwargs,
) -> Union[torch.Tensor, np.ndarray]:
    """GPU separable resample. ``data`` must be ``(c, x, y, z)``.

    **By default this is nnU-Net's own resampler, on the GPU.** With
    ``convention="center"`` and ``anti_alias=False`` (the defaults) the result equals
    ``resample_data_or_seg_to_shape`` / skimage ``resize(order, mode="edge",
    anti_aliasing=False)`` to float precision on CPU (float64) and to ~1e-4 relative on
    MPS/CUDA (float32): voxel-center sampling, spline prefilter, skimage's per-channel clip to
    the input range, and for ``is_seg=True`` nnU-Net's own label rule (each label's resized
    indicator thresholded at 0.5 and painted in ascending label order; ``order=0`` is an exact
    nearest-neighbor gather). nnU-Net's *separate-z* policy is replicated too: when
    ``current_spacing`` / ``new_spacing`` are given and the data are anisotropic beyond
    ``separate_z_anisotropy_threshold`` (nnU-Net's ``ANISO_THRESHOLD``, 3) - or when
    ``force_separate_z`` - the in-plane axes are resampled with ``order`` (per-slice clip, as
    skimage does per 2-D call) and the low-resolution axis with ``order_z``, labels included
    (in-plane ``>= 0.5`` rule, along-axis ``> 0.5`` rule, exactly as upstream). The decision
    itself is upstream's ``determine_do_sep_z_and_axis``. Without spacing information no
    separate-z is performed (upstream always passes it).

    Two sampling conventions, selected with ``convention`` and named by where the value
    sits in its voxel:

    * ``"center"`` (default) - voxel-center: the value sits at the center of a cell that
      tiles the field of view, so output sample ``j`` reads input coordinate
      ``(j + 0.5) * n_in/n_out - 0.5`` (half-pixel, ``align_corners=False``; skimage
      ``resize``, ``F.interpolate``, ITK, nnU-Net's own resampler). Exact skimage/nnU-Net
      operator for ``order`` / ``mode`` when ``anti_alias=False``; with ``anti_alias=True``
      an anti-aliased policy instead - Catmull-Rom scaled by the factor when downsampling by
      more than ``aa_threshold``, linear otherwise (``order`` / ``mode`` ignored).
    * ``"corner"`` - voxel-corner point grid, exactly as ``scipy.ndimage.zoom(order=order,
      mode=mode, grid_mode=False)``: values are points at ``i * spacing`` and the rescale
      preserves the span of those points, ``j * (n_in-1)/(n_out-1)`` (``align_corners=True``;
      TotalSegmentator's ``change_spacing``). Spline prefilter, no anti-aliasing. Results
      match scipy to float precision on CPU (float64 in -> float64 math) and to ~1e-4
      relative on MPS/CUDA (float32); integer inputs are rounded half-away-from-zero as
      scipy does; ``is_seg=True, order=0`` is the exact ``zoom(order=0)`` label gather.
      A corner-sampled grid that preserved the *cell* extent instead (``j * n_in/n_out``,
      the spacing-exact convention some fused kernels use) differs by ``(n-1)/n`` per axis;
      that is a third convention, not offered here.

    Anti-aliasing at inference is a distribution shift for models trained with the
    scipy/skimage resamplers (it lowers recall on sub-centimeter structures), so it is
    opt-in (``anti_alias=True``) and meant for models trained with it. For existing
    nnU-Net models use the defaults; for TotalSegmentator's own pre-resampling use
    ``convention="corner"``.

    Cost for ``is_seg=True`` scales as the label count times the output volume: every label
    is resampled as its own indicator, in groups of ``seg_resample_chunk_labels``. That is
    the right shape of work at the model grid, where nnU-Net resamples segmentations, and
    the wrong one for a many-label map at full resolution. Measured on a 112-label whole-body
    map restored to a 418 M voxel grid: 495 s and 7.8 GB, against 30 s and 2.3 GB for a CPU
    implementation that crops each label to its own bounding box - anatomy is local, so those
    boxes sum to roughly one volume however many labels there are. For that direction prefer
    ``order=0``, or an inverse that decides per output voxel instead of per label.

    ``device`` defaults to the best one available. nnU-Net preprocesses in several worker
    processes (``-npp``), and each worker that resamples on the GPU holds a context and a
    working set there, so keep ``-npp`` low when this is the plans' resampler. Note that the
    reason it defaults high is that resampling is slow on the CPU, which is the thing this
    changes: a couple of workers is plenty once a resample costs tens of milliseconds.

    Signature-compatible with nnU-Net's ``resampling_fn_data`` /
    ``resampling_fn_seg`` / ``resampling_fn_probabilities`` (extra plans kwargs
    such as ``force_separate_z`` / ``order`` are accepted and ignored - the
    per-axis policy subsumes them). ``current_spacing`` / ``new_spacing`` are
    accepted for interface parity but the resample is driven purely by
    ``new_shape`` (as nnU-Net always passes an explicit target shape).

    Returns the same container type it was given (numpy in -> numpy out; tensor
    in -> tensor on its original device).
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

    # float64 math on CPU for scipy parity when the input is float64; float32 elsewhere.
    is_f64 = (data.dtype == np.float64) if input_was_numpy else (data.dtype == torch.float64)
    exact = convention == "corner" or not anti_alias
    # nnU-Net's separate-z decision (only meaningful for the exact voxel-center operator)
    do_separate_z, sep_axis = False, None
    if convention == "center" and not anti_alias:
        if current_spacing is not None and new_spacing is not None:
            from nnunetv2.configuration import ANISO_THRESHOLD
            from nnunetv2.preprocessing.resampling.default_resampling import determine_do_sep_z_and_axis
            thr = ANISO_THRESHOLD if separate_z_anisotropy_threshold is None else separate_z_anisotropy_threshold
            do_separate_z, sep_axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing, thr)
            sep_axis = None if sep_axis is None else int(sep_axis)
        elif force_separate_z:
            raise ValueError("force_separate_z=True needs current_spacing to pick the low-resolution axis")
    work = torch.float64 if (device.type == "cpu" and is_f64) else torch.float32    # float64 math on CPU for float64 input
    with torch.no_grad():
        if input_was_numpy:
            t = torch.as_tensor(np.ascontiguousarray(data)).to(device)
        else:
            t = data.to(device)

        if do_separate_z and is_seg:
            result = _separate_z_seg(t, new_shape, sep_axis, order, order_z, seg_resample_chunk_labels, mode)
        elif do_separate_z:
            C = t.shape[0]
            result = torch.empty((C, *new_shape), dtype=work, device=device)
            for s in range(0, C, channel_chunk):
                e = min(s + channel_chunk, C)
                result[s:e] = _separate_z_data(t[s:e].to(work), new_shape, sep_axis, order, order_z, mode)
        elif is_seg:
            result = _resample_seg(t, new_shape, aa_threshold, seg_resample_chunk_labels,
                                   convention=convention, order=order, mode=mode, anti_alias=anti_alias)
        elif t.shape[0] > channel_chunk:
            # Multi-channel (e.g. K-class probabilities/logits at export): the
            # per-axis matmul materializes a full (C, *new_shape) intermediate
            # per axis, which blows up memory for large C (e.g. K=117 at full
            # acquisition resolution). Resample in channel chunks into a
            # preallocated output so peak memory stays ~chunk-sized.
            C = t.shape[0]
            result = torch.empty((C, *new_shape), dtype=work, device=device)
            for s in range(0, C, channel_chunk):
                e = min(s + channel_chunk, C)
                result[s:e] = _separable_resample(t[s:e].to(work), new_shape, aa_threshold,
                                                  convention=convention, order=order, mode=mode, anti_alias=anti_alias)
        else:
            result = _separable_resample(t.to(work), new_shape, aa_threshold,
                                         convention=convention, order=order, mode=mode, anti_alias=anti_alias)

        if input_was_numpy:
            r = result.cpu().numpy()
            if exact and np.issubdtype(data.dtype, np.integer) and not is_seg:
                r = np.where(r > 0, r + 0.5, r - 0.5)      # scipy's / skimage's integer-output rounding
            result = r.astype(data.dtype, copy=False)
        else:
            result = result.to(orig_device)
            if (not is_seg) and data.dtype.is_floating_point and result.dtype != data.dtype:
                result = result.to(data.dtype)
    return result


def _resample_seg(
    data: torch.Tensor,                     # (C, X, Y, Z), integer labels
    new_shape: Tuple[int, int, int],
    aa_threshold: float,
    chunk_labels: int,
    convention: str = "center",
    order: int = 3,
    mode: str = "nearest",
    anti_alias: bool = False,
) -> torch.Tensor:
    """Label-preserving resample.

    * exact conventions with ``order=0``: the nearest-neighbor gather of ``zoom(order=0)``;
    * ``"center"`` without anti-aliasing: nnU-Net's rule (``resize_segmentation``) - resize
      each label's indicator, paint where ``>= 0.5`` in ascending label order (later labels
      overwrite; unpainted voxels are 0);
    * otherwise: one-hot -> separable resample -> argmax, in label chunks to bound memory.
    Assumes a single channel (C == 1), as nnU-Net seg.
    """
    assert data.shape[0] == 1, "seg resampling expects a single channel"
    exact = convention == "corner" or not anti_alias
    if exact and order == 0:
        out = data
        for sp_axis in range(3):
            n_in, n_out = out.shape[sp_axis + 1], int(new_shape[sp_axis])
            if n_in == n_out:
                continue
            idx = torch.as_tensor(_scipy_nn_index(int(n_in), int(n_out), str(mode), convention == "center"), device=out.device)
            out = out.index_select(sp_axis + 1, idx)
        return out
    unique = torch.unique(data)
    result_dtype = torch.int16 if int(unique.max()) > 127 else torch.int8
    if convention == "center" and not anti_alias:
        out = torch.zeros((1, *new_shape), dtype=result_dtype, device=data.device)
        for start in range(0, len(unique), chunk_labels):
            labs = unique[start:start + chunk_labels]
            onehot = (data[0][None] == labs.view(-1, 1, 1, 1)).float()
            soft = _separable_resample(onehot, new_shape, aa_threshold, clamp_range=False,
                                       convention=convention, order=order, mode=mode, anti_alias=False)
            for i in range(len(labs)):
                out[0][soft[i] >= 0.5] = labs[i].to(result_dtype)     # resize_segmentation: >= 0.5
        return out
    best_val = None
    best_lab = None
    for start in range(0, len(unique), chunk_labels):
        labs = unique[start:start + chunk_labels]
        onehot = (data[0][None] == labs.view(-1, 1, 1, 1)).float()  # (L, X, Y, Z)
        soft = _separable_resample(onehot, new_shape, aa_threshold, clamp_range=False,
                                   convention=convention, order=order, mode=mode, anti_alias=anti_alias)
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
    out = resample_data_or_seg_to_shape_gpu(x, (32, 64, 64), device="cpu")
    print("data:", x.shape, "->", out.shape, "range", float(out.min()), float(out.max()))
    seg = torch.zeros(1, 64, 128, 128, dtype=torch.int16)
    seg[:, 16:48, 32:96, 32:96] = 3
    outs = resample_data_or_seg_to_shape_gpu(seg, (32, 64, 64), is_seg=True, device="cpu")
    print("seg:", seg.shape, "->", outs.shape, "labels", torch.unique(outs).tolist())
