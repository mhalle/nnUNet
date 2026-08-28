"""Logit field -> labels on the grid you ask for.

The inverse leg of the pipeline: for every voxel of a caller-chosen grid, interpolate the K
logits from the model grid and decide (argmax, or per-region threshold + paint) - fused, so
nothing K-channel-sized is ever materialized at the output resolution.
"""
from __future__ import annotations

import numpy as np
import torch

from nnunetv2.inference.restore import backends
from nnunetv2.inference.restore.mapping import Mapping
from nnunetv2.inference.restore.tables import build_tables

MODES = ("argmax", "regions")


def _as_torch(logits) -> torch.Tensor:
    if isinstance(logits, torch.Tensor):
        return logits
    return torch.from_numpy(np.ascontiguousarray(logits))


def to_labels(logits, new_shape, mapping: Mapping, *, interp="linear", outside: str = "background",
              lut=None, mode: str = "argmax", paint: bool = False, threshold: float = 0.0,
              background: int = 0, out: torch.Tensor | None = None, out_dtype=None,
              backend: str = "auto", coord_dtype=np.float64, slab_voxels: int = 1 << 26) -> torch.Tensor:
    """Labels on ``new_shape`` from a ``(K, Z, Y, X)`` logit field.

    Parameters
    ----------
    logits : torch.Tensor or ndarray, ``(K, Zs, Ys, Xs)`` float32 / float16
        The field at the model's grid. Stays where it is (device, dtype).
    new_shape : (Z, Y, X)
        Shape of the output label volume.
    mapping : Mapping
        Output index -> model-grid coordinate, per axis. Build it from the
        output grid, the source image and the forward resampler's convention
        (``Mapping.between(...) >> Mapping.center/corner/spacing(...)``).
    interp : "linear" | "nearest" or a (Z, Y, X) tuple
        Per-axis interpolation. ``nearest`` along z reproduces nnU-Net's
        separate-z (``order_z=0``) export on anisotropic data.
    outside : "background" | "clamp"
        Output voxels beyond the model grid get ``background`` (or are skipped
        when painting), or sample the clamped edge.
    lut : sequence of K ints, optional
        Local channel -> output label. Defaults to the channel index.
    mode : "argmax" | "regions"
        ``argmax``: label of the first maximal channel. ``regions``: every
        channel above ``threshold`` paints ``lut[k]`` in channel order
        (later wins) - for sigmoid-head models.
    paint : bool
        Background-transparent write into ``out``: voxels whose decision is
        background (``argmax`` channel 0; no region above threshold) are left
        untouched. Compositing multi-model tasks = calling this once per part
        with that part's ``lut`` into one shared ``out``.
    background : int
        Label written for outside / no-region voxels when not painting.
    out : torch.Tensor, optional
        ``(Z, Y, X)`` uint8 / uint16 on ``logits.device``; allocated if absent.
    backend : "auto" | "metal" | "torch" | "triton"
    coord_dtype : np.float64 | np.float32
        Coordinate arithmetic. float64 matches scipy / skimage; float32 is offered
        for parity with kernels that compute coordinates in single precision.
    """
    lg = _as_torch(logits)
    if lg.ndim != 4:
        raise ValueError(f"logits must be (K, Z, Y, X); got shape {tuple(lg.shape)}")
    if not lg.dtype.is_floating_point:
        raise TypeError(f"logits must be floating point; got {lg.dtype}")
    K = int(lg.shape[0])
    src_shape = tuple(int(s) for s in lg.shape[1:])
    out_shape = tuple(int(x) for x in new_shape)
    if len(out_shape) != 3:
        raise ValueError(f"new_shape must be a (Z, Y, X) shape; got {new_shape!r}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}; got {mode!r}")
    lut_arr = np.arange(K, dtype=np.int64) if lut is None else np.asarray(lut, dtype=np.int64).reshape(-1)
    if lut_arr.shape[0] != K:
        raise ValueError(f"lut must have K={K} entries; got {lut_arr.shape[0]}")
    if lut_arr.min() < 0 or int(background) < 0:
        raise ValueError("labels must be non-negative")
    max_label = max(int(lut_arr.max()), int(background))

    if out is None:
        dt = out_dtype if out_dtype is not None else (torch.uint8 if max_label <= 255 else torch.uint16)
        out = torch.zeros(out_shape, dtype=dt, device=lg.device)
    else:
        if tuple(out.shape) != tuple(out_shape):
            raise ValueError(f"out has shape {tuple(out.shape)}, new_shape is {out_shape}")
        if out.device != lg.device:
            raise ValueError(f"out is on {out.device}, logits on {lg.device}")
    if out.dtype not in (torch.uint8, torch.uint16):
        raise TypeError(f"out must be uint8 or uint16; got {out.dtype}")
    if max_label > (255 if out.dtype == torch.uint8 else 65535):
        raise ValueError(f"label {max_label} does not fit {out.dtype}")

    tables = build_tables(out_shape, src_shape, mapping, interp=interp, outside=outside, coord_dtype=coord_dtype)
    name, mod = backends.select(backend, lg.device)
    opts = {"slab_voxels": int(slab_voxels)} if name == "metal" else {}
    mod.run(lg, out, tables, lut_arr.astype(np.int32), mode=mode, paint=bool(paint),
            background=int(background), threshold=float(threshold), **opts)
    return out


def resample_argmax(logits, out_shape, mapping: Mapping, **kw) -> torch.Tensor:
    """Mechanism-level name: trilinear resample of the logits + argmax, fused."""
    return to_labels(logits, out_shape, mapping, mode="argmax", **kw)


def resample_paint(logits, out_shape, mapping: Mapping, labels, *, threshold: float = 0.0, **kw) -> torch.Tensor:
    """Mechanism-level name for region (sigmoid-head) models: trilinear resample
    of each region's logit, threshold, paint ``labels[k]`` in channel order."""
    return to_labels(logits, out_shape, mapping, mode="regions", lut=labels, threshold=threshold, **kw)


def available_backends() -> list[str]:
    return backends.available_backends()
