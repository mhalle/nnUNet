"""Host-side per-axis index / weight tables.

This is the one place output coordinates are computed. Every backend consumes
the same tables, so backends can only differ in the fixed-order lerp chain and
the final decision - never in *where* they sample.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nnunetv2.inference.restore.mapping import Mapping

INTERP = ("linear", "nearest")
OUTSIDE = ("background", "clamp")


@dataclass(frozen=True)
class AxisTable:
    """For each output index along one axis: the two source indices to blend
    and the weight of the second. ``i0 == -1`` marks an output index that
    falls outside the source extent (only with ``outside="background"``)."""

    i0: np.ndarray   # int32
    i1: np.ndarray   # int32
    f: np.ndarray    # float32, weight of i1 (0 for nearest) - what the kernels use
    f64: np.ndarray  # float64, the same weight before rounding - what the reference uses

    @property
    def n(self) -> int:
        return int(self.i0.shape[0])


def normalize_interp(interp) -> tuple[str, str, str]:
    if isinstance(interp, str):
        interp = (interp, interp, interp)
    t = tuple(str(x) for x in interp)
    if len(t) != 3 or any(x not in INTERP for x in t):
        raise ValueError(f"interp must be 'linear' / 'nearest' or a (Z, Y, X) tuple of them; got {interp!r}")
    return t


def axis_table(n_out: int, n_src: int, a: float, b: float, *, interp: str = "linear",
               outside: str = "background", coord_dtype=np.float64) -> AxisTable:
    """Tables for one axis of ``x_src = a * j + b``, ``j = 0 .. n_out-1``.

    ``coord_dtype=np.float64`` evaluates the coordinate the way scipy / skimage
    do; ``np.float32`` evaluates it in single precision, for parity with kernels
    that compute their coordinates in-kernel. Decisions (validity, floor,
    rounding) are then made in float64, which is exact for either input.

    Linear: ``i0 = floor(c)``, ``f = c - i0``, ``i1 = min(i0 + 1, n_src - 1)``.
    Nearest: ``i0 = i1 = floor(c + 0.5)`` (scipy's order-0 rule), ``f = 0``.
    A coordinate is inside the source if it lies within the voxel volumes,
    ``-0.5 <= c <= n_src - 0.5``; inside, it is clamped to ``[0, n_src - 1]``
    (edge extension, as skimage ``mode="edge"`` / scipy ``mode="nearest"``).
    Outside: ``-1`` sentinel (``outside="background"``) or clamped anyway
    (``outside="clamp"``).
    """
    if interp not in INTERP:
        raise ValueError(f"interp must be one of {INTERP}; got {interp!r}")
    if outside not in OUTSIDE:
        raise ValueError(f"outside must be one of {OUTSIDE}; got {outside!r}")
    dt = np.dtype(coord_dtype)
    if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"coord_dtype must be float32 or float64; got {coord_dtype!r}")
    n_out, n_src = int(n_out), int(n_src)
    if n_out < 1 or n_src < 1:
        raise ValueError("n_out and n_src must be >= 1")
    j = np.arange(n_out, dtype=dt)
    c = (j * dt.type(a) + dt.type(b)).astype(np.float64)
    valid = (c >= -0.5) & (c <= n_src - 0.5)
    if outside == "clamp":
        valid = np.ones_like(valid)
    c = np.clip(c, 0.0, float(n_src - 1))
    if interp == "linear":
        i0 = np.floor(c)
        f = c - i0
        i1 = np.minimum(i0 + 1, n_src - 1)
    else:
        i0 = np.minimum(np.floor(c + 0.5), n_src - 1)
        i1 = i0
        f = np.zeros_like(c)
    i0 = i0.astype(np.int32)
    i0[~valid] = -1
    return AxisTable(i0, i1.astype(np.int32), f.astype(np.float32), f.astype(np.float64))


def build_tables(out_shape, src_shape, mapping: Mapping, *, interp="linear", outside: str = "background",
                 coord_dtype=np.float64) -> tuple[AxisTable, AxisTable, AxisTable]:
    """(Z, Y, X) tables for ``mapping`` from an ``out_shape`` grid into a ``src_shape`` grid."""
    interp3 = normalize_interp(interp)
    out_shape = tuple(int(x) for x in out_shape)
    src_shape = tuple(int(x) for x in src_shape)
    if len(out_shape) != 3 or len(src_shape) != 3:
        raise ValueError("out_shape and src_shape must be (Z, Y, X)")
    return tuple(
        axis_table(out_shape[ax], src_shape[ax], mapping.a[ax], mapping.b[ax],
                   interp=interp3[ax], outside=outside, coord_dtype=coord_dtype)
        for ax in range(3)
    )
