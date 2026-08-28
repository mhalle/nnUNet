"""Per-axis affine maps between index spaces."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Vec3 = tuple[float, float, float]


def _vec3(v, name: str) -> Vec3:
    if np.isscalar(v):
        v = (v, v, v)
    t = tuple(float(x) for x in v)
    if len(t) != 3:
        raise ValueError(f"{name} must have 3 entries (Z, Y, X); got {v!r}")
    return t


@dataclass(frozen=True)
class Mapping:
    """``x_to = a * x_from + b`` per axis (Z, Y, X).

    Maps integer voxel indices of a *from* grid to continuous voxel
    coordinates of a *to* grid. Separable: no rotation, no shear - flips and
    permutations belong to the caller's frame, not here. ``a >= 0``.

    Compose with ``>>``: ``m1 >> m2`` applies ``m1`` first.
    """

    a: Vec3
    b: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, "a", _vec3(self.a, "a"))
        object.__setattr__(self, "b", _vec3(self.b, "b"))
        if any(x < 0 for x in self.a):
            raise ValueError(f"a must be >= 0 on every axis (flips belong to the frame); got {self.a}")

    def apply(self, x_from) -> np.ndarray:
        """Apply to coordinates (..., 3)."""
        return np.asarray(x_from, dtype=np.float64) * np.asarray(self.a) + np.asarray(self.b)

    def then(self, other: "Mapping") -> "Mapping":
        """``self`` first, then ``other``."""
        a1, b1 = np.asarray(self.a), np.asarray(self.b)
        a2, b2 = np.asarray(other.a), np.asarray(other.b)
        return Mapping(tuple(a2 * a1), tuple(a2 * b1 + b2))

    __rshift__ = then

    def inverse(self) -> "Mapping":
        if any(x == 0 for x in self.a):
            raise ValueError("mapping with a zero factor has no inverse")
        a = 1.0 / np.asarray(self.a)
        return Mapping(tuple(a), tuple(-np.asarray(self.b) * a))

    # -- constructors -----------------------------------------------------
    @classmethod
    def identity(cls) -> "Mapping":
        return cls((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))

    @classmethod
    def center(cls, shape_from, shape_to) -> "Mapping":
        """Voxel-center (half-pixel) rule: ``x_to = (x_from + 0.5) * n_to / n_from - 0.5``.

        The rule of ``skimage.transform.resize``, ITK, ``F.interpolate(align_corners=False)``
        and therefore nnU-Net's own resampler. ``Mapping.center(n_src, n_model)`` maps
        source-image indices to model-grid coordinates for a model grid produced
        by that resampler - i.e. it inverts the forward resample exactly.
        """
        n_from = np.asarray(shape_from, dtype=np.float64)
        n_to = np.asarray(shape_to, dtype=np.float64)
        a = n_to / n_from
        return cls(tuple(a), tuple(0.5 * a - 0.5))

    @classmethod
    def corner(cls, shape_from, shape_to) -> "Mapping":
        """Voxel-corner rule: ``x_to = x_from * (n_to - 1) / (n_from - 1)``.

        The rule of ``scipy.ndimage.zoom(grid_mode=False)`` and
        ``F.interpolate(align_corners=True)`` - what TotalSegmentator's
        ``change_spacing`` uses on both legs. An axis with a single sample maps
        to coordinate 0.
        """
        n_from = np.asarray(shape_from, dtype=np.float64)
        n_to = np.asarray(shape_to, dtype=np.float64)
        a = np.where(n_from > 1, (n_to - 1) / np.maximum(n_from - 1, 1), 0.0)
        return cls(tuple(a), (0.0, 0.0, 0.0))

    @classmethod
    def spacing(cls, spacing_from, spacing_to, shift=(0.0, 0.0, 0.0)) -> "Mapping":
        """Origin-aligned, spacing-exact rule: ``x_to = x_from * s_from / s_to + shift``.

        Voxel (0, 0, 0) of both grids coincide (``shift = 0``).
        """
        s_from = np.asarray(_vec3(spacing_from, "spacing_from"))
        s_to = np.asarray(_vec3(spacing_to, "spacing_to"))
        return cls(tuple(s_from / s_to), _vec3(shift, "shift"))


    def __repr__(self) -> str:
        a = ", ".join(f"{x:g}" for x in self.a)
        b = ", ".join(f"{x:g}" for x in self.b)
        return f"Mapping(a=({a}), b=({b}))"
