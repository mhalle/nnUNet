"""Portable torch backend: index_select / lerp per output z-plane, with a
cache of x/y-interpolated source planes. Runs on any device; this is the
reference the fused kernels are checked against."""
from __future__ import annotations

import numpy as np
import torch

from ..tables import AxisTable


def available() -> bool:
    return True


def _axis(t: AxisTable, device):
    i0 = torch.from_numpy(np.where(t.i0 < 0, 0, t.i0).astype(np.int64)).to(device)
    i1 = torch.from_numpy(t.i1.astype(np.int64)).to(device)
    f = torch.from_numpy(np.ascontiguousarray(t.f, dtype=np.float32)).to(device)
    return i0, i1, f


@torch.no_grad()
def run(logits: torch.Tensor, out: torch.Tensor, tables, lut, *, mode: str, paint: bool,
        background: int, threshold: float) -> None:
    device = logits.device
    K, Zt, Yt, Xt = logits.shape
    Za, Ya, Xa = out.shape
    tz, ty, tx = tables
    y0, y1, yf = _axis(ty, device)
    x0, x1, xf = _axis(tx, device)
    valid_plane = torch.from_numpy((ty.i0 >= 0)[:, None] & (tx.i0 >= 0)[None, :]).to(device)
    lut_t = torch.from_numpy(np.ascontiguousarray(lut, dtype=np.int32)).to(device)
    bg = torch.full((Ya, Xa), int(background), dtype=torch.int32, device=device)
    wx0 = 1.0 - xf
    wy0 = (1.0 - yf)[:, None]
    wy1 = yf[:, None]
    cache: dict[int, torch.Tensor] = {}

    def plane(zi: int) -> torch.Tensor:
        t = cache.get(zi)
        if t is None:
            p = logits[:, zi].float()                                            # (K, Yt, Xt)
            px = p.index_select(2, x0) * wx0 + p.index_select(2, x1) * xf        # (K, Yt, Xa)
            t = px.index_select(1, y0) * wy0 + px.index_select(1, y1) * wy1      # (K, Ya, Xa)
            cache[zi] = t
        return t

    z0s, z1s, zfs = tz.i0.tolist(), tz.i1.tolist(), tz.f.tolist()
    for z in range(Za):
        z0 = z0s[z]
        if z0 < 0:
            if not paint:
                out[z].fill_(int(background))
            continue
        z1, w = z1s[z], zfs[z]
        a = plane(z0)
        v = a if (z1 == z0 or w == 0.0) else a * (1.0 - w) + plane(z1) * w
        if mode == "argmax":
            best = v.argmax(0)                       # first maximal channel
            lab = lut_t[best]
            hit = best != 0
        else:
            lab = bg.clone()
            hit = torch.zeros((Ya, Xa), dtype=torch.bool, device=device)
            for k in range(K):
                m = v[k] > threshold
                lab = torch.where(m, lut_t[k], lab)
                hit |= m
        if paint:
            out[z] = torch.where(valid_plane & hit, lab.to(out.dtype), out[z])
        else:
            out[z] = torch.where(valid_plane, lab, bg).to(out.dtype)
        for dead in [k for k in cache if k < z0]:   # z0 is monotonic (a >= 0): older planes are finished
            del cache[dead]
