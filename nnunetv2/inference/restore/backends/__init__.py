"""Backend registry. Every backend exposes ``available()`` and
``run(logits, out, tables, lut, *, mode, paint, background, threshold, **opts)``
writing labels into ``out`` in place."""
from __future__ import annotations

import torch

from nnunetv2.inference.restore.backends import metal, torch_gather, triton_gpu

BACKENDS = {"torch": torch_gather, "metal": metal, "triton": triton_gpu}


def select(name: str, device: torch.device):
    if name == "auto":
        if device.type == "mps" and metal.available():
            return "metal", metal
        if device.type == "cuda" and triton_gpu.available():
            return "triton", triton_gpu
        return "torch", torch_gather
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)} or 'auto'")
    if name == "metal" and device.type != "mps":
        raise ValueError("backend='metal' needs logits on the 'mps' device")
    if name == "triton":
        if device.type != "cuda":
            raise ValueError("backend='triton' needs logits on a 'cuda' device")
        if not triton_gpu.available():
            raise ValueError(f"backend='triton' unavailable: {triton_gpu.why_unavailable()}")
    return name, BACKENDS[name]


def available_backends() -> list[str]:
    return [n for n, m in BACKENDS.items() if m.available()]
