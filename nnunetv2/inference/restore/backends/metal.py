"""Apple GPU backend: the fused trilinear + argmax / paint kernel, compiled
from Metal source with ``torch.mps.compile_shader`` (PyTorch >= 2.7).

One thread per output voxel, an 8-corner gather per channel and a running
decision over K, driven by host-built per-axis tables and a label LUT, with
paint mode, region mode, fp16 logits, uint8 / uint16 output and z-slab
launches.
"""
from __future__ import annotations

import numpy as np
import torch

_HEADER = r"""
#include <metal_stdlib>
using namespace metal;
{PRAGMA}

template <typename T>
inline float lg_sample(const device T* logits, long off, thread const int* b,
                       float wx0, float xf, float wy0, float yf, float wz0, float zf)
{
    float c00 = (float)logits[off + b[0]] * wx0 + (float)logits[off + b[1]] * xf;
    float c01 = (float)logits[off + b[2]] * wx0 + (float)logits[off + b[3]] * xf;
    float c10 = (float)logits[off + b[4]] * wx0 + (float)logits[off + b[5]] * xf;
    float c11 = (float)logits[off + b[6]] * wx0 + (float)logits[off + b[7]] * xf;
    float c0 = c00 * wy0 + c01 * yf;
    float c1 = c10 * wy0 + c11 * yf;
    return c0 * wz0 + c1 * zf;
}
"""

_KERNEL = r"""
kernel void {NAME}(
    device const {LOGIT_T}* logits  [[buffer(0)]],
    device const int*       iparams [[buffer(1)]],
    device const float*     fparams [[buffer(2)]],
    device const int*       tz0     [[buffer(3)]],
    device const int*       tz1     [[buffer(4)]],
    device const float*     tzf     [[buffer(5)]],
    device const int*       ty0     [[buffer(6)]],
    device const int*       ty1     [[buffer(7)]],
    device const float*     tyf     [[buffer(8)]],
    device const int*       tx0     [[buffer(9)]],
    device const int*       tx1     [[buffer(10)]],
    device const float*     txf     [[buffer(11)]],
    device const int*       lut     [[buffer(12)]],
    device {OUT_T}*         out     [[buffer(13)]],
    uint elem [[thread_position_in_grid]])
{
    const int n_slab = iparams[8];
    if ((int)elem >= n_slab) return;
    const int K = iparams[0], Zt = iparams[1], Yt = iparams[2], Xt = iparams[3];
    const int Ya = iparams[5], Xa = iparams[6];
    const int z_offset = iparams[7];
    const int mode = iparams[9], paint = iparams[10], background = iparams[11];
    const float threshold = fparams[0];

    const uint plane_out = (uint)Xa * (uint)Ya;
    const int x = (int)(elem % (uint)Xa);
    const int y = (int)((elem / (uint)Xa) % (uint)Ya);
    const int z = z_offset + (int)(elem / plane_out);
    const long oidx = (long)z * (long)plane_out + (long)y * (long)Xa + (long)x;

    const int z0 = tz0[z], y0 = ty0[y], x0 = tx0[x];
    if (z0 < 0 || y0 < 0 || x0 < 0) {
        if (!paint) out[oidx] = ({OUT_T})background;
        return;
    }
    const int z1 = tz1[z], y1 = ty1[y], x1 = tx1[x];
    const float zf = tzf[z], yf = tyf[y], xf = txf[x];
    const float wx0 = 1.0f - xf, wy0 = 1.0f - yf, wz0 = 1.0f - zf;
    // 32-bit offsets within one channel (per-channel volume < 2^31, checked on the host):
    // 64-bit offsets here cost 2x on Apple GPUs. Only the channel stride is 64-bit.
    const int plane = Yt * Xt;
    int b[8];
    b[0] = z0 * plane + y0 * Xt + x0;
    b[1] = z0 * plane + y0 * Xt + x1;
    b[2] = z0 * plane + y1 * Xt + x0;
    b[3] = z0 * plane + y1 * Xt + x1;
    b[4] = z1 * plane + y0 * Xt + x0;
    b[5] = z1 * plane + y0 * Xt + x1;
    b[6] = z1 * plane + y1 * Xt + x0;
    b[7] = z1 * plane + y1 * Xt + x1;
    const long chan_stride = (long)Zt * (long)plane;

    int label = background;
    if (mode == 0) {
        float best = -INFINITY; int best_k = 0;
        for (int k = 0; k < K; k++) {
            float v = lg_sample<{LOGIT_T}>(logits, (long)k * chan_stride, b, wx0, xf, wy0, yf, wz0, zf);
            if (v > best) { best = v; best_k = k; }
        }
        if (paint && best_k == 0) return;
        label = lut[best_k];
    } else {
        int hit = 0;
        for (int k = 0; k < K; k++) {
            float v = lg_sample<{LOGIT_T}>(logits, (long)k * chan_stride, b, wx0, xf, wy0, yf, wz0, zf);
            if (v > threshold) { label = lut[k]; hit = 1; }
        }
        if (paint && !hit) return;
    }
    out[oidx] = ({OUT_T})label;
}
"""

_VARIANTS = [("float", "uchar"), ("float", "ushort"), ("half", "uchar"), ("half", "ushort")]
_LIB = None
_FP_CONTRACT: str | None = None


def available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                and hasattr(torch.mps, "compile_shader"))


def source(fp_contract_off: bool = True) -> str:
    pragma = "#pragma clang fp contract(off)" if fp_contract_off else ""
    parts = [_HEADER.replace("{PRAGMA}", pragma)]
    for logit_t, out_t in _VARIANTS:
        parts.append(_KERNEL.replace("{NAME}", f"lg_{logit_t}_{out_t}")
                     .replace("{LOGIT_T}", logit_t).replace("{OUT_T}", out_t))
    return "\n".join(parts)


def library():
    """Compile once. Tries to switch fused-multiply-add contraction off so the
    lerp chain rounds exactly like the torch backend (measured cost ~4 %);
    falls back to the compiler default if the pragma is rejected."""
    global _LIB, _FP_CONTRACT
    if _LIB is None:
        if not available():
            raise RuntimeError("nnunetv2.inference.restore.backends.metal: needs an MPS device and torch.mps.compile_shader (torch >= 2.7)")
        try:
            _LIB = torch.mps.compile_shader(source(True))
            _FP_CONTRACT = "off"
        except Exception:
            _LIB = torch.mps.compile_shader(source(False))
            _FP_CONTRACT = "default"
    return _LIB


def fp_contract() -> str | None:
    library()
    return _FP_CONTRACT


@torch.no_grad()
def run(logits: torch.Tensor, out: torch.Tensor, tables, lut, *, mode: str, paint: bool, background: int,
        threshold: float, slab_voxels: int = 1 << 26, group_size: int = 256) -> None:
    if logits.device.type != "mps" or out.device.type != "mps":
        raise ValueError("nnunetv2.inference.restore.backends.metal: logits and out must be on the 'mps' device")
    if logits.dtype == torch.float32:
        logit_t = "float"
    elif logits.dtype == torch.float16:
        logit_t = "half"
    else:
        raise TypeError(f"nnunetv2.inference.restore.backends.metal: logits must be float32 or float16; got {logits.dtype} (cast first)")
    if out.dtype == torch.uint8:
        out_t = "uchar"
    elif out.dtype == torch.uint16:
        out_t = "ushort"
    else:
        raise TypeError(f"nnunetv2.inference.restore.backends.metal: out must be uint8 or uint16; got {out.dtype}")
    if not out.is_contiguous():
        raise ValueError("nnunetv2.inference.restore.backends.metal: out must be contiguous")
    logits = logits.contiguous()
    dev = logits.device
    K, Zt, Yt, Xt = (int(s) for s in logits.shape)
    if Zt * Yt * Xt >= 2 ** 31:
        raise ValueError("nnunetv2.inference.restore.backends.metal: per-channel volume must be < 2^31 voxels (32-bit in-channel offsets)")
    Za, Ya, Xa = (int(s) for s in out.shape)
    tz, ty, tx = tables

    def to_i(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.int32)).to(dev)

    def to_f(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev)

    bufs_tables = [to_i(tz.i0), to_i(tz.i1), to_f(tz.f), to_i(ty.i0), to_i(ty.i1), to_f(ty.f),
                   to_i(tx.i0), to_i(tx.i1), to_f(tx.f)]
    lut_t = to_i(lut)
    fparams = torch.tensor([float(threshold)], dtype=torch.float32, device=dev)
    kernel = getattr(library(), f"lg_{logit_t}_{out_t}")
    plane = Ya * Xa
    planes_per_launch = max(1, int(slab_voxels) // plane)
    mode_i = 0 if mode == "argmax" else 1
    for z_off in range(0, Za, planes_per_launch):
        nz = min(planes_per_launch, Za - z_off)
        n_slab = nz * plane
        iparams = torch.tensor([K, Zt, Yt, Xt, Za, Ya, Xa, z_off, n_slab, mode_i, int(bool(paint)), int(background)],
                               dtype=torch.int32, device=dev)
        kernel(logits, iparams, fparams, *bufs_tables, lut_t, out, threads=n_slab, group_size=group_size)
