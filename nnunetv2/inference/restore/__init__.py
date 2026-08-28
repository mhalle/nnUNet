"""Fused logit resampling and label decision.

``to_labels`` interpolates the K predicted logits at every voxel of a target grid and
decides there and then - argmax, or a per-region threshold - so the resampled
``(K, *new_shape)`` volume is never built. That volume is the reason the two-step form
becomes impractical for many-class models: 118 heads restored to a 418 M voxel grid is
34 GB of float32, where the fused pass needs the output label volume and little else.

The kernels are pure PyTorch plus, where available, a Metal shader compiled at runtime
(``torch.mps.compile_shader``, PyTorch >= 2.7) or a Triton kernel on CUDA. Neither needs a
build step and both are guarded by ``available()``, so this package imports and runs
anywhere PyTorch does.
"""
from nnunetv2.inference.restore.core import (  # noqa: F401
    to_labels,
    resample_argmax,
    resample_paint,
    available_backends,
)
from nnunetv2.inference.restore.mapping import Mapping  # noqa: F401
