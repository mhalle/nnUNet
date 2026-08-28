"""Two-step export against the fused one, on the same logits.

The point of the fused pass is the intermediate it does not build: the two-step form
materializes (num_segmentation_heads, *new_shape) floats, which is 34 GB for a 118-head
model restored to a 418 M voxel grid. Run with a head count and a target shape to see both
the time and that allocation.

    python -m nnunetv2.tests.benchmark_logits_to_segmentation --heads 118 --device cuda
"""
import argparse
import time
from functools import partial

import numpy as np
import torch

from nnunetv2.inference.logits_to_segmentation import fused_resample_and_convert, resample_and_convert
from nnunetv2.inference.restore import available_backends
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
from nnunetv2.utilities.label_handling.label_handling import LabelManager


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--heads', type=int, default=32)
    p.add_argument('--src', type=int, nargs=3, default=[112, 101, 122])
    p.add_argument('--target', type=int, nargs=3, default=[224, 202, 244])
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--skip-two-step', action='store_true',
                   help='the two-step path allocates heads x target floats; skip it when that '
                        'does not fit')
    args = p.parse_args()

    src, target = tuple(args.src), tuple(args.target)
    label_manager = LabelManager({'background': 0, **{f'c{i}': i for i in range(1, args.heads)}}, None)
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, (args.heads, *src)).astype(np.float32)
    prob_fn = partial(resample_data_or_seg_to_shape, order=1, order_z=0, force_separate_z=None)
    spacing = [1.0, 1.0, 1.0]

    print(f'  {args.heads} heads, {src} -> {target} ({np.prod(target) / 1e6:.0f} Mvox)')
    print(f'  logits {logits.nbytes / 1e9:.2f} GB; the two-step intermediate would be '
          f'{args.heads * np.prod(target) * 4 / 1e9:.1f} GB')
    print(f'  backends available: {available_backends()}')

    def bench(fn, label, **kw):
        fn(logits, target, spacing, spacing, label_manager=label_manager,
           resampling_fn_probabilities=prob_fn, **kw)     # warm-up, discarded
        best = float('inf')
        for _ in range(args.reps):
            t = time.perf_counter()
            out = fn(logits, target, spacing, spacing, label_manager=label_manager,
                     resampling_fn_probabilities=prob_fn, **kw)
            if args.device in ('cuda', 'mps'):
                getattr(torch, args.device).synchronize()
            best = min(best, time.perf_counter() - t)
        print(f'  {label:24} {best:7.2f} s')
        return np.asarray(out)

    fused = bench(fused_resample_and_convert, 'fused', device=args.device)
    if not args.skip_two_step:
        two_step = bench(resample_and_convert, 'two-step (default)')
        differing = int((np.asarray(two_step) != fused).sum())
        print(f'  agreement: {differing} of {fused.size} voxels differ')


if __name__ == '__main__':
    main()
