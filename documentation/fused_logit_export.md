# Fused logit export

## What this is

Turning predicted logits into a segmentation is two steps: resample the logits to the
target shape, then decide - `argmax` over the heads, or, for region models, threshold each
head and paint in `regions_class_order`. nnU-Net does them in that order, which is exact
and simple, but it means an intermediate of shape `(num_segmentation_heads, *new_shape)`
exists between them.

For a handful of classes that is unremarkable. For a 118-head model restored to a
768 x 768 x 709 grid it is **34 GB of float32**, and that single number is why
memory-lighter approximations of this step exist downstream.

The decision at an output voxel needs only the eight source corners around it. So the two
steps can be fused: interpolate and decide in one pass, and the resampled probability
volume never has to exist. What gets built is the output label volume, one byte per voxel.

## Using it

Two plans keys, both optional:

```json
"logits_to_segmentation_fn": "fused_resample_and_convert",
"logits_to_segmentation_fn_kwargs": {"order": 1, "order_z": 0, "force_separate_z": null}
```

Absent - as they are in every plans file written before they existed - the default is
`resample_and_convert`, the two-step behavior, unchanged.

To have a planner write them, use `nnUNetPlannerFusedExport` (or
`nnUNetPlannerResEncLFusedExport`):

```bash
nnUNetv2_plan_experiment -d DATASET -pl nnUNetPlannerFusedExport
```

## What it produces

The same segmentation as the two-step path. Sampling is at voxel centers, as nnU-Net's own
resamplers do; `argmax` takes the first maximal head; region models threshold each head at
a probability of 0.5 (which is a logit greater than zero) and paint in
`regions_class_order`, later region winning. nnU-Net's separate-z policy is followed too:
when the spacings are anisotropic past `ANISO_THRESHOLD`, the low-resolution axis is
resampled with nearest neighbor, decided by `determine_do_sep_z_and_axis` itself.

Where a request cannot be expressed this way the function falls back to the two-step path
rather than approximating. That happens for a spline order other than 1, for `order_z`
values other than 0 or 1, when probabilities are being exported (they need the resampled
volume by definition), and when no device is available. Pass `verbose=True` in the kwargs
to print the reason.

## Devices

The fused pass runs where the logits are. It uses a Metal shader compiled at runtime on
Apple GPUs (`torch.mps.compile_shader`, PyTorch >= 2.7), a Triton kernel on CUDA, and a
plain PyTorch fallback everywhere else. None of these needs a build step, and each is
guarded by an `available()` check, so the module imports and runs wherever PyTorch does.

A GPU is claimed when one is available. The predictor moves logits to the CPU before
export, so waiting for them to arrive on a device would mean never fusing at all. On a
CPU-only machine this declines and the two-step path runs instead - the PyTorch fallback
kernel is there for correctness, not speed, and is slower than what it would replace. An
explicit `"device"` in the kwargs always wins, `"cpu"` included.

Export can run in several worker processes (`-nps`), each of which will then hold a GPU
context and a working set of its own. Keep `-nps` small when fusing. Worth noting that the
reason it defaults high is that this step is slow on the CPU, which is the thing fusing
fixes: a couple of workers is plenty once the work takes tens of milliseconds.

## Measured

`benchmark_logits_to_segmentation`, A10, a 112 x 101 x 122 model grid restored to
224 x 202 x 244 (11 Mvox), logits drawn from `normal(0, 3)` - unstructured noise, which is
the worst case for the tie-breaking and so a lower bound on agreement:

| heads | two-step | fused | intermediate the two-step builds | differing voxels |
|------:|---------:|------:|---------------------------------:|-----------------:|
|     3 |   1.70 s |  0.00 s |                           0.1 GB |    0 of 11040512 |
|     6 |   3.29 s |  0.01 s |                           0.3 GB |    0 of 11040512 |
|    16 |   8.63 s |  0.01 s |                           0.7 GB |    1 of 11040512 |
|   118 |  62.08 s |  0.08 s |                           5.2 GB |    3 of 11040512 |


The two-step cost is linear in the head count because each head is resampled in full before
anything is decided; the fused pass does not have that term. Both the time and the
intermediate also scale with the output volume, so the same 118-head model restored to a
418 M voxel grid would build 34 GB.

The same benchmark on an M2 (Metal) gives 0.01, 0.01, 0.01 and 0.08 s for those four head
counts - the same as the A10 - because at this size the pass is bounded by launch overhead
and memory traffic rather than by arithmetic. Agreement there is 0, 0, 1 and 2 voxels; the
one-voxel difference between the two backends is float32 tie-breaking on noise.

## Checking it

```bash
python -m nnunetv2.tests.benchmark_logits_to_segmentation --heads 118 --device cuda
python -m pytest nnunetv2/tests/test_logits_to_segmentation.py
```

The tests assert the property that makes the fused path substitutable at all: that it
returns what the two-step path returns, for `argmax`, for regions, for separate-z, and on
whatever GPU backend the machine has.
