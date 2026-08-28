# GPU resampling (`resample_data_or_seg_to_shape_gpu`)

`nnunetv2/preprocessing/resampling/resample_gpu.py` provides nnU-Net's default resampler,
`resample_data_or_seg_to_shape`, as a torch implementation that runs on **MPS / CUDA / CPU**
and produces the same arrays. It is selected by name from `plans.json` like any other
resampling function.

## Why

| resampler | runs on | same results as the default? |
|---|---|---|
| `resample_data_or_seg_to_shape` (scipy / skimage, **default**) | CPU only | - |
| `resample_torch_fornnunet` (`F.interpolate`) | CPU / GPU (MPS needs torch >= 2.11) | no: trilinear, no spline prefilter (~60 HU max difference on CT) |
| **`resample_data_or_seg_to_shape_gpu` (this)** | **MPS / CUDA / CPU** | **yes** |

On a chest CT (768 x 768 x 709 voxels at 0.65 x 0.65 x 1.0 mm, resampled to 3 mm) the
scipy path takes 16.5 s on an M2 MacBook Air; this function takes 3.4 s on its GPU with
voxel-identical downstream segmentations (see Benchmark). `resample_torch_fornnunet` is faster
still but is a different resampler (no spline prefilter), so it changes the model's inputs.

## What "same results" means

Verified in `nnunetv2/tests/test_resample_gpu.py` directly against
`resample_data_or_seg_to_shape` (and against `scipy.ndimage.zoom`, `skimage.transform.resize`
and `torch.nn.functional.interpolate` where those are the references):

| aspect | status |
|---|---|
| image data, orders 0 / 1 / 3, up- and downsampling | equal to ~1e-12 on CPU (float64); ~1e-3 on HU-scale values on MPS/CUDA (float32) |
| skimage's per-call clip of the output to the input range (`resize(clip=True)`) | replicated, per channel |
| labels (`is_seg=True`): `resize_segmentation`'s rule - indicator resized, painted where `>= 0.5` in ascending label order; `order=0` nearest | bit-identical on CPU and GPU |
| separate-z policy (anisotropy > `ANISO_THRESHOLD`, or `force_separate_z`): in-plane with `order`, per-slice clip, low-resolution axis with `order_z`; label rules `>= 0.5` in-plane and `round(v) > 0.5` along the axis | replicated with upstream's own `determine_do_sep_z_and_axis`; equal to ~1e-12 / bit-identical |
| integer inputs | rounded half-away-from-zero on output, as scipy does |

How: each axis is a dense `(n_out x n_in)` matrix applied by matmul. The matrices are obtained
by **probing `scipy.ndimage.zoom` with an identity matrix** - the resampler is linear and
separable, so one probe per axis captures the spline prefilter, the boundary mode and the
coordinate map exactly, and nothing of scipy's boundary handling is re-implemented. Probes are
cached per `(n_in, n_out, order, mode)`. Multi-channel inputs (K-class probabilities at
export) are processed in channel chunks (`channel_chunk`) to bound GPU memory.

## Sampling conventions: `convention="center"` (default) vs `"corner"`

Two resamplers can agree on "cubic" and still disagree about *where* the output samples sit.

| `convention` | model | output sample `j` reads input coordinate | matches |
|---|---|---|---|
| `"center"` (default) | voxel-center: value at the center of a cell, cells tile the field of view | `(j + 0.5) * n_in/n_out - 0.5` (`align_corners=False`) | skimage `resize`, `F.interpolate`, ITK, **nnU-Net's default resampler** |
| `"corner"` | voxel-corner point grid: values are points at `i * spacing`, the span of the points is preserved | `j * (n_in - 1) / (n_out - 1)` (`align_corners=True`, scipy `grid_mode=False`) | `scipy.ndimage.zoom` with its defaults, i.e. pipelines that pre-resample with it (e.g. TotalSegmentator's `change_spacing`) |

The two grids differ by a factor `(n-1)/n` per axis (0.46 % for 768 -> 167 voxels, up to ~2
native voxels at the far edge). This is not cosmetic: because the label upsample snaps every
boundary to the model grid, resampling with the wrong convention moved mean Dice against the
reference pipeline from 0.995 to 0.888 on a chest CT. Match the convention of the pipeline
the model was trained with; for nnU-Net-native models that is the default.

## `anti_alias=True` (opt-in)

With `anti_alias=True` downsampling axes use a PIL-style anti-aliasing filter (Catmull-Rom
cubic scaled by the factor, half-pixel centers, border taps renormalized) and upsampling axes
are linear; the output is clipped to the input range. This equals
`F.interpolate(mode="bicubic"/"bilinear", antialias=True)` to float precision (tested).

It is **opt-in** because every model trained with the default resampler has learned the
statistics of non-anti-aliased inputs: on a chest CT with a 3 mm model, anti-aliased inputs
lowered the contrast of sub-centimeter structures by ~30 % and the model stopped detecting two
of them. Use it for models that were trained with it (set it in the plans so that
preprocessing and inference agree), not as an inference-time switch for existing models.

## Usage

New datasets - plan with the GPU resampler (same kwargs as the default planner):

```
nnUNetv2_plan_and_preprocess -d DATASET_ID -pl nnUNetPlanner_gpures
nnUNetv2_plan_and_preprocess -d DATASET_ID -pl nnUNetPlannerResEncL_gpures
```

Existing models - the plans choose the resampler, so edit the model's `plans.json` and replace
the three function names (`resampling_fn_data`, `resampling_fn_seg`, `resampling_fn_probabilities`)
with `resample_data_or_seg_to_shape_gpu`; the kwargs (`order`, `order_z`, `force_separate_z`,
`is_seg`) are the same and stay as they are.

Direct call (`data` is `(c, x, y, z)`):

```python
from nnunetv2.preprocessing.resampling.resample_gpu import resample_data_or_seg_to_shape_gpu

out = resample_data_or_seg_to_shape_gpu(data, new_shape, current_spacing, new_spacing,
                                        is_seg=False, order=3, order_z=0, force_separate_z=None,
                                        device="mps")          # == resample_data_or_seg_to_shape
seg = resample_data_or_seg_to_shape_gpu(seg, new_shape, current_spacing, new_spacing,
                                        is_seg=True, order=1, device="mps")
```

`device=None` picks CUDA, then MPS, then CPU. Numpy in gives numpy out (dtype preserved);
tensors come back on their original device.

## Tests and benchmark

```
python -m pytest nnunetv2/tests/test_resample_gpu.py
python nnunetv2/tests/benchmark_resample_gpu.py [path/to/image.nii.gz] [target_mm]
```

The benchmark times the scipy default, `resample_torch_fornnunet` and this function on the
given image (or a synthetic volume) and reports the agreement between them. Measured on an
M2 MacBook Air (16 GB, MPS) with TotalSegmentator's 3 mm model and the chest CT above:
forward resample 16.5 s (scipy) vs 3.4 s (this function, `convention="corner"` to match that
pipeline), whole fast-mode run 93.9 s vs 73.3 s, output voxel-identical (Dice 1.0 on all
111 labels). On CUDA the speed-up is larger; on CPU this function is a float64 matmul
implementation of the same operator and is not faster than scipy.

## Limitations

- float32 on MPS/CUDA: differences to the float64 reference of order 1e-3 on HU-scale data
  (1e-6 relative); labels are unaffected in the tests.
- K-channel probability export at the original resolution holds `channel_chunk` channels at
  full resolution on the device; lower `channel_chunk` for very large volumes.
- `mode` other than `"nearest"` is supported for `convention="corner"` (scipy semantics);
  the default convention always uses skimage's `"edge"` (= scipy `"nearest"`) as nnU-Net does.
