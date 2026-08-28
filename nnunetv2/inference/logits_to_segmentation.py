"""How predicted logits become a segmentation.

nnU-Net does this in two steps: resample the logits to the target shape, then decide -
argmax, or a per-region threshold. That is exact, but it materializes an intermediate of
shape ``(num_segmentation_heads, *new_shape)``. For a few classes that is unremarkable;
for a 118-head model restored to a 418 M voxel grid it is 34 GB of float32, which is why
memory-lighter approximations exist downstream.

Fusing the two steps removes the intermediate: the decision at each output voxel needs
only the eight source corners around it, so the resampled probability volume never has to
exist. What is built instead is the output label volume, one byte per voxel.

Which function is used is named by the plans key ``logits_to_segmentation_fn``, the way
the resampling functions are named. It defaults to ``resample_and_convert``, which is the
existing two-step behavior, so plans that do not mention it are unaffected.
"""
from typing import Union

import numpy as np
import torch

from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.preprocessing.resampling.default_resampling import determine_do_sep_z_and_axis
from nnunetv2.utilities.label_handling.label_handling import LabelManager


def resample_and_convert(predicted_logits: Union[torch.Tensor, np.ndarray],
                         new_shape,
                         current_spacing,
                         new_spacing,
                         *,
                         label_manager: LabelManager,
                         resampling_fn_probabilities,
                         **kwargs) -> Union[torch.Tensor, np.ndarray]:
    """Resample the logits, then decide. nnU-Net's behavior, and the default."""
    predicted_logits = resampling_fn_probabilities(predicted_logits, new_shape,
                                                   current_spacing, new_spacing)
    return label_manager.convert_logits_to_segmentation(predicted_logits)


def _select_device(predicted_logits, device) -> Union[torch.device, None]:
    """Where to run the fused pass, or None to decline.

    The predictor moves logits to the CPU before export, so "use the device they are on"
    would almost never fuse; a GPU is claimed when one exists. On a CPU-only machine this
    declines instead, because the PyTorch fallback kernel is slower than the two-step path
    it would replace - it is there for correctness, not for speed. An explicit ``device``
    from the plans always wins, ``"cpu"`` included.

    Export can run in several worker processes, each of which would then hold a context and
    a working set on the GPU. That is a reason to keep ``-nps`` small when fusing, not a
    reason to leave seconds of CPU work on the table; see the documentation.
    """
    if device is not None:
        return torch.device(device)
    if isinstance(predicted_logits, torch.Tensor) and predicted_logits.device.type in ("cuda", "mps"):
        return predicted_logits.device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def _decline_reason(predicted_logits, label_manager, order, order_z, do_separate_z, dev, backend) -> Union[str, None]:
    """Why the fused pass cannot be used here, or None if it can."""
    from nnunetv2.inference.restore import backends as restore_backends

    if order != 1:
        return f"the fused pass is trilinear; plans ask for order={order}"
    if do_separate_z and order_z not in (0, 1):
        return f"separate-z with order_z={order_z} is not expressible per axis"
    if dev is None:
        return "no device was selected (see _select_device)"
    name, module = restore_backends.select(backend, dev)
    if not module.available():
        return f"backend {name!r} is unavailable on {dev}"
    if label_manager.has_regions and label_manager.regions_class_order is None:
        return "region model without regions_class_order"
    n_heads = int(predicted_logits.shape[0])
    if n_heads != label_manager.num_segmentation_heads:
        return f"expected {label_manager.num_segmentation_heads} heads, got {n_heads}"
    return None


def fused_resample_and_convert(predicted_logits: Union[torch.Tensor, np.ndarray],
                               new_shape,
                               current_spacing,
                               new_spacing,
                               *,
                               label_manager: LabelManager,
                               resampling_fn_probabilities,
                               order: int = 1,
                               order_z: int = 0,
                               force_separate_z: Union[bool, None] = None,
                               separate_z_anisotropy_threshold: float = ANISO_THRESHOLD,
                               device: Union[str, torch.device, None] = None,
                               backend: str = "auto",
                               verbose: bool = False,
                               **kwargs) -> Union[torch.Tensor, np.ndarray]:
    """Resample and decide in one pass, without building the resampled probability volume.

    Produces what ``resample_and_convert`` produces - voxel-center sampling, ``argmax``
    over the heads, or for region models each head thresholded at a probability of 0.5 and
    painted in ``regions_class_order`` - and falls back to it whenever the request cannot
    be expressed this way (a spline order other than 1, an unavailable device, and so on).
    Set ``verbose`` to see the reason.
    """
    from nnunetv2.inference.restore import Mapping, to_labels

    new_shape = tuple(int(i) for i in new_shape)
    do_separate_z, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing,
                                                      separate_z_anisotropy_threshold)
    dev = _select_device(predicted_logits, device)
    reason = _decline_reason(predicted_logits, label_manager, order, order_z, do_separate_z, dev, backend)
    if reason is not None:
        if verbose:
            print(f"fused_resample_and_convert: falling back ({reason})")
        return resample_and_convert(predicted_logits, new_shape, current_spacing, new_spacing,
                                    label_manager=label_manager,
                                    resampling_fn_probabilities=resampling_fn_probabilities)

    input_was_numpy = isinstance(predicted_logits, np.ndarray)
    logits = torch.from_numpy(np.ascontiguousarray(predicted_logits)) if input_was_numpy else predicted_logits
    original_device = torch.device("cpu") if input_was_numpy else logits.device
    if not logits.dtype.is_floating_point:
        logits = logits.float()
    logits = logits.to(dev, non_blocking=True)

    src_shape = tuple(int(s) for s in logits.shape[1:])
    # nnU-Net's resamplers sample at voxel centers; separate-z replaces the interpolation
    # along the low-resolution axis with nearest neighbor, which is a per-axis choice here.
    interp = "linear"
    if do_separate_z and order_z == 0 and axis is not None:
        per_axis = ["linear", "linear", "linear"]
        for a in (axis if isinstance(axis, (list, tuple, np.ndarray)) else [axis]):
            per_axis[int(a)] = "nearest"
        interp = tuple(per_axis)

    if label_manager.has_regions:
        # sigmoid(x) > 0.5 is x > 0, so the region rule is a threshold on the logit itself
        segmentation = to_labels(logits, new_shape, Mapping.center(new_shape, src_shape),
                                 interp=interp, mode="regions", threshold=0.0,
                                 lut=list(label_manager.regions_class_order), backend=backend)
    else:
        segmentation = to_labels(logits, new_shape, Mapping.center(new_shape, src_shape),
                                 interp=interp, mode="argmax", backend=backend)

    if input_was_numpy:
        return segmentation.cpu().numpy()
    return segmentation.to(original_device)
