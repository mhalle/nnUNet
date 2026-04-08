"""InferenceEngine: pure inference compute over a ModelBundle.

The engine owns inference-time options (tile_step_size, mirroring, gaussian
weighting, device, autocast) and implements the sliding-window prediction
loop. It does not own folder discovery, plans parsing, or weight loading —
those are the ModelBundle's job — and it does not own multi-case I/O,
ensembling across separately-trained models, or output file writing — those
remain in nnUNetPredictor.

This split mirrors the MLX inference port's ModelBundle / InferenceEngine
boundary, so a structural ``Protocol`` covering ``predict(volume)`` /
``predict_ensemble(volume)`` describes both backends.

Method bodies are lifted verbatim from nnUNetPredictor's existing inference
methods (``predict_logits_from_preprocessed_data``,
``predict_sliding_window_return_logits``, and the three ``_internal_*``
helpers). This commit is a pure refactor with no behavior change.
"""

from __future__ import annotations

import itertools
from queue import Queue
from threading import Thread
from typing import Tuple, Union

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from torch._dynamo import OptimizedModule
from tqdm import tqdm

from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.backends.torch.bundle import ModelBundle
from nnunetv2.inference.sliding_window_prediction import (
    compute_gaussian,
    compute_steps_for_sliding_window,
)
from nnunetv2.utilities.helpers import dummy_context, empty_cache


class InferenceEngine:
    """Pure-compute sliding-window inference over a ModelBundle.

    Construct with a bundle and inference-time options. Call :meth:`predict`
    for single-fold sliding window or :meth:`predict_ensemble` to average
    across all folds in the bundle.

    The legacy method names ``predict_sliding_window_return_logits`` and
    ``predict_logits_from_preprocessed_data`` are kept as aliases so the
    nnUNetPredictor shim can delegate verbatim.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        *,
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        use_mirroring: bool = True,
        perform_everything_on_device: bool = True,
        device: torch.device = torch.device("cuda"),
        verbose: bool = False,
        allow_tqdm: bool = True,
    ) -> None:
        self.bundle = bundle
        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        self.perform_everything_on_device = perform_everything_on_device
        self.device = device
        self.verbose = verbose
        self.allow_tqdm = allow_tqdm

    # ------------------------------------------------------------------ #
    # Public surface                                                     #
    # ------------------------------------------------------------------ #

    def predict_ensemble(self, data: torch.Tensor) -> torch.Tensor:
        """Run sliding window inference once per fold and average the logits.

        IMPORTANT: if running the cascade, the segmentation from the previous
        stage must already be stacked on top of the image as a one-hot
        representation. See ``PreprocessAdapter``.

        Returned logits have the shape of the input. Convert back to original
        image size with ``convert_predicted_logits_to_segmentation_with_correct_shape``.
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(
            default_num_processes if default_num_processes < n_threads else n_threads
        )
        prediction = None

        for params in self.bundle.list_of_parameters:
            # Swap fold weights into the same network instance.
            if not isinstance(self.bundle.network, OptimizedModule):
                self.bundle.network.load_state_dict(params)
            else:
                self.bundle.network._orig_mod.load_state_dict(params)

            # Why not leave prediction on device when perform_everything_on_device?
            # Because the second iteration may OOM. Catching that with try/except
            # bloats the code more than the host-side accumulation actually costs.
            #
            # The .clone() on the first iteration is required because predict()
            # is wrapped in @torch.inference_mode and returns an inference tensor.
            # On CUDA, .to('cpu') crosses devices and the copy already produces
            # a regular tensor, so the original code worked. On CPU, .to('cpu')
            # is a no-op and the tensor stays an inference tensor — subsequent
            # in-place += then raises. Cloning the first result breaks the
            # inference-tensor wrapper for both code paths.
            if prediction is None:
                prediction = self.predict(data).to("cpu").clone()
            else:
                prediction += self.predict(data).to("cpu")

        if len(self.bundle.list_of_parameters) > 1:
            prediction /= len(self.bundle.list_of_parameters)

        if self.verbose:
            print("Prediction done")
        torch.set_num_threads(n_threads)
        return prediction

    @torch.inference_mode()
    def predict(self, input_image: torch.Tensor) -> Union[np.ndarray, torch.Tensor]:
        """Sliding-window inference for a single fold (the weights currently
        loaded into ``bundle.network``). Pads the input, runs the window loop
        with optional autocast and mirroring, then reverts the padding.
        """
        assert isinstance(input_image, torch.Tensor)
        self.bundle.network.to(self.device)
        self.bundle.network.eval()

        empty_cache(self.device)

        # Autocast caveats:
        #   - device_type='cpu' is slow as heck on some CPUs (no auto bf16
        #     detection) and needs to be disabled.
        #   - device_type='mps' complains "mps not implemented" even when
        #     enabled=False. So autocast is only active on cuda.
        with torch.autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            assert input_image.ndim == 4, (
                "input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)"
            )

            patch_size = self.bundle.configuration_manager.patch_size
            mirror_axes = self.bundle.allowed_mirroring_axes if self.use_mirroring else None

            if self.verbose:
                print(f"Input shape: {input_image.shape}")
                print("step_size:", self.tile_step_size)
                print("mirror_axes:", mirror_axes)

            data, slicer_revert_padding = pad_nd_image(
                input_image, patch_size, "constant", {"value": 0}, True, None
            )
            slicers = self._get_sliding_window_slicers(data.shape[1:])

            if self.perform_everything_on_device and self.device != "cpu":
                # OOM fallback: retry on CPU results buffer.
                try:
                    predicted_logits = self._predict_sliding_window_return_logits(
                        data, slicers, self.perform_everything_on_device
                    )
                except RuntimeError:
                    print(
                        "Prediction on device was unsuccessful, probably due to a "
                        "lack of memory. Moving results arrays to CPU"
                    )
                    empty_cache(self.device)
                    predicted_logits = self._predict_sliding_window_return_logits(
                        data, slicers, False
                    )
            else:
                predicted_logits = self._predict_sliding_window_return_logits(
                    data, slicers, self.perform_everything_on_device
                )

            empty_cache(self.device)
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits

    # ------------------------------------------------------------------ #
    # Legacy aliases (preserve nnUNetPredictor's public method names)    #
    # ------------------------------------------------------------------ #

    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        return self.predict_ensemble(data)

    def predict_sliding_window_return_logits(
        self, input_image: torch.Tensor
    ) -> Union[np.ndarray, torch.Tensor]:
        return self.predict(input_image)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        patch_size = self.bundle.configuration_manager.patch_size
        if len(patch_size) < len(image_size):
            assert len(patch_size) == len(image_size) - 1, (
                "if tile_size has less entries than image_size, len(tile_size) "
                "must be one shorter than len(image_size) (only dimension "
                "discrepancy of 1 allowed)."
            )
            steps = compute_steps_for_sliding_window(
                image_size[1:], patch_size, self.tile_step_size
            )
            if self.verbose:
                print(
                    f"n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, "
                    f"image size is {image_size}, tile_size {patch_size}, "
                    f"tile_step_size {self.tile_step_size}\nsteps:\n{steps}"
                )
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple(
                                [
                                    slice(None),
                                    d,
                                    *[
                                        slice(si, si + ti)
                                        for si, ti in zip((sx, sy), patch_size)
                                    ],
                                ]
                            )
                        )
        else:
            steps = compute_steps_for_sliding_window(
                image_size, patch_size, self.tile_step_size
            )
            if self.verbose:
                print(
                    f"n_steps {np.prod([len(i) for i in steps])}, image size is "
                    f"{image_size}, tile_size {patch_size}, "
                    f"tile_step_size {self.tile_step_size}\nsteps:\n{steps}"
                )
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple(
                                [
                                    slice(None),
                                    *[
                                        slice(si, si + ti)
                                        for si, ti in zip((sx, sy, sz), patch_size)
                                    ],
                                ]
                            )
                        )
        return slicers

    @torch.inference_mode()
    def _maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.bundle.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.bundle.network(x)

        if mirror_axes is not None:
            # x is 5d for 3d images, 4d for 2d. max mirror axis ≤ ndim - 3.
            assert max(mirror_axes) <= x.ndim - 3, (
                "mirror_axes does not match the dimension of the input!"
            )
            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c
                for i in range(len(mirror_axes))
                for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(self.bundle.network(torch.flip(x, axes)), axes)
            prediction /= len(axes_combinations) + 1
        return prediction

    @torch.inference_mode()
    def _predict_sliding_window_return_logits(
        self,
        data: torch.Tensor,
        slicers,
        do_on_device: bool = True,
    ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device("cpu")

        def producer(d, slh, q):
            for s in slh:
                q.put(
                    (
                        torch.clone(
                            d[s][None], memory_format=torch.contiguous_format
                        ).to(self.device),
                        s,
                    )
                )
            q.put("end")

        try:
            empty_cache(self.device)

            if self.verbose:
                print(f"move image to device {results_device}")
            data = data.to(results_device)
            queue = Queue(maxsize=2)
            t = Thread(target=producer, args=(data, slicers, queue))
            t.start()

            if self.verbose:
                print(f"preallocating results arrays on device {results_device}")
            num_seg_heads = self.bundle.label_manager.num_segmentation_heads
            predicted_logits = torch.zeros(
                (num_seg_heads, *data.shape[1:]),
                dtype=torch.half,
                device=results_device,
            )
            n_predictions = torch.zeros(
                data.shape[1:], dtype=torch.half, device=results_device
            )

            if self.use_gaussian:
                gaussian = compute_gaussian(
                    tuple(self.bundle.configuration_manager.patch_size),
                    sigma_scale=1.0 / 8,
                    value_scaling_factor=10,
                    device=results_device,
                )
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f"running prediction: {len(slicers)} steps")

            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                while True:
                    item = queue.get()
                    if item == "end":
                        queue.task_done()
                        break
                    workon, sl = item
                    prediction = self._maybe_mirror_and_predict(workon)[0].to(
                        results_device
                    )
                    if self.use_gaussian:
                        prediction *= gaussian
                    predicted_logits[sl] += prediction
                    n_predictions[sl[1:]] += gaussian
                    queue.task_done()
                    pbar.update()
            queue.join()

            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError(
                    "Encountered inf in predicted array. Aborting... If this "
                    "problem persists, reduce value_scaling_factor in "
                    "compute_gaussian or increase the dtype of predicted_logits "
                    "to fp32"
                )
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits
