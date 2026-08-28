import unittest
from functools import partial

import numpy as np
import torch

from nnunetv2.inference.logits_to_segmentation import fused_resample_and_convert, resample_and_convert
from nnunetv2.inference.restore import available_backends
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
from nnunetv2.utilities.label_handling.label_handling import LabelManager


def multiclass_label_manager(n_classes: int = 5) -> LabelManager:
    return LabelManager({'background': 0, **{f'c{i}': i for i in range(1, n_classes)}}, None)


def region_label_manager() -> LabelManager:
    # nested regions, the BraTS shape: each head is a region, painted in regions_class_order
    return LabelManager({'background': 0, 'whole': (1, 2, 3), 'core': (2, 3), 'enhancing': 3},
                        regions_class_order=(1, 2, 3))


def smooth_logits(n_heads, shape, seed=0):
    """Logits with spatial structure - random noise makes every voxel a boundary and would
    exercise only the tie-breaking, not the interpolation."""
    rng = np.random.default_rng(seed)
    coarse = rng.normal(0, 3, (n_heads, *[max(2, s // 3) for s in shape])).astype(np.float32)
    return resample_data_or_seg_to_shape(coarse, shape, [1.0] * 3, [1.0] * 3, is_seg=False).astype(np.float32)


# what the plans pass for probabilities: linear, and let the spacing decide separate-z.
# The bare function defaults to order=3, which the fused trilinear pass cannot reproduce.
PROB_FN = partial(resample_data_or_seg_to_shape, order=1, order_z=0, force_separate_z=None)


def default_of(logits, new_shape, label_manager):
    return resample_and_convert(logits, new_shape, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
                                label_manager=label_manager,
                                resampling_fn_probabilities=PROB_FN)


def fused_of(logits, new_shape, label_manager, **kw):
    return fused_resample_and_convert(logits, new_shape, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
                                      label_manager=label_manager,
                                      resampling_fn_probabilities=PROB_FN, **kw)


class TestLogitsToSegmentation(unittest.TestCase):
    """The fused path must produce what the two-step path produces. Everything else about it
    is an optimization; this is the only property that makes it substitutable."""

    def test_argmax_matches_the_two_step_path(self):
        lm = multiclass_label_manager(5)
        logits = smooth_logits(5, (14, 16, 12))
        for new_shape in ((28, 32, 24), (9, 11, 7), (14, 16, 12)):
            with self.subTest(new_shape=new_shape):
                np.testing.assert_array_equal(np.asarray(fused_of(logits, new_shape, lm)),
                                              np.asarray(default_of(logits, new_shape, lm)))

    def test_regions_match_the_two_step_path(self):
        """sigmoid(x) > 0.5 is x > 0, so the fused pass thresholds the logit directly. It also
        has to paint in regions_class_order, later region winning, as nnU-Net does."""
        lm = region_label_manager()
        logits = smooth_logits(3, (12, 14, 10))
        np.testing.assert_array_equal(np.asarray(fused_of(logits, (24, 28, 20), lm)),
                                      np.asarray(default_of(logits, (24, 28, 20), lm)))

    def test_falls_back_when_the_order_is_not_trilinear(self):
        lm = multiclass_label_manager(4)
        logits = smooth_logits(4, (10, 12, 8))
        out = fused_of(logits, (20, 24, 16), lm, order=3)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(default_of(logits, (20, 24, 16), lm)))

    def test_the_cpu_kernel_matches_the_two_step_path(self):
        """An explicit device='cpu' uses the PyTorch fallback kernel. It is slower than the
        two-step path - it exists for correctness - but it must still agree with it."""
        lm = multiclass_label_manager(4)
        logits = smooth_logits(4, (10, 12, 8))
        out = fused_of(logits, (20, 24, 16), lm, device='cpu', backend='torch')
        np.testing.assert_array_equal(np.asarray(out), np.asarray(default_of(logits, (20, 24, 16), lm)))

    def test_container_type_is_preserved(self):
        lm = multiclass_label_manager(4)
        logits = smooth_logits(4, (10, 12, 8))
        self.assertIsInstance(fused_of(logits, (20, 24, 16), lm), np.ndarray)
        self.assertIsInstance(fused_of(torch.from_numpy(logits), (20, 24, 16), lm), torch.Tensor)

    def test_separate_z_matches_the_two_step_path(self):
        """Anisotropic spacing makes nnU-Net resample the low-resolution axis with order_z.
        The fused pass has to make the same decision and use nearest along that axis."""
        lm = multiclass_label_manager(4)
        logits = smooth_logits(4, (6, 16, 16))
        expected = resample_and_convert(logits, (12, 32, 32), [5.0, 1.0, 1.0], [2.5, 0.5, 0.5],
                                        label_manager=lm,
                                        resampling_fn_probabilities=PROB_FN)
        actual = fused_resample_and_convert(logits, (12, 32, 32), [5.0, 1.0, 1.0], [2.5, 0.5, 0.5],
                                            label_manager=lm,
                                            resampling_fn_probabilities=PROB_FN)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_identity_shape_is_a_plain_decision(self):
        lm = multiclass_label_manager(4)
        logits = smooth_logits(4, (8, 8, 8))
        np.testing.assert_array_equal(np.asarray(fused_of(logits, (8, 8, 8), lm)),
                                      np.asarray(logits.argmax(0)))

    def test_a_gpu_backend_is_used_when_one_exists(self):
        """Guards against the fused path quietly degrading to the fallback everywhere, which
        would make every test above pass for the wrong reason."""
        gpu = [b for b in available_backends() if b in ('metal', 'triton')]
        if not gpu:
            self.skipTest('no GPU backend available on this machine')
        lm = multiclass_label_manager(5)
        logits = smooth_logits(5, (12, 14, 10))
        device = 'mps' if 'metal' in gpu else 'cuda'
        np.testing.assert_array_equal(np.asarray(fused_of(logits, (24, 28, 20), lm, device=device)),
                                      np.asarray(default_of(logits, (24, 28, 20), lm)))


class TestPlansResolution(unittest.TestCase):
    """The plans key must be optional: every plans file written before it existed has to keep
    resolving to the two-step behavior."""

    @staticmethod
    def _config(**extra):
        from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager
        base = {'architecture': {'network_class_name': 'x', 'arch_kwargs': {}, '_kw_requires_import': []}}
        return ConfigurationManager({**base, **extra})

    def test_absent_key_resolves_to_the_two_step_path(self):
        fn = self._config().logits_to_segmentation_fn
        self.assertIs(fn.func, resample_and_convert)

    def test_key_can_name_the_fused_path(self):
        fn = self._config(logits_to_segmentation_fn='fused_resample_and_convert',
                          logits_to_segmentation_fn_kwargs={'order': 1}).logits_to_segmentation_fn
        self.assertIs(fn.func, fused_resample_and_convert)
        self.assertEqual(fn.keywords, {'order': 1})

    def test_unknown_name_raises(self):
        with self.assertRaises(RuntimeError):
            _ = self._config(logits_to_segmentation_fn='no_such_function').logits_to_segmentation_fn


class TestExportUsesTheSeam(unittest.TestCase):
    """convert_predicted_logits_to_segmentation_with_correct_shape now routes the resampling
    and the decision through one call. These check that it still produces what it did, and
    that exporting probabilities keeps the two steps apart (it needs the resampled volume)."""

    @staticmethod
    def _fixture(fn):
        from types import SimpleNamespace
        src, target = (10, 12, 8), (20, 24, 16)
        logits = smooth_logits(4, src)
        plans_manager = SimpleNamespace(transpose_forward=[0, 1, 2], transpose_backward=[0, 1, 2])
        configuration_manager = SimpleNamespace(spacing=[1.0, 1.0, 1.0],
                                                resampling_fn_probabilities=PROB_FN,
                                                logits_to_segmentation_fn=fn)
        properties = {'spacing': [1.0, 1.0, 1.0],
                      'shape_after_cropping_and_before_resampling': target,
                      'shape_before_cropping': target,
                      'bbox_used_for_cropping': [[0, target[0]], [0, target[1]], [0, target[2]]]}
        return logits, plans_manager, configuration_manager, properties

    def _run(self, fn, **kw):
        from functools import partial as _partial
        from nnunetv2.inference.export_prediction import \
            convert_predicted_logits_to_segmentation_with_correct_shape
        bound = _partial(fn, **kw) if kw else fn
        logits, pm, cm, props = self._fixture(bound)
        return convert_predicted_logits_to_segmentation_with_correct_shape(
            logits, pm, cm, multiclass_label_manager(4), props)

    def test_fused_export_matches_the_default_export(self):
        np.testing.assert_array_equal(self._run(fused_resample_and_convert),
                                      self._run(resample_and_convert))

    def test_exporting_probabilities_still_works(self):
        from nnunetv2.inference.export_prediction import \
            convert_predicted_logits_to_segmentation_with_correct_shape
        logits, pm, cm, props = self._fixture(fused_resample_and_convert)
        seg, probs = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits, pm, cm, multiclass_label_manager(4), props, return_probabilities=True)
        self.assertEqual(tuple(seg.shape), tuple(props['shape_before_cropping']))
        self.assertEqual(probs.shape[0], 4)


if __name__ == '__main__':
    unittest.main()
