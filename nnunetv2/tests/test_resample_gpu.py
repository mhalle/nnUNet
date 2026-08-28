"""Parity tests for resample_data_or_seg_to_shape_gpu against nnU-Net's default resampler,
scipy.ndimage.zoom, skimage.transform.resize and torch's anti-aliased interpolation.

Run with:  python -m pytest nnunetv2/tests/test_resample_gpu.py      (or python -m unittest ...)
GPU-specific checks run only when a CUDA or MPS device is available.
"""
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
from nnunetv2.preprocessing.resampling.resample_gpu import resample_data_or_seg_to_shape_gpu, _best_device
from nnunetv2.preprocessing.resampling.utils import recursive_find_resampling_fn_by_name

DEV = _best_device()
HAS_GPU = DEV.type != "cpu"
ISO = (1.0, 1.0, 1.0)


def _rand_vol(shape, seed=0, dtype=np.float64, scale=1000.0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(shape) * scale
    v[:, shape[1] // 3:2 * shape[1] // 3] += 800.0      # structure: exercises prefilter + boundaries
    return v.astype(dtype)


def _smooth_field(shape, seed=5):
    """Low-frequency random field with an offset so cubic ringing stays inside the value range."""
    rng = np.random.default_rng(seed)
    low = tuple(max(2, s // 6) for s in shape)
    f = ndimage.zoom(rng.standard_normal(low), tuple(s / lo for s, lo in zip(shape, low)), order=3)[:shape[0], :shape[1]]
    return (f * 200.0 + 1000.0).astype(np.float64)


def _shape_for(vol_shape, cur, new):
    return tuple(int(round(s * c / n)) for s, c, n in zip(vol_shape, cur, new))


class TestDefaultsMatchNnunet(unittest.TestCase):
    """With defaults the function is resample_data_or_seg_to_shape (skimage resize, voxel-center)."""

    def test_data_parity_cpu_float64(self):
        vol = _rand_vol((2, 40, 52, 36))
        for new_shape in [(17, 23, 61), (80, 26, 36), (40, 52, 36)]:
            for order in (0, 1, 3):
                with self.subTest(new_shape=new_shape, order=order):
                    ref = resample_data_or_seg_to_shape(vol, new_shape, ISO, ISO, is_seg=False, order=order, order_z=0, force_separate_z=False)
                    out = resample_data_or_seg_to_shape_gpu(vol, new_shape, ISO, ISO, device="cpu", order=order)
                    self.assertLess(float(np.abs(out - ref).max()), 1e-6)

    @unittest.skipUnless(HAS_GPU, "needs a CUDA or MPS device")
    def test_data_parity_gpu_float32(self):
        vol = _rand_vol((2, 40, 52, 36)).astype(np.float32)
        for new_shape in [(17, 23, 61), (80, 26, 36)]:
            with self.subTest(new_shape=new_shape):
                ref = resample_data_or_seg_to_shape(vol.astype(np.float64), new_shape, ISO, ISO, is_seg=False, order=3, order_z=0, force_separate_z=False)
                out = resample_data_or_seg_to_shape_gpu(vol, new_shape, ISO, ISO, device=DEV, order=3)
                self.assertLess(float(np.abs(out.astype(np.float64) - ref).max()), 2e-2)   # float32 on HU-scale values

    def test_seg_parity(self):
        """Labels follow resize_segmentation: indicator resized, painted where >= 0.5 in ascending order."""
        rng = np.random.default_rng(2)
        seg = np.zeros((1, 31, 29, 35), dtype=np.int16)
        for lab in range(1, 9):
            z, y, x = rng.integers(0, 20, 3)
            seg[0, z:z + 12, y:y + 11, x:x + 13] = lab
        for new_shape in [(14, 67, 35), (62, 58, 70)]:
            for order in (0, 1):
                ref = resample_data_or_seg_to_shape(seg, new_shape, ISO, ISO, is_seg=True, order=order, order_z=0, force_separate_z=False)
                for device in (["cpu"] + ([DEV] if HAS_GPU else [])):
                    with self.subTest(new_shape=new_shape, order=order, device=str(device)):
                        out = resample_data_or_seg_to_shape_gpu(seg, new_shape, ISO, ISO, is_seg=True, device=device, order=order)
                        self.assertTrue(np.array_equal(out.astype(np.int64), ref.astype(np.int64)))

    def test_seg_threshold_at_exact_half(self):
        """2x downsampling with order 1 yields indicator values of exactly 0.5: the >= rule must match."""
        seg = np.random.default_rng(4).integers(0, 5, size=(1, 32, 30, 28)).astype(np.int16)
        ref = resample_data_or_seg_to_shape(seg, (16, 15, 14), ISO, ISO, is_seg=True, order=1, order_z=0, force_separate_z=False)
        out = resample_data_or_seg_to_shape_gpu(seg, (16, 15, 14), ISO, ISO, is_seg=True, device="cpu", order=1)
        self.assertTrue(np.array_equal(out.astype(np.int64), ref.astype(np.int64)))

    def test_separate_z_data(self):
        vol = _rand_vol((2, 40, 44, 12))
        cases = [((0.7, 0.7, 5.0), (1.0, 1.0, 1.0), None),      # anisotropic -> separate along z, z upsampled 5x
                 ((5.0, 0.7, 0.7), (1.5, 1.5, 1.5), None),      # low-res axis 0
                 ((0.24, 1.25, 1.25), (1.0, 1.0, 1.0), None),   # two equally coarse axes -> upstream does not separate
                 ((0.7, 0.7, 5.0), (1.0, 1.0, 1.0), True),
                 ((0.7, 0.7, 5.0), (1.0, 1.0, 1.0), False)]
        for cur, new, force in cases:
            new_shape = _shape_for(vol.shape[1:], cur, new)
            for order, order_z in ((3, 0), (1, 0), (3, 1), (3, 3)):
                with self.subTest(cur=cur, new=new, force=force, order=order, order_z=order_z):
                    ref = resample_data_or_seg_to_shape(vol, new_shape, cur, new, is_seg=False, order=order, order_z=order_z, force_separate_z=force)
                    out = resample_data_or_seg_to_shape_gpu(vol, new_shape, cur, new, device="cpu", order=order, order_z=order_z, force_separate_z=force)
                    self.assertLess(float(np.abs(out - ref).max()), 1e-6)

    def test_separate_z_seg(self):
        rng = np.random.default_rng(3)
        seg = np.zeros((1, 40, 44, 12), dtype=np.int16)
        for lab in range(1, 9):
            z, y = rng.integers(0, 24, 2)
            x = int(rng.integers(0, 6))
            seg[0, z:z + 14, y:y + 13, x:x + 5] = lab
        cur, new = (0.7, 0.7, 5.0), (1.0, 1.0, 1.0)
        new_shape = _shape_for(seg.shape[1:], cur, new)
        for order, order_z, force in ((1, 0, None), (0, 0, None), (1, 1, True), (3, 3, True), (1, 0, False)):
            ref = resample_data_or_seg_to_shape(seg, new_shape, cur, new, is_seg=True, order=order, order_z=order_z, force_separate_z=force)
            for device in (["cpu"] + ([DEV] if HAS_GPU else [])):
                with self.subTest(order=order, order_z=order_z, force=force, device=str(device)):
                    out = resample_data_or_seg_to_shape_gpu(seg, new_shape, cur, new, is_seg=True, device=device, order=order, order_z=order_z, force_separate_z=force)
                    self.assertTrue(np.array_equal(out.astype(np.int64), ref.astype(np.int64)))


class TestCornerConventionMatchesScipyZoom(unittest.TestCase):
    """convention="corner" is scipy.ndimage.zoom (grid_mode=False): voxel-corner point grid."""

    def test_float64_cpu(self):
        vol = _rand_vol((1, 40, 52, 36))
        for new_shape in [(17, 23, 61), (80, 26, 36), (40, 52, 36)]:
            for order in (0, 1, 3):
                for mode in ("nearest", "mirror"):
                    with self.subTest(new_shape=new_shape, order=order, mode=mode):
                        ref = ndimage.zoom(vol[0], np.array(new_shape) / np.array(vol.shape[1:]), order=order, mode=mode)
                        out = resample_data_or_seg_to_shape_gpu(vol, new_shape, device="cpu", convention="corner", order=order, mode=mode)[0]
                        self.assertLess(float(np.abs(out - ref).max()), 1e-6)

    def test_integer_input_rounds_like_scipy(self):
        vol = _rand_vol((1, 30, 34, 28)).astype(np.int16)
        ref = ndimage.zoom(vol[0], np.array((13, 61, 28)) / np.array(vol.shape[1:]), order=3, mode="nearest")
        out = resample_data_or_seg_to_shape_gpu(vol, (13, 61, 28), device="cpu", convention="corner", order=3)[0]
        self.assertEqual(out.dtype, np.int16)
        self.assertGreater(float((out == ref).mean()), 0.999)

    def test_seg_nearest_exact(self):
        seg = np.random.default_rng(1).integers(0, 120, size=(1, 31, 29, 35)).astype(np.int16)
        for new_shape in [(14, 67, 35), (62, 58, 70)]:
            ref = ndimage.zoom(seg[0], np.array(new_shape) / np.array(seg.shape[1:]), order=0, mode="nearest")
            for device in (["cpu"] + ([DEV] if HAS_GPU else [])):
                with self.subTest(new_shape=new_shape, device=str(device)):
                    out = resample_data_or_seg_to_shape_gpu(seg, new_shape, is_seg=True, device=device, convention="corner", order=0)[0]
                    self.assertTrue(np.array_equal(out.astype(np.int64), ref.astype(np.int64)))


class TestAntiAliasPolicy(unittest.TestCase):
    """anti_alias=True: PIL-style filter, identical to torch's F.interpolate(antialias=True)."""

    def test_is_opt_in(self):
        vol = _rand_vol((1, 40, 52, 36)).astype(np.float32)
        a = resample_data_or_seg_to_shape_gpu(vol, (17, 23, 61), device="cpu")[0]
        b = resample_data_or_seg_to_shape_gpu(vol, (17, 23, 61), device="cpu", anti_alias=True)[0]
        self.assertFalse(np.array_equal(a, b))

    def test_downsample_matches_torch_bicubic_antialias(self):
        img = _smooth_field((96, 80))
        vol = img[None, None]                                   # (c=1, x=1, y, z): x is identity
        ours = resample_data_or_seg_to_shape_gpu(vol, (1, 42, 35), device="cpu", anti_alias=True)[0, 0]
        ref = F.interpolate(torch.as_tensor(img)[None, None], size=(42, 35), mode="bicubic", antialias=True, align_corners=False)[0, 0].numpy()
        interior = (ours > img.min() + 1e-9) & (ours < img.max() - 1e-9)   # where the range clip cannot have acted
        self.assertGreater(interior.mean(), 0.9)
        self.assertLess(float(np.abs(ours - ref)[interior].max()), 1e-6)

    def test_upsample_matches_torch_bilinear_antialias(self):
        img = _smooth_field((40, 36), seed=6)
        ours = resample_data_or_seg_to_shape_gpu(img[None, None], (1, 68, 61), device="cpu", anti_alias=True)[0, 0]
        ref = F.interpolate(torch.as_tensor(img)[None, None], size=(68, 61), mode="bilinear", antialias=True, align_corners=False)[0, 0].numpy()
        self.assertLess(float(np.abs(ours - ref).max()), 1e-6)

    def test_3d_is_separable(self):
        vol = np.stack([_smooth_field((64, 56), seed=10 + i) for i in range(30)], axis=0)[None]   # (1, 30, 64, 56)
        full = resample_data_or_seg_to_shape_gpu(vol, (13, 28, 24), device="cpu", anti_alias=True)[0]
        step = F.interpolate(torch.as_tensor(vol[0])[:, None], size=(28, 24), mode="bicubic", antialias=True, align_corners=False)[:, 0]
        step_t = step.permute(1, 2, 0).reshape(1, 1, 28 * 24, 30)
        ref = F.interpolate(step_t, size=(28 * 24, 13), mode="bicubic", antialias=True, align_corners=False)[0, 0].reshape(28, 24, 13).permute(2, 0, 1).numpy()
        interior = (full > vol.min() + 1e-9) & (full < vol.max() - 1e-9)
        self.assertGreater(interior.mean(), 0.9)
        self.assertLess(float(np.abs(full - ref)[interior].max()), 1e-6)

    def test_downsample_suppresses_aliasing(self):
        """White noise (std 100) decimated 8x: the anti-aliased filter averages ~16 samples per axis and
        collapses the variance; plain trilinear (2 taps) keeps most of it as aliased noise."""
        d = (np.random.default_rng(0).standard_normal((1, 64, 64, 64)) * 100.0).astype(np.float32)
        out = resample_data_or_seg_to_shape_gpu(d, (64, 8, 8), device="cpu", anti_alias=True)
        plain = F.interpolate(torch.as_tensor(d)[None], (64, 8, 8), mode="trilinear", antialias=False)[0].numpy()
        self.assertLess(float(out.std()), 0.6 * float(plain.std()))
        self.assertLess(float(out.std()), 0.4 * float(d.std()))

    def test_output_clipped_to_input_range(self):
        d = np.zeros((1, 64, 64, 64), dtype=np.float32)
        d[:, 16:48, 16:48, 16:48] = 1000.0
        out = resample_data_or_seg_to_shape_gpu(d, (32, 32, 32), device="cpu", anti_alias=True)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1000.0)


class TestContract(unittest.TestCase):
    def test_identity_shape_is_noop(self):
        x = np.random.default_rng(0).standard_normal((1, 8, 9, 10)).astype(np.float32)
        self.assertIs(resample_data_or_seg_to_shape_gpu(x, (8, 9, 10), device="cpu"), x)

    def test_container_and_dtype(self):
        x = np.random.default_rng(0).standard_normal((1, 12, 14, 16)).astype(np.float32)
        out = resample_data_or_seg_to_shape_gpu(x, (6, 7, 8), device="cpu")
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.dtype, np.float32)
        t = torch.as_tensor(x)
        out_t = resample_data_or_seg_to_shape_gpu(t, (6, 7, 8), device="cpu")
        self.assertIsInstance(out_t, torch.Tensor)
        self.assertEqual(out_t.device, t.device)

    def test_channel_chunking_is_exact(self):
        x = np.random.default_rng(0).standard_normal((20, 12, 14, 16)).astype(np.float32)
        a = resample_data_or_seg_to_shape_gpu(x, (6, 7, 8), device="cpu", channel_chunk=8)
        b = resample_data_or_seg_to_shape_gpu(x, (6, 7, 8), device="cpu", channel_chunk=64)
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)      # BLAS blocking differs with matrix height: ulp-level only

    def test_seg_emits_only_input_labels(self):
        seg = np.zeros((1, 32, 32, 32), dtype=np.int16)
        seg[:, 4:20, 4:20, 4:20] = 3
        seg[:, 10:30, 10:30, 10:30] = 7
        out = resample_data_or_seg_to_shape_gpu(seg, (10, 10, 10), is_seg=True, device="cpu", anti_alias=True)
        self.assertTrue(set(np.unique(out).tolist()) <= {0, 3, 7})
        self.assertEqual(out.shape, (1, 10, 10, 10))

    def test_discoverable_by_plans_name(self):
        self.assertIs(recursive_find_resampling_fn_by_name("resample_data_or_seg_to_shape_gpu"), resample_data_or_seg_to_shape_gpu)

    def test_unknown_convention_raises(self):
        x = np.zeros((1, 4, 4, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            resample_data_or_seg_to_shape_gpu(x, (2, 2, 2), device="cpu", convention="edge")


if __name__ == "__main__":
    unittest.main()
