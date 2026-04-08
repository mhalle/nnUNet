"""Synthetic tests for nnunetv2.inference.backends.torch.InferenceEngine.

These exercise the engine's real sliding-window code path on a tiny
identity-like Conv3d network. The plans/configuration/label manager are
mocked because they only need to expose ``patch_size`` and
``num_segmentation_heads``; we deliberately do not construct a real
PlansManager (which would require a full plans.json) so the tests stay
self-contained and CI-friendly.

CPU-only — no CUDA, no MPS, no autocast (autocast is gated on
device.type == 'cuda' inside the engine).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from nnunetv2.inference.backends.torch import InferenceEngine, ModelBundle


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

class _TinyNet(nn.Module):
    """1x1x1 Conv3d. Output channels = num segmentation heads. The simplest
    possible network that the engine can call as ``network(patch)``."""

    def __init__(self, in_ch: int = 1, out_ch: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _make_bundle(
    *,
    in_ch: int = 1,
    out_ch: int = 2,
    patch_size=(4, 4, 4),
    network: nn.Module | None = None,
    n_folds: int = 1,
    allowed_mirroring_axes=None,
) -> ModelBundle:
    if network is None:
        network = _TinyNet(in_ch=in_ch, out_ch=out_ch)
    plans_manager = MagicMock(name="PlansManager")
    label_manager = MagicMock(name="LabelManager")
    label_manager.num_segmentation_heads = out_ch
    plans_manager.get_label_manager.return_value = label_manager
    configuration_manager = MagicMock(name="ConfigurationManager")
    configuration_manager.patch_size = patch_size
    return ModelBundle(
        network=network,
        plans_manager=plans_manager,
        configuration_manager=configuration_manager,
        list_of_parameters=[network.state_dict() for _ in range(n_folds)],
        dataset_json={"channel_names": {"0": "CT"}},
        trainer_name="nnUNetTrainer",
        allowed_mirroring_axes=allowed_mirroring_axes,
    )


def _make_engine(bundle: ModelBundle, **overrides) -> InferenceEngine:
    defaults = dict(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=False,
        device=torch.device("cpu"),
        verbose=False,
        allow_tqdm=False,
    )
    defaults.update(overrides)
    return InferenceEngine(bundle, **defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_engine_holds_bundle_and_config() -> None:
    bundle = _make_bundle()
    engine = _make_engine(bundle, tile_step_size=0.75, use_mirroring=True)

    assert engine.bundle is bundle
    assert engine.tile_step_size == 0.75
    assert engine.use_mirroring is True
    assert engine.use_gaussian is True
    assert engine.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Single-fold sliding window
# ---------------------------------------------------------------------------

def test_predict_returns_logits_with_input_spatial_shape() -> None:
    bundle = _make_bundle(out_ch=3)
    engine = _make_engine(bundle)
    volume = torch.randn(1, 8, 8, 8)  # (C, D, H, W)

    logits = engine.predict(volume)

    assert logits.shape == (3, 8, 8, 8)
    assert logits.dtype == torch.half  # engine preallocates fp16 buffer


def test_predict_pads_when_input_smaller_than_patch() -> None:
    """Patch is 4x4x4 but volume is 3x3x3. The engine should pad up."""
    bundle = _make_bundle(patch_size=(4, 4, 4), out_ch=2)
    engine = _make_engine(bundle)
    volume = torch.randn(1, 3, 3, 3)

    logits = engine.predict(volume)
    assert logits.shape == (2, 3, 3, 3)


def test_predict_handles_volume_larger_than_patch() -> None:
    """Multi-window sliding case: 12x12x12 volume with 4x4x4 patches."""
    bundle = _make_bundle(patch_size=(4, 4, 4), out_ch=2)
    engine = _make_engine(bundle)
    volume = torch.randn(1, 12, 12, 12)

    logits = engine.predict(volume)
    assert logits.shape == (2, 12, 12, 12)


def test_predict_without_gaussian() -> None:
    bundle = _make_bundle()
    engine = _make_engine(bundle, use_gaussian=False)
    volume = torch.randn(1, 8, 8, 8)

    logits = engine.predict(volume)
    assert logits.shape == (2, 8, 8, 8)


# ---------------------------------------------------------------------------
# Mirroring (TTA)
# ---------------------------------------------------------------------------

def test_predict_with_mirroring_axes() -> None:
    bundle = _make_bundle(allowed_mirroring_axes=(0, 1, 2))
    engine = _make_engine(bundle, use_mirroring=True)
    volume = torch.randn(1, 8, 8, 8)

    logits = engine.predict(volume)
    assert logits.shape == (2, 8, 8, 8)


def test_predict_mirroring_disabled_by_engine_flag() -> None:
    """Even with mirroring axes set on the bundle, use_mirroring=False
    suppresses TTA."""
    bundle = _make_bundle(allowed_mirroring_axes=(0, 1, 2))
    engine = _make_engine(bundle, use_mirroring=False)
    volume = torch.randn(1, 8, 8, 8)

    logits = engine.predict(volume)
    assert logits.shape == (2, 8, 8, 8)


# ---------------------------------------------------------------------------
# Ensemble across folds
# ---------------------------------------------------------------------------

def test_predict_ensemble_single_fold_matches_predict() -> None:
    """With one fold, predict_ensemble should produce the same shape as
    predict (and roughly the same values, modulo CPU half-precision noise)."""
    bundle = _make_bundle(n_folds=1)
    engine = _make_engine(bundle)
    torch.manual_seed(0)
    volume = torch.randn(1, 8, 8, 8)

    single = engine.predict(volume)
    ensemble = engine.predict_ensemble(volume)

    assert single.shape == ensemble.shape == (2, 8, 8, 8)


def test_predict_ensemble_averages_across_identical_folds() -> None:
    """Two folds with identical weights → ensemble logits ≈ single-fold
    logits (averaging identical values is a no-op)."""
    bundle = _make_bundle(n_folds=2)
    engine = _make_engine(bundle)
    volume = torch.randn(1, 8, 8, 8)

    single = engine.predict(volume).to("cpu").float()
    ensemble = engine.predict_ensemble(volume).float()

    # fp16 accumulation noise on CPU is small but nonzero. Loose tolerance.
    assert torch.allclose(single, ensemble, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Legacy aliases
# ---------------------------------------------------------------------------

def test_legacy_predict_sliding_window_return_logits_alias() -> None:
    bundle = _make_bundle()
    engine = _make_engine(bundle)
    volume = torch.randn(1, 8, 8, 8)

    out = engine.predict_sliding_window_return_logits(volume)
    assert out.shape == (2, 8, 8, 8)


def test_legacy_predict_logits_from_preprocessed_data_alias() -> None:
    bundle = _make_bundle(n_folds=2)
    engine = _make_engine(bundle)
    volume = torch.randn(1, 8, 8, 8)

    out = engine.predict_logits_from_preprocessed_data(volume)
    assert out.shape == (2, 8, 8, 8)


# ---------------------------------------------------------------------------
# Predictor shim still delegates correctly
# ---------------------------------------------------------------------------

def test_nnunet_predictor_shim_delegates_to_engine() -> None:
    """The predictor's predict_sliding_window_return_logits and
    predict_logits_from_preprocessed_data should produce identical output
    to driving the engine directly with the same fields."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    bundle = _make_bundle(n_folds=2)
    volume = torch.randn(1, 8, 8, 8)

    # Drive the engine directly.
    engine = _make_engine(bundle)
    direct = engine.predict_ensemble(volume).float()

    # Drive the same logic through the predictor shim, manually populating
    # the bundle fields the way manual_initialization would.
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=False,
        device=torch.device("cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.network = bundle.network
    predictor.plans_manager = bundle.plans_manager
    predictor.configuration_manager = bundle.configuration_manager
    predictor.list_of_parameters = bundle.list_of_parameters
    predictor.dataset_json = bundle.dataset_json
    predictor.trainer_name = bundle.trainer_name
    predictor.allowed_mirroring_axes = bundle.allowed_mirroring_axes
    predictor.label_manager = bundle.label_manager

    via_shim = predictor.predict_logits_from_preprocessed_data(volume).float()

    assert direct.shape == via_shim.shape
    assert torch.allclose(direct, via_shim, atol=1e-2, rtol=1e-2)
