"""Synthetic tests for nnunetv2.inference.backends.torch.ModelBundle.

These tests deliberately avoid loading a real trained model folder. They
exercise:

* The dataclass surface (construction, attribute access, label_manager
  property delegation).
* The ``auto_detect_available_folds`` helper, against a synthetic temp
  directory layout.
* The smoke import of ``nnUNetPredictor`` after the refactor — proves the
  shim's import chain still resolves and the public class instantiates.

End-to-end ``from_folder`` is exercised by the integration test suite, which
needs a real CUDA box and the Hippocampus dataset and is out of scope here.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from nnunetv2.inference.backends.torch import ModelBundle
from nnunetv2.inference.backends.torch.bundle import auto_detect_available_folds


# ---------------------------------------------------------------------------
# Dataclass surface
# ---------------------------------------------------------------------------

def _make_synthetic_bundle() -> ModelBundle:
    """Build a ModelBundle with mock collaborators. Sufficient to test the
    container surface without a real plans.json or trained network."""
    network = nn.Conv3d(1, 2, kernel_size=1)
    plans_manager = MagicMock(name="PlansManager")
    configuration_manager = MagicMock(name="ConfigurationManager")
    dataset_json = {"channel_names": {"0": "CT"}, "file_ending": ".nii.gz"}
    parameters = [network.state_dict()]
    return ModelBundle(
        network=network,
        plans_manager=plans_manager,
        configuration_manager=configuration_manager,
        list_of_parameters=parameters,
        dataset_json=dataset_json,
        trainer_name="nnUNetTrainer",
        allowed_mirroring_axes=(0, 1, 2),
    )


def test_bundle_holds_all_fields() -> None:
    bundle = _make_synthetic_bundle()
    assert isinstance(bundle.network, nn.Module)
    assert bundle.trainer_name == "nnUNetTrainer"
    assert bundle.allowed_mirroring_axes == (0, 1, 2)
    assert bundle.dataset_json["file_ending"] == ".nii.gz"
    assert len(bundle.list_of_parameters) == 1


def test_label_manager_delegates_to_plans_manager() -> None:
    bundle = _make_synthetic_bundle()
    sentinel = object()
    bundle.plans_manager.get_label_manager.return_value = sentinel

    result = bundle.label_manager

    assert result is sentinel
    bundle.plans_manager.get_label_manager.assert_called_once_with(bundle.dataset_json)


def test_label_manager_recomputed_each_access() -> None:
    """It is a property, not cached state, so callers always see fresh
    plans_manager output. Important if a future ModelBundle ever swaps the
    dataset_json or plans on the fly."""
    bundle = _make_synthetic_bundle()
    bundle.plans_manager.get_label_manager.return_value = "first"
    assert bundle.label_manager == "first"
    bundle.plans_manager.get_label_manager.return_value = "second"
    assert bundle.label_manager == "second"


def test_allowed_mirroring_axes_may_be_none() -> None:
    bundle = ModelBundle(
        network=nn.Identity(),
        plans_manager=MagicMock(),
        configuration_manager=MagicMock(),
        list_of_parameters=[{}],
        dataset_json={},
        trainer_name="nnUNetTrainer",
        allowed_mirroring_axes=None,
    )
    assert bundle.allowed_mirroring_axes is None


# ---------------------------------------------------------------------------
# Fold auto-detection
# ---------------------------------------------------------------------------

def _make_fold(root: Path, fold_name: str, with_checkpoint: bool,
               checkpoint_name: str = "checkpoint_final.pth") -> None:
    fold_dir = root / fold_name
    fold_dir.mkdir()
    if with_checkpoint:
        (fold_dir / checkpoint_name).write_bytes(b"")


def test_auto_detect_finds_folds_with_checkpoint(tmp_path: Path) -> None:
    _make_fold(tmp_path, "fold_0", with_checkpoint=True)
    _make_fold(tmp_path, "fold_1", with_checkpoint=True)
    _make_fold(tmp_path, "fold_2", with_checkpoint=True)

    folds = auto_detect_available_folds(str(tmp_path), "checkpoint_final.pth")
    assert sorted(folds) == [0, 1, 2]


def test_auto_detect_skips_folds_without_checkpoint(tmp_path: Path) -> None:
    _make_fold(tmp_path, "fold_0", with_checkpoint=True)
    _make_fold(tmp_path, "fold_1", with_checkpoint=False)
    _make_fold(tmp_path, "fold_2", with_checkpoint=True)

    folds = auto_detect_available_folds(str(tmp_path), "checkpoint_final.pth")
    assert sorted(folds) == [0, 2]


def test_auto_detect_excludes_fold_all(tmp_path: Path) -> None:
    _make_fold(tmp_path, "fold_0", with_checkpoint=True)
    _make_fold(tmp_path, "fold_all", with_checkpoint=True)

    folds = auto_detect_available_folds(str(tmp_path), "checkpoint_final.pth")
    assert folds == [0]


def test_auto_detect_respects_checkpoint_name(tmp_path: Path) -> None:
    _make_fold(tmp_path, "fold_0", with_checkpoint=True,
               checkpoint_name="checkpoint_best.pth")

    assert auto_detect_available_folds(str(tmp_path), "checkpoint_best.pth") == [0]
    assert auto_detect_available_folds(str(tmp_path), "checkpoint_final.pth") == []


# ---------------------------------------------------------------------------
# Smoke: nnUNetPredictor still imports and constructs after the shim
# ---------------------------------------------------------------------------

def test_from_task_resolves_to_standard_layout(monkeypatch) -> None:
    """from_task should call get_output_folder with the right arguments and
    forward the resolved path to from_folder."""
    captured = {}

    def fake_get_output_folder(dataset_name_or_id, trainer_name, plans_identifier, configuration):
        captured.update(
            dataset_name_or_id=dataset_name_or_id,
            trainer_name=trainer_name,
            plans_identifier=plans_identifier,
            configuration=configuration,
        )
        return "/fake/nnUNet_results/Dataset004/nnUNetTrainer__nnUNetPlans__3d_fullres"

    sentinel_bundle = MagicMock(name="ModelBundle")

    def fake_from_folder(cls_arg, model_training_output_dir, use_folds, checkpoint_name):
        captured["from_folder_dir"] = model_training_output_dir
        captured["from_folder_folds"] = use_folds
        captured["from_folder_checkpoint"] = checkpoint_name
        return sentinel_bundle

    monkeypatch.setattr(
        "nnunetv2.utilities.file_path_utilities.get_output_folder",
        fake_get_output_folder,
    )
    monkeypatch.setattr(ModelBundle, "from_folder", classmethod(fake_from_folder))

    result = ModelBundle.from_task(
        4,
        trainer_name="nnUNetTrainer",
        plans_identifier="nnUNetPlans",
        configuration="3d_fullres",
        use_folds=(0, 1),
        checkpoint_name="checkpoint_best.pth",
    )

    assert result is sentinel_bundle
    assert captured["dataset_name_or_id"] == 4
    assert captured["trainer_name"] == "nnUNetTrainer"
    assert captured["plans_identifier"] == "nnUNetPlans"
    assert captured["configuration"] == "3d_fullres"
    assert captured["from_folder_dir"] == \
        "/fake/nnUNet_results/Dataset004/nnUNetTrainer__nnUNetPlans__3d_fullres"
    assert captured["from_folder_folds"] == (0, 1)
    assert captured["from_folder_checkpoint"] == "checkpoint_best.pth"


def test_from_task_defaults() -> None:
    """from_task's defaults should match nnU-Net's standard 3D full-res run."""
    import inspect
    sig = inspect.signature(ModelBundle.from_task)
    assert sig.parameters["trainer_name"].default == "nnUNetTrainer"
    assert sig.parameters["plans_identifier"].default == "nnUNetPlans"
    assert sig.parameters["configuration"].default == "3d_fullres"
    assert sig.parameters["use_folds"].default is None
    assert sig.parameters["checkpoint_name"].default == "checkpoint_final.pth"


def test_predictor_imports_after_refactor() -> None:
    """The shim'd initialize_from_trained_model_folder must not break the
    predictor's import chain or constructor."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=False,
        device=torch.device("cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    assert predictor.network is None
    assert predictor.plans_manager is None
    assert predictor.list_of_parameters is None
