"""ModelBundle: filesystem + config view of a trained nnU-Net model folder.

A bundle is a passive container. It holds the network module, the per-fold
parameter sets, the parsed plans/configuration objects, and the dataset
metadata. It owns no inference state — that lives in the ``InferenceEngine``
(forthcoming) which takes a ``ModelBundle`` plus inference-time options.

This is the torch-side counterpart to the MLX inference port's ``ModelBundle``
and is named to match. The two backends will have the same surface so a
caller can target either through a structural ``Protocol``.

The body of :meth:`ModelBundle.from_folder` is lifted verbatim from
``nnUNetPredictor.initialize_from_trained_model_folder``; this commit is a
pure refactor with no behavior change. Compile, mirroring config, and
device placement remain in the predictor (and will move to ``InferenceEngine``
in a follow-up commit).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from batchgenerators.utilities.file_and_folder_operations import (
    isfile, join, load_json, subdirs,
)

import nnunetv2
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager, PlansManager,
)


@dataclass
class ModelBundle:
    """Trained-model artifacts needed for inference.

    Construct directly for synthetic test bundles, or via :meth:`from_folder`
    to load a real trained model from disk.

    Attributes
    ----------
    network
        The instantiated network module, already loaded with the first fold's
        weights. The remaining folds' weights live in ``list_of_parameters``
        and are swapped in by the inference engine for ensemble prediction.
    plans_manager
        Parsed nnU-Net plans.
    configuration_manager
        The specific configuration (e.g. ``3d_fullres``) used for this model.
    list_of_parameters
        One ``state_dict`` per fold, in the order of the requested folds.
    dataset_json
        Raw ``dataset.json`` contents.
    trainer_name
        Name of the trainer class that produced the checkpoints. Used by the
        engine to look up the trainer for ``build_network_architecture``.
    allowed_mirroring_axes
        Axes along which test-time mirroring may be applied. ``None`` if the
        trainer did not record this.
    """

    network: nn.Module
    plans_manager: PlansManager
    configuration_manager: ConfigurationManager
    list_of_parameters: List[dict]
    dataset_json: dict
    trainer_name: str
    allowed_mirroring_axes: Optional[Tuple[int, ...]]

    @property
    def label_manager(self):
        """Label manager derived from plans + dataset_json. Computed lazily."""
        return self.plans_manager.get_label_manager(self.dataset_json)

    @classmethod
    def from_folder(
        cls,
        model_training_output_dir: str,
        use_folds: Union[Tuple[Union[int, str], ...], List, str, None],
        checkpoint_name: str = "checkpoint_final.pth",
    ) -> "ModelBundle":
        """Discover folds, load checkpoints, build the network."""
        from nnunetv2.utilities.checkpoint_io import load_checkpoint

        if use_folds is None:
            use_folds = auto_detect_available_folds(
                model_training_output_dir, checkpoint_name
            )
        if isinstance(use_folds, str):
            use_folds = [use_folds]

        dataset_json = load_json(join(model_training_output_dir, "dataset.json"))
        plans = load_json(join(model_training_output_dir, "plans.json"))
        plans_manager = PlansManager(plans)

        parameters: List[dict] = []
        trainer_name: Optional[str] = None
        configuration_name: Optional[str] = None
        allowed_mirroring_axes: Optional[Tuple[int, ...]] = None

        for i, f in enumerate(use_folds):
            f = int(f) if f != "all" else f
            checkpoint = load_checkpoint(
                join(model_training_output_dir, f"fold_{f}", checkpoint_name),
                map_location=torch.device("cpu"),
                load_optimizer=False,
            )
            if i == 0:
                trainer_name = checkpoint["trainer_name"]
                configuration_name = checkpoint["init_args"]["configuration"]
                allowed_mirroring_axes = checkpoint.get(
                    "inference_allowed_mirroring_axes"
                )
            parameters.append(checkpoint["network_weights"])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        num_input_channels = determine_num_input_channels(
            plans_manager, configuration_manager, dataset_json
        )
        trainer_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
            trainer_name,
            "nnunetv2.training.nnUNetTrainer",
        )
        if trainer_class is None:
            raise RuntimeError(
                f"Unable to locate trainer class {trainer_name} in "
                f"nnunetv2.training.nnUNetTrainer. "
                f"Please place it there (in any .py file)!"
            )
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False,
        )
        # Initialize network with first set of parameters; see
        # https://github.com/MIC-DKFZ/nnUNet/issues/2520
        network.load_state_dict(parameters[0])

        return cls(
            network=network,
            plans_manager=plans_manager,
            configuration_manager=configuration_manager,
            list_of_parameters=parameters,
            dataset_json=dataset_json,
            trainer_name=trainer_name,
            allowed_mirroring_axes=allowed_mirroring_axes,
        )


def auto_detect_available_folds(
    model_training_output_dir: str, checkpoint_name: str
) -> List[int]:
    """Inspect a model output directory and return the fold ids that have a
    checkpoint with the given name. Excludes ``fold_all``."""
    print("use_folds is None, attempting to auto detect available folds")
    fold_folders = subdirs(model_training_output_dir, prefix="fold_", join=False)
    fold_folders = [i for i in fold_folders if i != "fold_all"]
    fold_folders = [
        i
        for i in fold_folders
        if isfile(join(model_training_output_dir, i, checkpoint_name))
    ]
    use_folds = [int(i.split("_")[-1]) for i in fold_folders]
    print(f"found the following folds: {use_folds}")
    return use_folds
