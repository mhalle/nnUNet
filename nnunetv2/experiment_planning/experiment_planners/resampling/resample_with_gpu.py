from typing import Union, List, Tuple

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import \
    nnUNetPlannerResEncL
from nnunetv2.preprocessing.resampling.resample_gpu import resample_data_or_seg_to_shape_gpu


class _GPUResamplingMixin:
    """Same resampling as the default planner (resample_data_or_seg_to_shape), executed by
    resample_data_or_seg_to_shape_gpu on MPS / CUDA / CPU. The kwargs are the defaults' kwargs,
    so preprocessing and export produce the same arrays as the scipy/skimage path."""

    def determine_resampling(self, *args, **kwargs):
        resampling_data = resample_data_or_seg_to_shape_gpu
        resampling_data_kwargs = {
            "is_seg": False,
            "order": 3,
            "order_z": 0,
            "force_separate_z": None,
        }
        resampling_seg = resample_data_or_seg_to_shape_gpu
        resampling_seg_kwargs = {
            "is_seg": True,
            "order": 1,
            "order_z": 0,
            "force_separate_z": None,
        }
        return resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs

    def determine_segmentation_softmax_export_fn(self, *args, **kwargs):
        resampling_fn = resample_data_or_seg_to_shape_gpu
        resampling_fn_kwargs = {
            "is_seg": False,
            "order": 1,
            "order_z": 0,
            "force_separate_z": None,
        }
        return resampling_fn, resampling_fn_kwargs


class nnUNetPlanner_gpures(_GPUResamplingMixin, ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetPlans_gpures',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

    def generate_data_identifier(self, configuration_name: str) -> str:
        return self.plans_identifier + '_' + configuration_name


class nnUNetPlannerResEncL_gpures(_GPUResamplingMixin, nnUNetPlannerResEncL):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 24,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetLPlans_gpures',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

    def generate_data_identifier(self, configuration_name: str) -> str:
        return self.plans_identifier + '_' + configuration_name
