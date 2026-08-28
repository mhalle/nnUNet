from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import \
    nnUNetPlannerResEncL


class _FusedLogitsToSegmentationMixin:
    """Emit plans that fuse the export resampling with the decision.

    The result is the same segmentation the default two-step export produces; what changes
    is that the resampled (num_segmentation_heads, *new_shape) probability volume is never
    built. The kwargs mirror the softmax export kwargs, so the two paths agree - and the
    fused function falls back to the two-step one whenever they would not.
    """

    def get_plans_for_configuration(self, *args, **kwargs):
        plan = super().get_plans_for_configuration(*args, **kwargs)
        plan['logits_to_segmentation_fn'] = 'fused_resample_and_convert'
        plan['logits_to_segmentation_fn_kwargs'] = {
            'order': 1,
            'order_z': 0,
            'force_separate_z': None,
        }
        return plan


class nnUNetPlannerFusedExport(_FusedLogitsToSegmentationMixin, ExperimentPlanner):
    pass


class nnUNetPlannerResEncLFusedExport(_FusedLogitsToSegmentationMixin, nnUNetPlannerResEncL):
    pass
