from mmpose.datasets.pipelines import PIPELINES
from mmpose.datasets.builder import build_from_cfg

import albumentations as A


class Pose_Compose(A.Compose):

    def __init__(self,
                 transforms,
                 bbox_params=None,
                 keypoint_params=None,
                 additional_targets=None,
                 p: float = 1):
        pose_trans = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                pose_trans.append(transform)
            elif callable(transform):
                pose_trans.append(transform)
            else:
                raise TypeError('transform must be callable or a dict, but got'
                                f' {type(transform)}')
        super().__init__(transforms=pose_trans,
                         bbox_params=bbox_params,
                         keypoint_params=keypoint_params,
                         additional_targets=additional_targets,
                         p=p)
