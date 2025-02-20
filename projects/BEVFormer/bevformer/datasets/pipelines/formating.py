# Copyright (c) OpenMMLab. All rights reserved.
# Modified by Jack Morris
import torch
from mmdet.registry import TRANSFORMS
from mmdet3d.datasets.transforms import Pack3DDetInputs
from mmengine.structures import BaseDataElement as DC


@TRANSFORMS.register_module()
class CustomDefaultFormatBundle3D(Pack3DDetInputs):
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor, (3)to BaseDataElement (stack=True)
    - proposals: (1)to tensor, (2)to BaseDataElement
    - gt_bboxes: (1)to tensor, (2)to BaseDataElement
    - gt_bboxes_ignore: (1)to tensor, (2)to BaseDataElement
    - gt_labels: (1)to tensor, (2)to BaseDataElement
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        results = super(CustomDefaultFormatBundle3D, self).__call__(results)
        # results["gt_map_masks"] = DC(to_tensor(results["gt_map_masks"]), stack=True)
        results["gt_map_masks"] = DC(torch.tensor(results["gt_map_masks"]), stack=True)

        return results
