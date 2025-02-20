# Copyright 2021 Toyota Research Institute.  All rights reserved.
from typing import Sequence
import torch
from torch import nn

# from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess as resize_instances
from detectron2.structures import Instances
from detectron2.layers import ShapeSpec
# from mmcv.runner import force_fp32

from .fcos2d import FCOS2DHead, FCOS2DLoss
from .fcos3d import FCOS3DHead, FCOS3DLoss

# from tridet.modeling.dd3d.postprocessing import nuscenes_sample_aggregate
from .prepare_targets import DD3DTargetPreparer

# from tridet.modeling.feature_extractor import build_feature_extractor
from ..structures.image_list import ImageList
from ..utils.tensor2d import (
    compute_features_locations as compute_locations_per_level,
)


# @META_ARCH_REGISTRY.register()
class DD3D(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        strides: Sequence[int],
        fcos2d_cfg=dict(),
        fcos2d_loss_cfg=dict(),
        fcos3d_cfg=dict(),
        fcos3d_loss_cfg=dict(),
        target_assign_cfg=dict(),
        box3d_on=True,
        feature_locations_offset="none",
    ):
        super().__init__()

        self.backbone_output_shape = [
            ShapeSpec(channels=in_channels, stride=s) for s in strides
        ]

        self.feature_locations_offset = feature_locations_offset

        self.fcos2d_head = FCOS2DHead(
            num_classes=num_classes,
            input_shape=self.backbone_output_shape,
            **fcos2d_cfg,
        )
        self.fcos2d_loss = FCOS2DLoss(num_classes=num_classes, **fcos2d_loss_cfg)

        if box3d_on:
            self.fcos3d_head = FCOS3DHead(
                num_classes=num_classes,
                input_shape=self.backbone_output_shape,
                **fcos3d_cfg,
            )
            self.fcos3d_loss = FCOS3DLoss(num_classes=num_classes, **fcos3d_loss_cfg)
            # NOTE: inference later
            # self.fcos3d_inference = FCOS3DInference(cfg)
            self.only_box2d = False
        else:
            self.only_box2d = True

        self.prepare_targets = DD3DTargetPreparer(
            num_classes=num_classes,
            input_shape=self.backbone_output_shape,
            box3d_on=box3d_on,
            **target_assign_cfg,
        )

        # nuScenes inference aggregates detections over all 6 cameras.
        # self.nusc_sample_aggregate_in_inference = cfg.DD3D.INFERENCE.NUSC_SAMPLE_AGGREGATE
        self.num_classes = num_classes

    # @force_fp32(apply_to=("features"))
    def forward(self, features: Sequence[torch.Tensor], batched_inputs):
        features = [f.float() for f in features]
        # get inv_intrinsics
        if "inv_intrinsics" in batched_inputs[0]:
            inv_intrinsics = [
                x["inv_intrinsics"].to(features[0].device) for x in batched_inputs
            ]
            inv_intrinsics = torch.stack(inv_intrinsics, dim=0)
        else:
            inv_intrinsics = None
        # get instances
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(features[0].device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        locations = self.compute_locations(features)
        logits, box2d_reg, centerness, _ = self.fcos2d_head(features)
        if not self.only_box2d:
            box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth = (
                self.fcos3d_head(features)
            )
        else:
            box3d_quat = box3d_ctr = box3d_depth = box3d_size = box3d_conf = (
                dense_depth
            ) = None

        if self.training:
            assert gt_instances is not None
            feature_shapes = [x.shape[-2:] for x in features]
            training_targets = self.prepare_targets(
                locations, gt_instances, feature_shapes
            )

            losses = {}
            fcos2d_loss, fcos2d_info = self.fcos2d_loss(
                logits, box2d_reg, centerness, training_targets
            )
            losses.update(fcos2d_loss)

            if not self.only_box2d:
                fcos3d_loss = self.fcos3d_loss(
                    box3d_quat,
                    box3d_ctr,
                    box3d_depth,
                    box3d_size,
                    box3d_conf,
                    dense_depth,
                    inv_intrinsics,
                    fcos2d_info,
                    training_targets,
                )
                losses.update(fcos3d_loss)
            return losses

    def compute_locations(self, features):
        locations = []
        in_strides = [x.stride for x in self.backbone_output_shape]
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations_per_level(
                h,
                w,
                in_strides[level],
                feature.dtype,
                feature.device,
                offset=self.feature_locations_offset,
            )
            locations.append(locations_per_level)
        return locations

    def forward_train(self, features, batched_inputs):
        self.train()
        return self.forward(features, batched_inputs)
