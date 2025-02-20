# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
"""None defaults hurt type inference"""

import copy
from typing import Dict, List, Optional, Sequence

import torch
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.structures import Det3DDataSample
from mmdet3d.structures.ops import bbox3d2result
from mmengine.registry import MODELS
from mmengine.structures import InstanceData
from torch import Tensor

from ...models.utils.grid_mask import GridMask


@MODELS.register_module()
class BEVFormerOld(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(
        self,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
    ):
        super(BEVFormerOld, self).__init__(
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_fusion_layer,
            img_backbone,
            pts_backbone,
            img_neck,
            pts_neck,
            pts_bbox_head,
            img_roi_head,
            img_rpn_head,
            train_cfg,
            test_cfg,
            pretrained,
        )
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

    #
    # def loss(
    #     self,
    #     batch_inputs_dict: Dict[List, torch.Tensor],
    #     batch_data_samples: List[Det3DDataSample],
    #     **kwargs,
    # ) -> List[Det3DDataSample]:
    #     """
    #     Args:
    #         batch_inputs_dict (dict): The model input dict which include
    #             'points' and `imgs` keys.
    #
    #             - points (list[torch.Tensor]): Point cloud of each sample.
    #             - imgs (torch.Tensor): Tensor of batch images, has shape
    #               (B, C, H ,W)
    #         batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
    #             Samples. It usually includes information such as
    #             `gt_instance_3d`, .
    #
    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components.
    #
    #     """
    #
    #     batch_input_metas = [item.metainfo for item in batch_data_samples]
    #     img_feats, pts_feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
    #     losses = dict()
    #     if pts_feats:
    #         losses_pts = self.pts_bbox_head.loss(
    #             pts_feats, batch_data_samples, **kwargs
    #         )
    #         losses.update(losses_pts)
    #     if img_feats:
    #         losses_img = self.loss_imgs(img_feats, batch_data_samples)
    #         losses.update(losses_img)
    #     return losses
    #
    # def loss_imgs(
    #     self, x: List[Tensor], batch_data_samples: List[Det3DDataSample], **kwargs
    # ):
    #     """Forward function for image branch.
    #
    #     This function works similar to the forward function of Faster R-CNN.
    #
    #     Args:
    #         x (list[torch.Tensor]): Image features of shape (B, C, H, W)
    #             of multiple levels.
    #         batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
    #             Samples. It usually includes information such as
    #             `gt_instance_3d`, .
    #
    #     Returns:
    #         dict: Losses of each branch.
    #     """
    #     losses = dict()
    #     # RPN forward and loss
    #     if self.with_img_rpn:
    #         proposal_cfg = self.test_cfg.rpn
    #         rpn_data_samples = copy.deepcopy(batch_data_samples)
    #         # set cat_id of gt_labels to 0 in RPN
    #         for data_sample in rpn_data_samples:
    #             data_sample.gt_instances.labels = torch.zeros_like(
    #                 data_sample.gt_instances.labels
    #             )
    #         rpn_losses, rpn_results_list = self.img_rpn_head.loss_and_predict(
    #             x, rpn_data_samples, proposal_cfg=proposal_cfg, **kwargs
    #         )
    #         # avoid get same name with roi_head loss
    #         keys = rpn_losses.keys()
    #         for key in keys:
    #             if "loss" in key and "rpn" not in key:
    #                 rpn_losses[f"rpn_{key}"] = rpn_losses.pop(key)
    #         losses.update(rpn_losses)
    #
    #     else:
    #         if "proposals" in batch_data_samples[0]:
    #             # use pre-defined proposals in InstanceData
    #             # for the second stage
    #             # to extract ROI features.
    #             rpn_results_list = [
    #                 data_sample.proposals for data_sample in batch_data_samples
    #             ]
    #         else:
    #             rpn_results_list = None
    #     # bbox head forward and loss
    #     if self.with_img_bbox:
    #         roi_losses = self.img_roi_head.loss(
    #             x, rpn_results_list, batch_data_samples, **kwargs
    #         )
    #         losses.update(roi_losses)
    #     return losses
    #
    # def predict_imgs(
    #     self,
    #     x: List[Tensor],
    #     batch_data_samples: List[Det3DDataSample],
    #     rescale: bool = True,
    #     **kwargs,
    # ) -> InstanceData:
    #     """Predict results from a batch of inputs and data samples with post-
    #     processing.
    #
    #     Args:
    #         x (List[Tensor]): Image features from FPN.
    #         batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
    #             Samples. It usually includes information such as
    #             `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
    #         rescale (bool): Whether to rescale the results.
    #             Defaults to True.
    #     """
    #
    #     if batch_data_samples[0].get("proposals", None) is None:
    #         rpn_results_list = self.img_rpn_head.predict(
    #             x, batch_data_samples, rescale=False
    #         )
    #     else:
    #         rpn_results_list = [
    #             data_sample.proposals for data_sample in batch_data_samples
    #         ]
    #     results_list = self.img_roi_head.predict(
    #         x, rpn_results_list, batch_data_samples, rescale=rescale, **kwargs
    #     )
    #     return results_list
    #
    # def predict(
    #     self,
    #     batch_inputs_dict: Dict[str, Optional[Tensor]],
    #     batch_data_samples: List[Det3DDataSample],
    #     **kwargs,
    # ) -> List[Det3DDataSample]:
    #     """Forward of testing.
    #
    #     Args:
    #         batch_inputs_dict (dict): The model input dict which include
    #             'points' keys.
    #
    #             - points (list[torch.Tensor]): Point cloud of each sample.
    #         batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
    #             Samples. It usually includes information such as
    #             `gt_instance_3d`.
    #
    #     Returns:
    #         list[:obj:`Det3DDataSample`]: Detection results of the
    #         input sample. Each Det3DDataSample usually contain
    #         'pred_instances_3d'. And the ``pred_instances_3d`` usually
    #         contains following keys.
    #
    #         - scores_3d (Tensor): Classification scores, has a shape
    #             (num_instances, )
    #         - labels_3d (Tensor): Labels of bboxes, has a shape
    #             (num_instances, ).
    #         - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
    #             contains a tensor with shape (num_instances, 7).
    #     """
    #     batch_input_metas = [item.metainfo for item in batch_data_samples]
    #     img_feats, pts_feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
    #     if pts_feats and self.with_pts_bbox:
    #         results_list_3d = self.pts_bbox_head.predict(
    #             pts_feats, batch_data_samples, **kwargs
    #         )
    #     else:
    #         results_list_3d = None
    #
    #     if img_feats and self.with_img_bbox:
    #         # TODO check this for camera modality
    #         results_list_2d = self.predict_imgs(img_feats, batch_data_samples, **kwargs)
    #     else:
    #         results_list_2d = None
    #
    #     detsamples = self.add_pred_to_datasample(
    #         batch_data_samples, results_list_3d, results_list_2d
    #     )
    #     return detsamples
    #
    ### START OLD ###
    # len_queue is new?
    def extract_img_feat(
        self, img: Tensor, input_metas: List[dict], len_queue: Optional[int]
    ) -> dict:
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(
                    img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W)
                )
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(
        self,
        batch_inputs_dict: dict,
        batch_input_metas: List[dict],
        len_queue: Optional[int],
    ) -> tuple:
        """Extract features from images and points.

        Args:
            batch_inputs_dict (dict): Dict of batch inputs. It
                contains

                - points (List[tensor]):  Point cloud of multiple inputs.
                - imgs (tensor): Image tensor with shape (B, C, H, W).
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.

        Returns:
             tuple: Two elements in tuple arrange as
             image features and point cloud features.
        """

        # @auto_fp16(apply_to=("img"))

        img_feats = self.extract_img_feat(
            batch_inputs_dict["img"], batch_input_metas, len_queue=len_queue
        )

        return img_feats

    def forward_pts_train(
        self,
        pts_feats,
        gt_bboxes_3d,
        gt_labels_3d,
        img_metas,
        gt_bboxes_ignore=None,
        prev_bev=None,
    ):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        print(list((k, type(v)) for k, v in kwargs.items()))

        # print("inputs: ", list(kwargs["inputs"].keys()))
        # print("data_samples: ", kwargs["data_samples"][0])
        print("mode: ", kwargs["mode"])
        # assert(000)

        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated."""
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]["prev_bev_exists"]:
                    prev_bev = None
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True
                )
            self.train()
            return prev_bev

    # @auto_fp16(apply_to=("img", "points"))
    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
        img_depth=None,
        img_mask=None,
    ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue - 1] for each in img_metas]
        if not img_metas[0]["prev_bev_exists"]:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(
            img_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore, prev_bev
        )

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            # the first sample of each scene is truncated
            self.prev_frame_info["prev_bev"] = None
        # update idx
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])
        if self.prev_frame_info["prev_bev"] is not None:
            img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        else:
            img_metas[0][0]["can_bus"][-1] = 0
            img_metas[0][0]["can_bus"][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info["prev_bev"], **kwargs
        )
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = new_prev_bev
        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs["bev_embed"], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return new_prev_bev, bbox_list
