import copy
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

from einops.packing import Shape
from mmdet3d.structures import limit_period
from mmdet3d.structures.det3d_data_sample import InstanceData
from mmengine.model import BaseModule
import torch
import torch.nn as nn
from mmdet.models.layers.transformer import inverse_sigmoid
from mmdet.models.utils import multi_apply
from mmdet.utils import InstanceList, OptInstanceList
from mmengine.config import Config
from mmengine.logging import print_log
from mmengine.registry import MODELS
from mmengine.runner import autocast
from torch import Tensor

from projects.DETR3D.detr3d import DETR3DHead


@MODELS.register_module()
class DummyPositionalEncoding(BaseModule):
    def __init__(self, init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg)

    def forward(self, x):
        return x.unsqueeze(1)


@MODELS.register_module(force=True)
class BEVFormerHeadV3(DETR3DHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(
        self,
        *args,
        transformer: Union[Dict, Config],
        bev_h: int,
        bev_w: int,
        encoder_self_attn: bool = True,
        depr_bbox_format: bool = False,
        **kwargs,
    ):
        self.bev_h = bev_h
        self.bev_w = bev_w
        super(BEVFormerHeadV3, self).__init__(*args, transformer=transformer, **kwargs)  # type: ignore
        if not encoder_self_attn:
            self.positional_encoding = DummyPositionalEncoding()
        self.real_w: float = self.pc_range[3] - self.pc_range[0]
        self.real_h: float = self.pc_range[4] - self.pc_range[1]
        self.depr_bbox_format = depr_bbox_format
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (
            (self.transformer.decoder.num_layers + 1)
            if self.as_two_stage
            else self.transformer.decoder.num_layers
        )

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)

    @autocast(dtype=torch.float32)
    def loss_by_feat(
        self,
        batch_gt_instances_3d: InstanceList,
        preds_dicts: Dict[str, Tensor],
        batch_gt_instances_3d_ignore: OptInstanceList = None,
    ) -> Dict:
        return super().loss_by_feat(
            batch_gt_instances_3d, preds_dicts, batch_gt_instances_3d_ignore
        )

    def predict_by_feat(self, preds_dicts, img_metas, rescale=False) -> InstanceList:
        """Transform network output for a batch into bbox predictions.

        Args:
            preds_dicts (Dict[str, Tensor]):
                -all_cls_scores (Tensor): Outputs from the classification head,
                    shape [nb_dec, bs, num_query, cls_out_channels]. Note
                    cls_out_channels should includes background.
                -all_bbox_preds (Tensor): Sigmoid outputs from the regression
                    head with normalized coordinate format
                    (cx, cy, l, w, cz, h, rot_sine, rot_cosine, v_x, v_y).
                    Shape [nb_dec, bs, num_query, 10].
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                  (num_instances, C), where C >= 7.
        """
        if not self.depr_bbox_format:
            return super().predict_by_feat(preds_dicts, img_metas, rescale)
        # sinθ & cosθ ---> θ
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)  # batch size
        ret_list = []
        for i in range(num_samples):
            results = InstanceData()
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            b_copys = bboxes.clone()
            bboxes[:, 3] = b_copys[:, 4]
            bboxes[:, 4] = b_copys[:, 3]
            bboxes[:, 6] = limit_period(
                -b_copys[:, 6] - torch.pi / 2, period=torch.pi * 2
            )
            del b_copys
            bboxes = img_metas[i]["box_type_3d"](bboxes, self.code_size - 1)

            results.bboxes_3d = bboxes
            results.scores_3d = preds["scores"]
            results.labels_3d = preds["labels"]
            ret_list.append(results)
        return ret_list

    @autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    def forward(
        self,
        mlvl_feats: Sequence[Tensor],
        img_metas: List[Dict[str, Any]],
        prev_bev: Optional[Tensor] = None,
        only_bev: bool = False,
        mode: str = "training",
        **kwargs,
    ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        object_query_embeds = self.query_embedding.weight
        if mode == "training":
            return self._forward(
                object_query_embeds, mlvl_feats, img_metas, prev_bev, only_bev, **kwargs
            )
        else:
            return self._forward_test(
                object_query_embeds, mlvl_feats, img_metas, **kwargs
            )

    def _forward_test(
        self,
        object_query_embeds: torch.Tensor,
        mlvl_feats: Sequence[Tensor],
        img_metas: List[Dict[str, Any]],
        unpack_shapes: List[Shape],
        unpack_metas: List[int],
        **kwargs,
    ):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight

        bev_mask = torch.zeros(
            (bs, self.bev_h, self.bev_w), device=bev_queries.device, dtype=dtype
        )
        bev_pos = self.positional_encoding(bev_mask)
        outputs = self.transformer.forward_test(
            mlvl_feats=mlvl_feats,
            bev_queries=bev_queries,
            object_query_embed=object_query_embeds,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            unpack_shapes=unpack_shapes,
            unpack_metas=unpack_metas,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
        )

        bev_embed, hs, init_reference, inter_references = outputs
        with autocast(dtype=torch.float32):
            hs = hs.permute(0, 2, 1, 3)
            outputs_classes = []
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self.cls_branches[lvl](hs[lvl])
                tmp = self.reg_branches[lvl](hs[lvl])

                # TODO: check the shape of reference
                assert reference.shape[-1] == 3
                tmp[..., 0:2] += reference[..., 0:2]
                tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
                tmp[..., 4:5] += reference[..., 2:3]
                tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
                tmp[..., 0:1] = (
                    tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0])
                    + self.pc_range[0]
                )
                tmp[..., 1:2] = (
                    tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1])
                    + self.pc_range[1]
                )
                tmp[..., 4:5] = (
                    tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2])
                    + self.pc_range[2]
                )

                # TODO: check if using sigmoid
                outputs_coord = tmp
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)

            outputs_classes = torch.stack(outputs_classes)
            outputs_coords = torch.stack(outputs_coords)

            outs = {
                "bev_embed": bev_embed.float(),
                "all_cls_scores": outputs_classes.float(),
                "all_bbox_preds": outputs_coords.float(),
                "enc_cls_scores": None,
                "enc_bbox_preds": None,
            }

            return outs

    def _forward(
        self,
        object_query_embeds: torch.Tensor,
        mlvl_feats: Sequence[Tensor],
        img_metas: List[Dict[str, Any]],
        prev_bev: Optional[Tensor] = None,
        only_bev: bool = False,
        **kwargs,
    ):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight

        bev_mask = torch.zeros(
            (bs, self.bev_h, self.bev_w), device=bev_queries.device, dtype=dtype
        )
        bev_pos = self.positional_encoding(bev_mask)
        # only use encoder to obtain BEV features,
        if only_bev:
            # TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats=mlvl_feats,
                bev_queries=bev_queries,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats=mlvl_feats,
                bev_queries=bev_queries,
                object_query_embed=object_query_embeds,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )

        bev_embed, hs, init_reference, inter_references = outputs
        with autocast(dtype=torch.float32):
            hs = hs.permute(0, 2, 1, 3)
            outputs_classes = []
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self.cls_branches[lvl](hs[lvl])
                tmp = self.reg_branches[lvl](hs[lvl])

                # TODO: check the shape of reference
                assert reference.shape[-1] == 3
                tmp[..., 0:2] += reference[..., 0:2]
                tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
                tmp[..., 4:5] += reference[..., 2:3]
                tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
                tmp[..., 0:1] = (
                    tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0])
                    + self.pc_range[0]
                )
                tmp[..., 1:2] = (
                    tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1])
                    + self.pc_range[1]
                )
                tmp[..., 4:5] = (
                    tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2])
                    + self.pc_range[2]
                )

                # TODO: check if using sigmoid
                outputs_coord = tmp
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)

            outputs_classes = torch.stack(outputs_classes)
            outputs_coords = torch.stack(outputs_coords)

            outs = {
                "bev_embed": bev_embed.float(),
                "all_cls_scores": outputs_classes.float(),
                "all_bbox_preds": outputs_coords.float(),
                "enc_cls_scores": None,
                "enc_bbox_preds": None,
            }

            return outs


@MODELS.register_module(force=True)
class BEVFormerHead_GroupDETRV3(BEVFormerHeadV3):
    def __init__(self, *args, group_detr=11, **kwargs):
        self.group_detr = group_detr
        assert "num_query" in kwargs
        kwargs["num_query"] = group_detr * kwargs["num_query"]
        super().__init__(*args, **kwargs)

    def forward(  # type: ignore
        self,
        mlvl_feats: List[torch.Tensor],
        img_metas: List[Dict[str, Any]],
        prev_bev: Optional[torch.Tensor] = None,
        only_bev: bool = False,
        mode: str = "training",
        **kwargs,
    ):
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        object_query_embeds = (
            object_query_embeds[: self.num_query // self.group_detr]
            if not self.training
            else object_query_embeds
        )
        if mode != "training":  # NOTE: Only difference to bevformer head
            return self._forward_test(
                object_query_embeds, mlvl_feats, img_metas, **kwargs
            )
        else:
            return self._forward(
                object_query_embeds, mlvl_feats, img_metas, prev_bev, only_bev, **kwargs
            )
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros(
            (bs, self.bev_h, self.bev_w), device=bev_queries.device, dtype=dtype
        )
        bev_pos = self.positional_encoding(bev_mask)

        # self.transformer: PerceptionTransformerV3
        if only_bev:
            return self.transformer.get_bev_features(
                mlvl_feats=mlvl_feats,
                bev_queries=bev_queries,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats=mlvl_feats,
                bev_queries=bev_queries,
                object_query_embed=object_query_embeds,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )

        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        if num_nans := torch.isnan(hs).sum():
            print_log(
                f"Number of nans in prediction outputs: {num_nans}",
                "current",
                logging.WARN,
            )

        hs = torch.nan_to_num(hs)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (
                tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            )
            tmp[..., 1:2] = (
                tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            )
            tmp[..., 4:5] = (
                tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            )
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
        }

        return outs

    def loss_by_feat(
        self,
        batch_gt_instances_3d: InstanceList,
        preds_dicts: Dict[str, Tensor],
        batch_gt_instances_3d_ignore: OptInstanceList = None,
    ) -> Dict:
        """Compute loss of the head.

        Args:
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d.  It usually includes ``bboxes_3d``、`
                `labels_3d``、``depths``、``centers_2d`` and attributes.
                gt_instance.  It usually includes ``bboxes``、``labels``.
            batch_gt_instances_3d_ignore (list[:obj:`InstanceData`], Optional):
                NOT supported.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert batch_gt_instances_3d_ignore is None, (
            f"{self.__class__.__name__} only supports "
            f"for batch_gt_instances_3d_ignore setting to None."
        )
        all_cls_scores = preds_dicts["all_cls_scores"]  # num_dec,bs,num_q,num_cls
        all_bbox_preds = preds_dicts["all_bbox_preds"]  # num_dec,bs,num_q,10
        enc_cls_scores = preds_dicts["enc_cls_scores"]
        enc_bbox_preds = preds_dicts["enc_bbox_preds"]

        # calculate loss for each decoder layer
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_3d_list = [
            batch_gt_instances_3d for _ in range(num_dec_layers)
        ]

        num_query_per_group = self.num_query // self.group_detr
        loss_dict = {}
        for group_index in range(
            0, self.group_detr * num_query_per_group, num_query_per_group
        ):
            # for batches of size > 1
            # if (
            #     isinstance(batch_gt_instances_3d, list)
            #     and len(batch_gt_instances_3d) > 1
            # ):
            #     group_cls_scores = [
            #         instances[:, :, group_index : group_index + num_query_per_group, :]
            #         for instances in all_cls_scores
            #     ]
            #     group_bbox_preds = [
            #         instances[:, :, group_index : group_index + num_query_per_group, :]
            #         for instances in all_bbox_preds
            #     ]
            # else:
            group_cls_scores = all_cls_scores[
                :, :, group_index : group_index + num_query_per_group, :
            ]
            group_bbox_preds = all_bbox_preds[
                :, :, group_index : group_index + num_query_per_group, :
            ]

            losses_cls, losses_bbox = multi_apply(
                self.loss_by_feat_single,
                group_cls_scores,
                group_bbox_preds,
                batch_gt_instances_3d_list,
            )
            try:
                loss_dict["loss_cls"] += losses_cls[-1] / self.group_detr
                loss_dict["loss_bbox"] += losses_bbox[-1] / self.group_detr
            except KeyError:
                loss_dict["loss_cls"] = losses_cls[-1] / self.group_detr
                loss_dict["loss_bbox"] = losses_bbox[-1] / self.group_detr
            # loss from other decoder layers
            for i, (loss_cls_i, loss_bbox_i) in enumerate(
                zip(losses_cls[:-1], losses_bbox[:-1])
            ):
                try:
                    loss_dict[f"d{i}.loss_cls"] += loss_cls_i / self.group_detr
                    loss_dict[f"d{i}.loss_bbox"] += loss_bbox_i / self.group_detr
                except KeyError:
                    loss_dict[f"d{i}.loss_cls"] = loss_cls_i / self.group_detr
                    loss_dict[f"d{i}.loss_bbox"] = loss_bbox_i / self.group_detr
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # this may not work if batch_gt_instances_3d_list is a list
            enc_loss_cls, enc_losses_bbox = self.loss_by_feat_single(
                enc_cls_scores,
                enc_bbox_preds,
                batch_gt_instances_3d_list,  # type:ignore
            )
            loss_dict["enc_loss_cls"] = enc_loss_cls
            loss_dict["enc_loss_bbox"] = enc_losses_bbox

        return loss_dict


if __name__ == "__main__":
    from pickle import Unpickler
    import projects.DETR3D
    import projects.MAMBEV

    # from torchviz import make_dot
    import io

    class TorchUnpickler(Unpickler):
        def __init__(self, file, device="cpu"):
            super(TorchUnpickler, self).__init__(file)
            self.device = device

        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(io.BytesIO(b), map_location=self.device)
            return super().find_class(module, name)

    device = "cuda"
    bev_h_ = bev_w_ = 50
    voxel_size = [102.4 / bev_h_, 102.4 / bev_w_, 8]
    _dim_ = 256
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    _pos_dim_ = _dim_ // 2
    _ffn_dim_ = _dim_ * 2
    _points_in_pillar_ = 4
    out_indices = (3,)
    in_channels = [2048]
    assert (
        len(out_indices) == len(in_channels)
    ), "Number of outputs from Img Backbone do not match the number of Inputs expected by the Img Neck"
    _img_feat_levels_ = 1
    _mono_img_feat_levels_ = 1
    _img_neck_outs_ = max(_img_feat_levels_, _mono_img_feat_levels_)
    _encoder_layers_ = 1
    _decoder_layers_ = 1
    _mamba_layers_ = 1
    traversal_methods = ["tl0snake"]
    frames = [0]
    mamba2_cfg = dict(
        type="mmdet3d.SimpleXAMamba2",
        traversal_methods=traversal_methods,
        d_model=256,
        d_state=128,
        d_conv=4,
        expand=2,
        headdim=64,
        # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        d_ssm=None,
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        construct_inside=True,
        chunk_size=128,
        use_mem_eff_path=True,
        sequence_parallel=True,
        device=torch.device(device),
        dtype=torch.float32,
    )

    pts_bbox_head = dict(
        type="BEVFormerHeadV3",
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type="PerceptionTransformerV3",
            embed_dims=_dim_,
            num_cams=6,
            num_fusion=3,
            num_feature_levels=4,
            frames=[0],
            use_cams_embeds=True,
            # rotate_prev_bev=True,
            # # use_shift=True,
            encoder=dict(
                type="BEVFormerEncoderV3",
                num_layers=1,
                pc_range=point_cloud_range,
                num_points_in_pillar=_points_in_pillar_,
                return_intermediate=False,
                transformerlayers=dict(
                    type="BEVFormerLayerV3",
                    attn_cfgs=[
                        mamba2_cfg,
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=[
                        "cross_attn",
                    ],
                    batch_first=True,
                ),
            ),
            decoder=dict(
                type="DetectionTransformerDecoder",
                num_layers=_decoder_layers_,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DeformableDetrTransformerDecoderLayer",
                    self_attn_cfg=dict(
                        # group_detr = group_detr,
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1,
                        batch_first=True,
                    ),
                    cross_attn_cfg=dict(
                        embed_dims=_dim_,
                        num_heads=8,
                        num_levels=1,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.1,
                        batch_first=True,
                        value_proj_ratio=1.0,
                    ),
                    ffn_cfg=dict(
                        embed_dims=_dim_,
                        feedforward_channels=_ffn_dim_,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                ),
            ),
        ),
        bbox_coder=dict(
            type="mmdet3d.NMSFreeCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10,
        ),
        positional_encoding=dict(
            type="mmdet.LearnedPositionalEncoding",
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        loss_cls=dict(
            type="mmdet.FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        # loss_bbox=dict(type="mmdet.SmoothL1Loss", loss_weight=0.75, beta=1.0),
        loss_bbox=dict(type="mmdet.L1Loss", loss_weight=0.25),
        loss_iou=dict(type="mmdet.GIoULoss", loss_weight=0.0),
        train_cfg=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="mmdet3d.HungarianAssigner3D",
                cls_cost=dict(type="mmdet.FocalLossCost", weight=2.0),
                # reg_cost=dict(type="mmdet3d.SmoothL1Cost", weight=0.75),
                reg_cost=dict(type="mmdet3d.BBox3DL1Cost", weight=0.25),
                pc_range=point_cloud_range,
                # Fake cost. This is just to make it compatible with DETR head.
                iou_cost=dict(type="mmdet.IoUCost", weight=0.0),
            ),
        ),
    )

    model = MODELS.build(pts_bbox_head)  # type:ignore
    model.to(device)
    input_path = "/home/jackmorris/BEVFormer/head_inputs.pkl"
    with open(input_path, "rb") as f:
        unpickler = TorchUnpickler(f, device)
        data = unpickler.load()

    # data["only_bev"] = True
    output = model(**data)
    # make_dot(
    #     output.mean(),
    #     params=dict(model.named_parameters()),
    #     show_attrs=True,
    #     show_saved=True,
    # ).render(
    #     "construct_inside_head__full_detail",
    #     directory="/home/jackmorris/BEVFormer/",
    #     format="png",
    # )
    # make_dot(
    #     output["all_cls_scores"].mean() + output["all_bbox_preds"].mean(),
    #     params=dict(model.named_parameters()),
    #     show_attrs=True,
    #     show_saved=True,
    # ).render(
    #     "construct_inside_head__full_detail",
    #     directory="/home/jackmorris/BEVFormer/",
    #     format="png",
    # )
    print(output.shape)
