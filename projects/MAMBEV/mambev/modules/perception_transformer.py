from collections import OrderedDict
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from einops import rearrange, unpack
from einops.packing import Shape
from mmdet3d.structures import Det3DDataSample
from mmengine.runner import autocast
import torch
import torch.nn as nn
from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList, xavier_init
from mmengine.registry import MODELS
from torch import Tensor

from .resnet_fusion import ResNetFusionV3


@MODELS.register_module()
class PerceptionTransformerV3(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(
        self,
        encoder: Dict[str, Any],
        decoder: Dict[str, Any],
        num_feature_levels: int,
        num_cams: int,
        embed_dims: int,
        use_cams_embeds: bool,
        frames: Tuple[int],
        num_fusion: int = 3,
        inter_channels: Optional[int] = None,
        global_traversal: Optional[Dict] = None,
        **kwargs,
    ):
        """Initialize layers of the Detr3DTransformer."""
        super(PerceptionTransformerV3, self).__init__()
        enc_cfg = dict(
            type="PerceptionTransformerBEVEncoderV3",
            encoder=encoder,
            num_feature_levels=num_feature_levels,
            num_cams=num_cams,
            embed_dims=embed_dims,
            use_cams_embeds=use_cams_embeds,
            global_traversal=global_traversal,
        )
        self.encoder = MODELS.build(enc_cfg)
        self.decoder = MODELS.build(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.use_cams_embeds = use_cams_embeds
        # decoder reference points
        try:
            self.reference_points = nn.Linear(self.embed_dims, kwargs["num_ref_points"])
        except KeyError:
            self.reference_points = nn.Linear(self.embed_dims, 3)
        self.frames = frames
        if len(self.frames) > 1:
            self.fusion = ResNetFusionV3(
                len(self.frames) * self.embed_dims,
                self.embed_dims,
                inter_channels
                if inter_channels is not None
                else len(self.frames) * self.embed_dims,
                num_fusion,
            )
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Could replace with this?
        # init_cfg = [
        #     dict(
        #         type="Xavier",
        #         layer=[
        #             "Linear",
        #         ],
        #         distribution="uniform",
        #     )
        # ]
        """Initialize the transformer weights."""

        print_log(
            "Initializing weights in perception transformer",
            "current",
            level=logging.INFO,
        )
        super().init_weights()
        xavier_init(self.reference_points, distribution="uniform", bias=0)

    def get_bev_features(
        self,
        mlvl_feats: Sequence[Tensor],
        bev_queries: Tensor,
        bev_h: int,
        bev_w: int,
        bev_pos: Tensor,
        grid_length: Tuple[float, float] = (0.512, 0.512),
        prev_bev: Optional[Tensor] = None,
        **kwargs,
    ):
        return self.encoder(
            mlvl_feats=mlvl_feats,
            bev_queries=bev_queries,
            bev_h=bev_h,
            bev_w=bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs,
        )

    def forward_test(
        self,
        mlvl_feats: Sequence[Tensor],
        bev_queries: Tensor,
        bev_h: int,
        bev_w: int,
        bev_pos: Tensor,
        unpack_shapes: List[Shape],
        unpack_metas: List[int],
        object_query_embed: Tensor,
        img_metas: List[Dict[int, Det3DDataSample]],
        grid_length: Tuple[float, float] = (0.512, 0.512),
        # prev_bev: Optional[Dict[int, Tuple[Tensor, List[bool]]]] = None,
        reg_branches: Optional[ModuleList] = None,
        cls_branches: Optional[ModuleList] = None,
        **kwargs,
    ):
        enc_metas = []
        for t in unpack_metas:
            enc_metas.extend([meta for meta in img_metas[t]])  # type:ignore
        bev_embed = self.encoder(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=None,
            img_metas=enc_metas,
            **kwargs,
        )

        prev_bev = OrderedDict({i: tuple() for i in self.frames})
        # print(bev_embed.shape)
        bev = unpack(bev_embed, packed_shapes=unpack_shapes, pattern="* L C")
        batch_size = bev[0].shape[0]

        for t in unpack_metas:
            if t == 0:
                prev_bev[t] = bev[t], [False] * batch_size
            else:
                prev_bev[t] = bev[t], [meta.mask for meta in img_metas[t]]  # type:ignore

        if len(self.frames) > 1:
            assert prev_bev is not None
            # cur_ind = list(self.frames).index(0)
            assert len(prev_bev) == len(self.frames) and isinstance(prev_bev, dict)

            for bn in range(batch_size):
                # fill prev frame feature
                for i in range(-1, min(self.frames), -1):
                    # print(len(prev_bev[i]))
                    if prev_bev[i][1][bn]:
                        prev_bev[i][0][bn] = prev_bev[i + 1][0][bn].detach()
            bev_embed = [
                rearrange(
                    x[0],  # select the tensor not the mask
                    "bs (bev_h bev_w) dim -> bs dim bev_h bev_w ",
                    bev_h=bev_h,
                    bev_w=bev_w,
                ).contiguous()
                for x in prev_bev.values()
            ]
            # print(len(bev_embed), bev_embed[0].shape, self.frames)
            bev_embed = self.fusion(bev_embed)

        # bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
        query = query.unsqueeze(0).expand(batch_size, -1, -1)
        with autocast(dtype=torch.float32):
            reference_points = self.reference_points(query_pos)
            reference_points = reference_points.sigmoid()
        ### TESTING DETR3D HEAD
        init_reference_out = reference_points
        # print_log({k: v.shape for k, v in inp_dict.items()}, "current", logging.DEBUG)
        inter_states, inter_references = self.decoder(
            query=query,
            key=bev_embed,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs,
        )

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out

    def forward(
        self,
        mlvl_feats: Sequence[Tensor],
        bev_queries: Tensor,
        bev_h: int,
        bev_w: int,
        bev_pos: Tensor,
        object_query_embed: Tensor,
        grid_length: Tuple[float, float] = (0.512, 0.512),
        prev_bev: Optional[Dict[int, Tuple[Tensor, List[bool]]]] = None,
        reg_branches: Optional[ModuleList] = None,
        cls_branches: Optional[ModuleList] = None,
        **kwargs,
    ):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        bev_embed = self.encoder(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=None,
            **kwargs,
        )  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        batch_size = bev_embed.shape[0]

        if len(self.frames) > 1:
            assert prev_bev is not None
            # cur_ind = list(self.frames).index(0)
            assert prev_bev[0] is None and len(prev_bev) == len(self.frames)
            prev_bev[0] = bev_embed, [False] * batch_size

            assert isinstance(prev_bev, dict)
            for bn in range(batch_size):
                # fill prev frame feature
                for i in range(-1, min(self.frames), -1):
                    if prev_bev[i][1][bn]:
                        prev_bev[i][0][bn] = prev_bev[i + 1][0][bn].detach()
                # # fill next frame feature
                # for i in range(cur_ind + 1, len(self.frames)):
                #     if prev_bev[i][1][bn]:
                #         prev_bev[i][0][bn] = prev_bev[i - 1][0][bn].detach()
                #     # if prev_bev[i] is None:
                #     #     prev_bev[i] = prev_bev[i - 1].detach()

            bev_embed = [
                rearrange(
                    x[0],  # select the tensor not the mask
                    "bs (bev_h bev_w) dim -> bs dim bev_h bev_w ",
                    bev_h=bev_h,
                    bev_w=bev_w,
                ).contiguous()
                for x in prev_bev.values()
            ]
            bev_embed = self.fusion(bev_embed)

        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
        query = query.unsqueeze(0).expand(batch_size, -1, -1)
        with autocast(dtype=torch.float32):
            reference_points = self.reference_points(query_pos)
            reference_points = reference_points.sigmoid()
        ### TESTING DETR3D HEAD
        init_reference_out = reference_points
        # print_log({k: v.shape for k, v in inp_dict.items()}, "current", logging.DEBUG)
        inter_states, inter_references = self.decoder(
            query=query,
            key=bev_embed,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs,
        )

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out
