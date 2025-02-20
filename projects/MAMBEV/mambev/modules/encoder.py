from typing import Any, Dict, Optional, Sequence

from mmdet3d.registry import TASK_UTILS
from mmengine.runner import autocast
import torch
import torch.nn as nn
from einops import rearrange, repeat
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from torch import Tensor
from torch.nn.init import normal_

from projects.MAMBEV.mambev.utils.reference_points_gen import (
    get_reference_points_2d,
    get_reference_points_3d,
    point_sampling,
)
from projects.MAMBEV.mambev.layers.global_traversal_layer import (
    GlobalTraversalConstructor,
)
from projects.mmcv_dep.transformer import TransformerLayerSequence
from mmdet.models.layers import SinePositionalEncoding
# from mmdet3d.datasets.transforms import GlobalRotScaleTrans


@MODELS.register_module()
class PerceptionTransformerBEVEncoderV3(BaseModule):
    def __init__(
        self,
        encoder: Dict[str, Any],
        num_feature_levels: int,
        num_cams: int,
        embed_dims: int,
        use_cams_embeds: bool = True,
        use_pos_encoding: bool = False,
        global_traversal: Optional[Dict] = None,
        **kwargs,
    ):
        super(PerceptionTransformerBEVEncoderV3, self).__init__(**kwargs)
        self.encoder: BEVFormerEncoderV3 = MODELS.build(encoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.use_cams_embeds = use_cams_embeds
        self.use_pos_encoding = use_pos_encoding

        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        if self.use_cams_embeds:
            self.cams_embeds = nn.Parameter(
                torch.Tensor(self.num_cams, self.embed_dims)
            )
        if self.use_pos_encoding:
            self.pos_enc = SinePositionalEncoding(self.embed_dims // 2)
        if global_traversal:
            self.traversal_generator: GlobalTraversalConstructor = TASK_UTILS.build(
                global_traversal
            )
        self.global_traversal = bool(global_traversal)
        self.init_weights()

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, BEVFormerEncoderV3):
                m.init_weights()
        normal_(self.level_embeds)
        if self.use_cams_embeds:
            normal_(self.cams_embeds)

    def forward(
        self,
        mlvl_feats: Sequence[Tensor],
        bev_queries: Tensor,
        bev_h: int,
        bev_w: int,
        bev_pos: Tensor,
        prev_bev: Optional[Tensor] = None,
        **kwargs,
    ):
        """
        obtain bev features.
        """
        bs = mlvl_feats[0].size(0)
        bev_queries = repeat(bev_queries, "n d -> n bs d", bs=bs)
        bev_pos = rearrange(bev_pos, "bs dim h w -> (h w) bs dim")

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            if self.use_pos_encoding:
                feat = feat + self.pos_enc(None, feat)
            feat = rearrange(feat, "bs num_cam c h w -> bs num_cam (h w) c")
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[None, :, None, :].to(feat.dtype)
            ### IF YOU ENCONTER AN ERROR HERE ITS BECAUSE YOU DIDNT
            ### PASS IN THE EXPECTED NUMBER OF FEATURE MAP LEVELS
            feat = feat + self.level_embeds[None, None, lvl : lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        if self.global_traversal:
            # TODO: construct value input from traversal indices
            # feat_flatten = self.combine_sequences(feat_flatten, traversal_indices)
            feat_flatten, indices, seq_lens, spatial_shapes = self.traversal_generator(
                feat_flatten, spatial_shapes
            )
        else:
            feat_flatten = torch.cat(feat_flatten, 2)
            indices, seq_lens = None, None

        bev_embed = self.encoder(
            query_pos=None,
            key_pos=None,
            attn_masks=None,
            query_key_padding_mask=None,
            key_padding_mask=None,
            mask=None,
            bev_query=bev_queries,
            key=feat_flatten,
            value=feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=None,
            shift=bev_queries.new_tensor([0, 0]).unsqueeze(0),
            value_map_indices=indices,
            value_seqlens=seq_lens,
            **kwargs,
        )
        with autocast(dtype=torch.float32):
            # rotate current bev to final aligned
            prev_bev = bev_embed
            try:
                assert (
                    "GlobalRotScaleTransImage_param" in kwargs["img_metas"][0].aug_param
                )
                assert prev_bev is not None
                *_, bda_mat, only_gt = kwargs["img_metas"][0].aug_param[
                    "GlobalRotScaleTransImage_param"
                ]
                prev_bev = rearrange(
                    prev_bev,
                    "bs (bevh bevw) c -> bs c bevh bevw",
                    bevh=bev_h,
                    bevw=bev_w,
                )
                # prev_bev = prev_bev.reshape(bs, bev_h, bev_w, -1).permute(
                #     0, 3, 1, 2
                # )  # bchw
                if only_gt:
                    # rot angle
                    # prev_bev = torchvision.transforms.functional.rotate(prev_bev, -30, InterpolationMode.BILINEAR)
                    grid = get_reference_points_2d(
                        bs, bev_h, bev_w, bev_queries.device, bev_queries.dtype
                    )
                    grid = rearrange(
                        grid, "bs (h w) e n -> bs h w n e", h=bev_h, w=bev_w
                    )
                    grid_shift = grid * 2.0 - 1.0
                    # grid_shift = grid_shift.unsqueeze(0).unsqueeze(-1)
                    bda_mat = repeat(
                        bda_mat[:2, :2],
                        "n m -> bs bevh bevw n m",
                        bs=bs,
                        bevh=bev_h,
                        bevw=bev_w,
                    )

                    grid_shift = torch.matmul(bda_mat, grid_shift).squeeze(-1)
                    prev_bev = torch.nn.functional.grid_sample(
                        prev_bev,
                        grid_shift,
                        align_corners=False,
                    )

                prev_bev = prev_bev.reshape(bs, -1, bev_h * bev_w)
                prev_bev = prev_bev.permute(0, 2, 1)

            except (AssertionError, KeyError, AttributeError):
                pass

        return prev_bev


@MODELS.register_module()
class BEVFormerEncoderV3(TransformerLayerSequence):
    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(
        self,
        *args,
        pc_range,
        num_points_in_pillar: int,
        return_intermediate: bool = False,
        dataset_type="nuscenes",
        **kwargs,
    ):
        super(BEVFormerEncoderV3, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range

    def forward(
        self,
        bev_query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *args,
        bev_h: int,
        bev_w: int,
        bev_pos: torch.Tensor,
        spatial_shapes=None,
        level_start_index: Optional[int] = None,
        valid_ratios=None,
        prev_bev: Optional[torch.Tensor] = None,
        shift: torch.Tensor = torch.Tensor([[0.0]]),
        **kwargs,
    ):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = bev_query
        intermediate = []

        ref_3d = get_reference_points_3d(
            H=bev_h,
            W=bev_w,
            Z=self.pc_range[5] - self.pc_range[2],
            D=self.num_points_in_pillar,
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype,
        )
        ref_2d = get_reference_points_2d(
            bs=bev_query.size(1),
            H=bev_h,
            W=bev_w,
            device=bev_query.device,
            dtype=bev_query.dtype,
        )

        reference_points_cam, bev_mask = point_sampling(
            reference_points=ref_3d,
            pc_range=self.pc_range,
            img_metas=kwargs["img_metas"],
        )

        # NOTE: remove clone to reproduce BEVFormerV1 Results
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = rearrange(bev_query, "q bs d -> bs q d")
        bev_pos = rearrange(bev_pos, "q bs d -> bs q d")
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        # NOTE: prev_bev is always None in BEVFormerV2
        if prev_bev is not None:
            prev_bev = rearrange(prev_bev, "q bs d -> bs q d")
            prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(
                bs * 2, len_bev, -1
            )
            hybrid_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2
            )
        else:
            hybrid_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2
            )

        for layer in self.layers:
            output = layer(
                query=bev_query,
                key=key,
                value=value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybrid_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs,
            )

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output
