# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
#  Modified by Jack Morris
# ---------------------------------------------
"""Consider replacing with XFormers"""

import copy
import logging
import inspect
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from mmcv.cnn import build_norm_layer
from mmdet.models.layers import DeformableDetrTransformerDecoderLayer as DeformableLayer
from mmcv.cnn.bricks.transformer import FFN
from mmcv.ops import MultiScaleDeformableAttention
from mmengine import ConfigDict
from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.registry import MODELS
from mmengine.runner import autocast
from torch import Tensor
from projects.BEVFormer.bevformer.bevformer.modules.group_attention import (
    GroupMultiheadAttention,
)
from einops import rearrange

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm

MODELS.register_module("RMS", module=RMSNorm)


def has_argument(func, argname):
    params = inspect.signature(func).parameters
    return argname in params


@MODELS.register_module()
class DeformableDetrTransformerDecoderLayer(DeformableLayer):
    def __init__(self, **kwargs) -> None:
        super(DeformableDetrTransformerDecoderLayer, self).__init__(**kwargs)
        self.pre_norm = False

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        key_pos: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,
        cross_attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs,
        )
        query = self.norms[0](query)
        with autocast(dtype=torch.float32):
            # assert query.dtype != torch.half
            # assert value.dtype != torch.half
            query = self.cross_attn(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_mask=cross_attn_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query.to(torch.float32)


@MODELS.register_module()
class MambaDeformableDetrTransformerDecoderLayer(DeformableLayer):
    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MODELS.build(self.self_attn_cfg)  # type:ignore
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)  # type:ignore
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)  # type:ignore
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]  # type:ignore
            for _ in range(3)
        ]
        self.pre_norm = False
        self.norms = ModuleList(norms_list)

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        key_pos: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,
        cross_attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        query = self.self_attn(
            u=query,
            query_pos=query_pos,
        )
        query = self.norms[0](query)
        with autocast(dtype=torch.float32):
            # assert query.dtype != torch.half
            # assert value.dtype != torch.half
            query = self.cross_attn(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_mask=cross_attn_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query.to(torch.float32)


@MODELS.register_module()
class GroupDeformableDetrTransformerDecoderLayer(DeformableDetrTransformerDecoderLayer):
    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = GroupMultiheadAttention(**self.self_attn_cfg)  # type:ignore
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)  # type:ignore
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)  # type:ignore
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]  # type:ignore
            for _ in range(3)  # type:ignore
        ]
        self.norms = ModuleList(norms_list)


@MODELS.register_module()
class CustomBaseTransformerLayer(BaseModule):
    """Base `TransformerLayer` for vision transformer.
    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(
        self,
        attn_cfgs: Union[Dict[str, Any], List[Dict[str, Any]]],
        operation_order: Tuple[str],
        ffn_cfgs: Dict[str, Any] = dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.0,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        norm_cfg: Dict[str, Any] = dict(type="RMS"),
        init_cfg: Optional[Dict[str, Any]] = None,
        batch_first: bool = True,
        layer_idx: Optional[int] = None,
        **kwargs,
    ):
        super(CustomBaseTransformerLayer, self).__init__(init_cfg)

        self.layer_idx = layer_idx
        self.batch_first = batch_first
        num_attn = (
            operation_order.count("self_attn")
            + operation_order.count("cross_attn")
            + operation_order.count("mamba_self_attn")
            + operation_order.count("both_attn")
        )
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), (
                f"The length "
                f"of attn_cfg {num_attn} is "
                f"not consistent with the number of attention"
                f"in operation_order {operation_order}."
            )

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = ModuleList()

        for attn_idx, operation_name in enumerate(
            filter(
                lambda x: x
                in ["self_attn", "cross_attn", "mamba_self_attn", "both_attn"],
                operation_order,
            )
        ):
            if "batch_first" in attn_cfgs[attn_idx]:
                assert self.batch_first == attn_cfgs[attn_idx]["batch_first"]
            else:
                attn_cfgs[attn_idx]["batch_first"] = self.batch_first
            # model_cls = MODELS.get(attn_cfgs[index]["type"])
            # if has_argument(model_cls.__init__, "layer_idx"):
            #     attn_cfgs[index]["layer_idx"] = index
            # else:
            #     print(f"{type(model_cls)} does not take layer_idx as an argument")

            attn_cfgs[attn_idx]["layer_idx"] = self.layer_idx
            attn_cfgs[attn_idx]["attn_idx"] = attn_idx
            attention = MODELS.build(attn_cfgs[attn_idx])
            # Some custom attentions used as `self_attn`
            # or `cross_attn` can have different behavior.
            attention.operation_name = operation_name
            self.attentions.append(attention)

        assert isinstance(
            self.attentions[0].embed_dims, int
        ), "For type checking, (should) always pass"

        self.embed_dims: int = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count("ffn")
        if isinstance(ffn_cfgs, dict):
            layerwise_ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            layerwise_ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(layerwise_ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if "embed_dims" not in layerwise_ffn_cfgs[ffn_index]:
                layerwise_ffn_cfgs[ffn_index]["embed_dims"] = self.embed_dims
            else:
                assert (
                    layerwise_ffn_cfgs[ffn_index]["embed_dims"] == self.embed_dims
                ), "Linear Layers must all match embed_dims"

            self.ffns.append(MODELS.build(layerwise_ffn_cfgs[ffn_index]))

        self.norms = ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        key_pos: Optional[Tensor] = None,
        attn_masks: List[Optional[Tensor]] = [],
        query_key_padding_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        **kwargs,
    ):
        """Forward function for `TransformerDecoderLayer`.
        **kwargs contains some specific arguments of attentions.
        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `key`, with
                shape [bs, num_keys]. Default: None.
        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(
                f"Use same attn_mask in all attentions in "
                f"{self.__class__.__name__} "
            )
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            if layer == "self_attn":
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "mamba_self_attn":
                query = self.attentions[attn_index](query, query_pos=query_pos)
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


@MODELS.register_module()
class MamBEVLayer(CustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(
        self,
        attn_cfgs: Union[List[Dict[str, Any]], Dict[str, Any]],
        feedforward_channels: int,
        operation_order: Tuple[str],
        ffn_dropout: float = 0.0,
        norm_cfg: Dict[str, Any] = dict(type="LN"),
        act_cfg: Optional[Dict[str, Any]] = dict(type="ReLU", inplace=True),
        ffn_num_fcs: Optional[int] = 2,
        **kwargs,
    ):
        super(MamBEVLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs,
        )
        # assert len(operation_order) == 6
        # assert set(operation_order) == set(["self_attn", "norm", "cross_attn", "ffn"])

    def forward(  # type:ignore
        self,
        query: Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor],
        query_pos: Optional[Tensor],
        key_pos: Optional[Tensor],
        # attn_masks: List[Optional[Tensor]] = [],
        attn_masks: Optional[Union[Tensor, List[Optional[Tensor]]]],
        query_key_padding_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        bev_pos: Tensor,
        ref_2d: Tensor,
        ref_3d: Tensor,
        bev_h: int,
        bev_w: int,
        reference_points_cam: Tensor,
        mask: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        prev_bev: Optional[Tensor] = None,
        **kwargs,
    ):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(
                f"Use same attn_mask in all attentions in "
                f"{self.__class__.__name__} "
            )
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for i, layer in enumerate(self.operation_order):
            # temporal self attention
            if layer == "self_attn":
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "mamba_self_attn":
                query = self.attentions[attn_index](query, query_pos=bev_pos)
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention

            elif layer == "cross_attn":
                attn_full = (
                    self.attentions[attn_index].__class__.__name__
                    == "MultiheadAttention"
                )
                query = self.attentions[attn_index](
                    query=query,
                    key=key
                    if not attn_full
                    else rearrange(key, "bs nc nv c -> bs (nc nv) c"),
                    value=value
                    if not attn_full
                    else rearrange(value, "bs nc nv c -> bs (nc nv) c"),
                    identity=identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


@MODELS.register_module()
class BEVFormerLayerV3(CustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(
        self,
        attn_cfgs: Union[List[Dict[str, Any]], Dict[str, Any]],
        feedforward_channels: int,
        operation_order: Tuple[str],
        ffn_dropout: float = 0.0,
        norm_cfg: Dict[str, Any] = dict(type="LN"),
        act_cfg: Optional[Dict[str, Any]] = dict(type="ReLU", inplace=True),
        ffn_num_fcs: Optional[int] = 2,
        **kwargs,
    ):
        super(BEVFormerLayerV3, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs,
        )
        # assert len(operation_order) == 6
        # assert set(operation_order) == set(["self_attn", "norm", "cross_attn", "ffn"])

    def check_attn_masks(
        self, attn_masks: Optional[torch.Tensor | List[Optional[torch.Tensor]]]
    ) -> List[Optional[torch.Tensor]]:
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(
                f"Use same attn_mask in all attentions in "
                f"{self.__class__.__name__} "
            )
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )
        return attn_masks

    def forward(  # type:ignore
        self,
        query: Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor],
        query_pos: Optional[Tensor],
        key_pos: Optional[Tensor],
        # attn_masks: List[Optional[Tensor]] = [],
        attn_masks: Optional[Union[Tensor, List[Optional[Tensor]]]],
        query_key_padding_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        bev_pos: Tensor,
        ref_2d: Tensor,
        ref_3d: Tensor,
        bev_h: int,
        bev_w: int,
        reference_points_cam: Tensor,
        mask: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        prev_bev: Optional[Tensor] = None,
        **kwargs,
    ):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        attn_masks = self.check_attn_masks(attn_masks)

        for i, layer in enumerate(self.operation_order):
            # temporal self attention
            if layer == "self_attn":
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "mamba_self_attn":
                query = self.attentions[attn_index](query, query_pos=bev_pos)
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1
            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query=query,
                    key=key,
                    value=value,
                    identity=identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "both_attn":
                query, value = self.attentions[attn_index](
                    query=query,
                    key=key,
                    value=value,
                    identity=identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
