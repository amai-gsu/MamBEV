from collections.abc import Sequence
import logging
from typing import Dict, List, Optional, Tuple

from mmdet.models.layers.transformer.detr_layers import MultiheadAttention
from mmengine.runner import autocast
import torch
from torch import nn
from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.registry import MODELS
from mmcv.cnn.bricks.drop import build_dropout

from projects.MAMBEV.mambev.utils.sequence_utils import (
    batch_seq_flatten_ex,
    index_add_to_residual_v2,
    reverse_batch_seq_flatten,
)
from mamba_ssm.ops.triton.layer_norm import layer_norm_fn
from einops import rearrange

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm
except ImportError:
    print_log(
        "Failed to import RMSNorm from mamba_ssmm, falling back on LayerNorm",
        "current",
        logging.WARNING,
    )
    RMSNorm = nn.LayerNorm


@MODELS.register_module(force=True)
class DummyLayer(BaseModule):
    def __init__(
        self,
        embed_dims: int,
        mamba_cfg: Dict,
        init_cfg: Optional[Dict] = None,
        **kwargs,
    ):
        super(DummyLayer, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.inner_block = MODELS.build(mamba_cfg)

    @autocast(dtype=torch.float32)
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        identity: Optional[torch.Tensor],
        *,
        spatial_shapes: torch.Tensor,
        reference_points_cam: torch.Tensor,
        bev_mask: torch.Tensor,
        level_start_index: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        flag="encoder",
        **kwargs,
    ):
        """
        testing if this is a mamba issue or a structure issue
        """

        if key is None:
            key = query
        if value is None:
            value = key
        return self.inner_block(
            query=query,
            key=key,
            value=value,
            identity=identity,
            spatial_shapes=spatial_shapes,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            level_start_index=level_start_index,
            query_pos=query_pos,
            **kwargs,
        )


@MODELS.register_module(force=True)
class DummyResidualLayer(BaseModule):
    def __init__(
        self,
        embed_dims: int,
        mamba_cfg: Dict,
        num_cams: int = 6,
        use_layer_norm: bool = True,
        init_cfg: Optional[Dict] = None,
    ):
        super(DummyResidualLayer, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_cams = num_cams

        self.layers = ModuleList([MODELS.build(mamba_cfg)])
        self.layer_norm = RMSNorm([embed_dims]) if use_layer_norm else nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        identity: Optional[torch.Tensor],
        *,
        spatial_shapes: torch.Tensor,
        reference_points_cam: torch.Tensor,
        bev_mask: torch.Tensor,
        level_start_index: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        flag="encoder",
        **kwargs,
    ):
        """
        testing if this is a mamba issue or a structure issue
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            inp_residual = query.clone()
        else:
            inp_residual = identity

        batch_size = query.shape[0]

        out, ext_map, split_cam = self.layers[0](
            query=query,
            key=key,
            value=value,
            identity=identity,
            spatial_shapes=spatial_shapes,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            level_start_index=level_start_index,
            query_pos=query_pos,
            **kwargs,
        )
        # TODO: use split trav instead
        out = torch.split(out.squeeze(0), split_cam)

        # expects out to be a list of tensors where each is
        for i in range(batch_size):
            for camera_index in range(self.num_cams):
                index_add_to_residual_v2(
                    out[i * self.num_cams + camera_index],
                    inp_residual[i],
                    ext_map[i * self.num_cams + camera_index],
                )

        output_bev_query = self.layer_norm(inp_residual)
        return output_bev_query


@MODELS.register_module(force=True)
class DummyResidualSelfAttnLayer(BaseModule):
    def __init__(
        self,
        embed_dims: int,
        mamba_cfg: Dict,
        use_layer_norm: bool = True,
        init_cfg: Optional[Dict] = None,
        **kwargs,
    ):
        super(DummyResidualSelfAttnLayer, self).__init__(init_cfg)
        self.embed_dims = embed_dims

        self.layers = ModuleList([MODELS.build(mamba_cfg)])
        self.layer_norm = RMSNorm([embed_dims]) if use_layer_norm else nn.Identity()

    @autocast(dtype=torch.float32)
    def forward(self, u: torch.Tensor, query_pos: Optional[torch.Tensor] = None):
        if query_pos is not None:
            u = u + query_pos
        res = u.clone()
        for layer in self.layers:
            u = layer(u)
        return self.layer_norm(u + res)


@MODELS.register_module(force=True)
class DummyDecoderDropoutSelfAttnLayer(BaseModule):
    def __init__(
        self,
        embed_dims: int,
        mamba_cfg: Dict,
        proj_drop: float = 0.0,
        dropout_layer: Dict = dict(type="Dropout", drop_prob=0.0),
        init_cfg: Optional[Dict] = None,
    ):
        super(DummyDecoderDropoutSelfAttnLayer, self).__init__(init_cfg)
        self.embed_dims = embed_dims

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = (
            build_dropout(dropout_layer) if dropout_layer else nn.Identity()
        )

        self.layers = ModuleList([MODELS.build(mamba_cfg)])

    @autocast(dtype=torch.float32)
    def forward(self, u: torch.Tensor, query_pos: Optional[torch.Tensor] = None):
        if query_pos is not None:
            u = u + query_pos
        for layer in self.layers:
            u = layer(u)
        return self.dropout_layer(self.proj_drop(u))


@MODELS.register_module(force=True)
class TraversalSelfAttnBlock(BaseModule):
    def __init__(
        self,
        embed_dims: int,
        bev_h: int,
        bev_w: int,
        mamba_cfg: Dict,
        traversal_methods: Sequence[str] = ["tl0snake"],
        patch_size: Tuple[int, int] = (4, 4),
        proj_drop: float = 0.0,
        init_cfg: Optional[Dict] = None,
        batch_first: bool = True,
        layer_idx: Optional[int] = None,
        attn_idx: Optional[int] = None,
    ):
        super(TraversalSelfAttnBlock, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.patch_size = patch_size
        self.traversal_methods = traversal_methods
        self.p_drop = proj_drop
        self.layer_idx = layer_idx
        self.attn_idx = attn_idx
        try:
            self.mixer = MODELS.build(mamba_cfg)
        except KeyError:
            # TEST:
            from projects.MAMBEV.mambev.hydra.hydra import Hydra

            del mamba_cfg["type"]
            self.mixer = Hydra(**mamba_cfg)

        self.layer_norm = RMSNorm([embed_dims])

    def flatten_query(
        self, query: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        # traversals of each image feature level
        # order: F_0 tr0, F_0 tr1, .... F_0 trN, F_1 tr0, ....
        idx_maps = []
        sequences = []
        shapes = []
        query = rearrange(query, "bs (H W)  c -> bs H W c ", H=self.bev_h, W=self.bev_w)
        for traversal in self.traversal_methods:
            vt, idx_map, *shape_out = batch_seq_flatten_ex(
                query, self.bev_h, self.bev_w, traversal, self.patch_size
            )
            shapes.append(shape_out)
            sequences.append(vt)
            idx_maps.append(idx_map)
        sequence = torch.cat(sequences, dim=0)
        idx_maps = torch.cat(idx_maps)
        return sequence, idx_maps, shapes

    def agg_traversals(
        self,
        trav_list: List[torch.Tensor],
        agg_tensor: torch.Tensor,
        shapes: List[Tuple[int, int]],
    ) -> None:
        for seq, method, shape in zip(trav_list, self.traversal_methods, shapes):
            # clone prevents error from setting values on a view of a view
            agg_tensor += rearrange(
                reverse_batch_seq_flatten(seq.clone(), *shape, method, self.patch_size),
                "... H W C -> ... (H W) C",
            )

    @autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    def forward(
        self, query: torch.Tensor, query_pos: Optional[torch.Tensor] = None, **kwargs
    ):
        batch_size = query.size(0)
        inp_res = query.clone()
        if query_pos is not None:
            query = query + query_pos
        # idx can be used to unflatten the seq, but is element wise
        query, idx, shapes = self.flatten_query(query)
        query = self.mixer(query, query_pos=None, **kwargs)
        agg_tensor = torch.zeros_like(inp_res)
        self.agg_traversals(query.split(batch_size), agg_tensor, shapes)
        out = layer_norm_fn(
            agg_tensor,
            self.layer_norm.weight,
            self.layer_norm.bias,
            inp_res,
            dropout_p=self.p_drop,
            is_rms_norm=True,
        )  # type:ignore
        return out


if __name__ == "__main__":
    device = "cuda"

    dim = 32
    bev_h, bev_w = 64, 64
    self_cfg = dict(
        type="TraversalSelfAttnBlock",
        embed_dims=dim,
        bev_h=bev_h,
        bev_w=bev_w,
        traversal_methods=["tl0snake"],
        patch_size=(4, 4),
        proj_drop=0.05,
        use_layer_norm=True,
        mamba_cfg=dict(
            type="Hydra",
            d_model=dim,
            d_state=4,
            d_conv=7,
            expand=1,
            headdim=dim,
            ngroups=1,
        ),
    )
    model = MODELS.build(self_cfg)

    model.to(device)
    test = torch.ones((4, 64**2, dim), device=device, dtype=torch.bfloat16)
    out = model(test)
    print(out.shape)
