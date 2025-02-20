# Copyright (c) 2024, Tri Dao, Albert Gu.
# Modified by Jack Morris

from typing import Optional

import torch
import torch.nn.functional as F
from einops import repeat
from mmdet3d.registry import MODELS
from mmengine.runner import autocast

from ..base_hydra_block import BaseHydraBlock, ZeroParams
from ..mixins import ConvMixin


@MODELS.register_module(force=True)
class PrePreHydraBlock(ConvMixin, BaseHydraBlock):
    def __init__(
        self,
        *args,
        v_zero_params: ZeroParams = {
            "z": True,
        },
        **kwargs,
    ):
        v_zero_params["z"] = True
        kwargs["v_zero_params"] = v_zero_params
        BaseHydraBlock.__init__(self, *args, **kwargs)
        self.v_in_mask_mask[: self.d_z] = False

    def _init_masks(self):
        super()._init_masks()
        conv_mask = self.construct_in_proj_mask(
            {"dt": True}, torch.ones((self.d_in_proj,), dtype=torch.bool)
        )

        self.register_buffer("conv_mask", conv_mask[self.d_z :])

    @autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        reference_points_cam: torch.Tensor,
        bev_mask: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        num_query=None,
        seq_idx=None,
        identity=None,
        **kwargs,
    ):
        """
        num_values = sum([level_h*level_w for level_h, level_w in FPN_output_HW])
        Value shape:          [batch_size, num_cameras, num_values, _dim_]
        Query shape:          [batch_size, bev_h*bev_w, _dim_]
        Refence points shape: [num_cameras, batch_size, bev_h*bev_w, z_levels, xy]
        BEV mask shape:       [num_cameras, batch_size, bev_h*bev_w, z_levels]
        """
        (
            query,
            value,
            inp_res,
            bev_mask,
            A,
            reference_points_cam,
            initial_states,
            spatial_shapes,
            level_sizes,
            batch_size,
            num_query,
        ) = self._prep_inputs(
            query=query,
            value=value,
            reference_points_cam=reference_points_cam,
            spatial_shapes=spatial_shapes,
            bev_mask=bev_mask,
            query_pos=query_pos,
            identity=identity,
            num_query=num_query,
        )
        qzxBCdt, vxBCdt = self._create_input_seqs(query, value)
        qzxBCdt, vxBCdt = self._masked_in_proj(query, value, qzxBCdt, vxBCdt)

        qz, qxBCdt = torch.split(
            qzxBCdt, [self.d_inner, len(self.q_in_mask) - self.d_inner], dim=-1
        )
        xBCdt, extract_map, vmask = self._construct_input_seq(
            qxBCdt, vxBCdt, reference_points_cam, spatial_shapes, bev_mask, level_sizes
        )
        # keep dt=0 if param is set to that
        if "dt" in self.q_zero_params and self.q_zero_params["dt"]:
            dt_mask = vmask
        else:
            dt_mask = torch.ones_like(vmask, dtype=torch.bool)

        xBCdt[..., -2 * self.nheads :][..., dt_mask, :] = F.softplus(
            xBCdt[..., -2 * self.nheads :][..., dt_mask, :]
            + repeat(self.dt_bias, "nh -> (2 nh)")
        ).to(xBCdt.dtype)

        # 1D Convolution
        xBCdt = self._conv(xBCdt)

        x_og = self.x_og_act(xBCdt[..., : self.d_x])
        y = self._inner_ssm(xBCdt, A, seq_idx, initial_states)
        qy = self._add_fb_masked(y, x_og, vmask, batch_size=1)

        slots = torch.zeros(
            (inp_res.shape[0], inp_res.shape[1], self.d_x),
            dtype=y.dtype,
            device=inp_res.device,
        )
        slots.index_put_(
            [extract_map["bs"], extract_map["q"]], qy.squeeze(0), accumulate=True
        )

        slots = self.post_average(bev_mask, slots)
        slots = self.norm(slots, qz)
        return self._fused_outproj_res_norm_drop(slots, inp_res)
