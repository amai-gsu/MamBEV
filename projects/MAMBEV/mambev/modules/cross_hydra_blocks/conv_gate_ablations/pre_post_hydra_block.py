# Copyright (c) 2024, Tri Dao, Albert Gu.
# Modified by Jack Morris

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from mmdet3d.registry import MODELS
from mmengine.runner import autocast

from ..base_hydra_block import BaseHydraBlock, ZeroParams
from ..mixins import ConvMixin


def compare_output_tensors(gt: torch.Tensor, test: torch.Tensor):
    gt = gt.float()
    test = test.float()
    emin = float("inf")
    emax = 0
    errors = 0
    passes = 0
    try:
        assert torch.allclose(test, gt)
    except AssertionError:
        gt, test = (
            rearrange(gt, "bs nq c -> (bs nq) c"),
            rearrange(test, "bs nq c -> (bs nq) c"),
        )
        for i, (truth, tt) in enumerate(zip(gt, test)):
            try:
                assert torch.allclose(truth, tt, rtol=1e-6, atol=1e-6)
                passes += 1

            except AssertionError:
                # print(f"test[{i}] ~= gt[{i}]")
                # print(f"{tt[:10].tolist()} != ")
                # print(f"{truth[:10].tolist()}")
                # print("MSE: ", ((tt - truth) ** 2).sum())
                # input()
                # print(f"{ ( tt % truth ).sum() } ~= 0")
                errors += 1
                emin = min(i, emin)
                emax = max(i, emax)
        print(f"Error index range: {emin}, {emax}")
        print(f"Num errors: {errors}")


@MODELS.register_module(force=True)
class PrePostHydraBlock(ConvMixin, BaseHydraBlock):
    def __init__(
        self,
        *args,
        v_zero_params: ZeroParams = {
            "z": True,
        },
        mem_eff_inference: bool = False,
        **kwargs,
    ):
        v_zero_params["z"] = True
        kwargs["v_zero_params"] = v_zero_params
        BaseHydraBlock.__init__(self, *args, **kwargs)

        self.v_in_mask_mask[: self.d_z] = False
        self.mem_eff_inference = mem_eff_inference

        conv_mask = self.construct_in_proj_mask(
            {"dt": True}, torch.ones((self.v_in_mask_mask.size(0),), dtype=torch.bool)
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
        # S6
        if not self.training and self.mem_eff_inference:
            x_og, y = self._inference_inner_ssm(xBCdt, A, vmask, initial_states)
            # return y
            qy = self._add_fb_masked(y, x_og, vmask[~vmask], batch_size=1)

        else:
            x_og = self.x_og_act(xBCdt[..., : self.d_x])
            y = self._inner_ssm(xBCdt, A, seq_idx, initial_states)
            # return y[:, ~vmask]
            qy = self._add_fb_masked(y, x_og, vmask, batch_size=1)

        qy = self.norm(qy.squeeze(0), qz[extract_map["bs"], extract_map["q"]])

        slots = torch.zeros(
            (inp_res.shape[0], inp_res.shape[1], self.d_x),
            dtype=y.dtype,
            device=inp_res.device,
        )
        slots.index_put_(
            [extract_map["bs"], extract_map["q"]], qy.squeeze(0), accumulate=True
        )

        slots = self.post_average(
            bev_mask, slots
        )  # @torch.compile(dynamic=True, backend="eager")
        return self._fused_outproj_res_norm_drop(slots, inp_res)

    # @autocast(dtype=torch.float64)
    def _inference_inner_ssm(
        self,
        xBCdt: torch.Tensor,
        A: torch.Tensor,
        v_mask: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ):
        x_og = xBCdt[..., : self.d_x][:, ~v_mask]
        x, B, C, dt = self._split_xBCdt(xBCdt)
        dtype = x.dtype
        batch_size = x.size(0)
        ssm_state = (
            initial_state
            if initial_state is not None
            else torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
        )
        assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"

        # Discretize A and B
        dA = torch.exp(dt[:, v_mask] * A)  # (batch, nheads)

        y = torch.zeros_like(x[:, ~v_mask])
        x = rearrange(x[:, v_mask], "b t (h p) -> b t h p", p=self.headdim)
        dBx = torch.einsum("bth,btn,bthp->bthpn", dt[:, v_mask], B[:, v_mask], x)
        # dBx[1] = torch.flip(dBx[1].clone(), dims=[1])
        out_idx = [0, 0]
        vpoint = [0, 0]
        for i in range(len(v_mask)):
            if v_mask[i]:
                # "bhpn,bh11->bhpn"
                ssm_state = (
                    ssm_state * rearrange(dA[:, vpoint[0]], "b h -> b h 1 1")
                    + dBx[:, vpoint[0]]
                )
                vpoint[0] += 1
            # elif not v_mask[i]:
            else:
                temp = torch.einsum("hpn,n->hp", ssm_state[0].to(dtype), C[0, i])
                y[0, out_idx[0]] = rearrange(temp, "h p -> (h p)")
                out_idx[0] += 1
            if v_mask[-(i + 1)]:
                # "bhpn,bh11->bhpn"
                ssm_state = (
                    ssm_state * rearrange(dA[:, vpoint[1]], "b h -> b h 1 1")
                    + dBx[:, vpoint[1]]
                )
                vpoint[1] += 1
            else:
                temp = torch.einsum("hpn,n->hp", ssm_state[1].to(dtype), C[1, i])
                y[1, out_idx[1]] = rearrange(temp, "h p -> (h p)")
                out_idx[1] += 1

        return x_og, y
