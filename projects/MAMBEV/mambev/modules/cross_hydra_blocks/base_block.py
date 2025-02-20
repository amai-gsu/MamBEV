# Copyright (c) 2024, Tri Dao, Albert Gu.
# Modified by Jack Morris
# import math
import logging
from abc import ABCMeta, abstractmethod
from functools import partial
from itertools import chain
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Tuple,
    TypeAlias,
    TypedDict,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from mamba_ssm.distributed.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mmengine.logging import print_log
from mmengine.model import BaseModule
from torch.distributed import ProcessGroup

from projects.MAMBEV.mambev.layers import Roll, Ref2Int, Traverse2D, batched
from projects.MAMBEV.mambev.utils.sequence_utils import (
    DeviceLikeType,
    get_merge_mask_and_sort,
)

torch._dynamo.config.capture_dynamic_output_shape_ops = True  # type:ignore
torch._dynamo.config.capture_scalar_outputs = True  # type:ignore
torch._dynamo.config.suppress_errors = True  # type:ignore
torch._dynamo.config.optimize_ddp = False  # type:ignore

ConvertMethodType: TypeAlias = Literal["floor", "ceil", "round"]
AverageMethodType: TypeAlias = Literal["slots", "cams"]
CollectMethodType: TypeAlias = Literal["tbc", "bct"]


class ZeroParams(TypedDict, total=False):
    z: bool
    x: bool
    B: bool
    C: bool
    dt: bool


class FactoryKwargs(TypedDict, total=True):
    device: Optional[DeviceLikeType]
    dtype: Optional[torch.dtype]
    # memory_format: Optional[torch.memory_format]


def average_cams(bev_mask: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    hits = reduce(bev_mask, "bs nc q nhnlnp -> bs q nc () ", "max")
    q_counts = torch.sum(hits, 2, dtype=torch.float32)
    slots = slots / torch.clamp(q_counts, min=1.0)
    return slots


def average_slots(bev_mask: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    q_counts = torch.sum(bev_mask, (1, 3), dtype=torch.float32)
    q_counts = rearrange(q_counts, "bs qc -> bs qc ()")
    slots = slots / torch.clamp(q_counts, min=1.0)
    return slots


class BaseHydraBlock(BaseModule, metaclass=ABCMeta):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 7,
        expand: int = 2,
        headdim: int = 64,
        # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        d_ssm: Optional[int] = None,
        ngroups: int = 1,
        D_has_hdim: bool = False,
        rmsnorm: bool = True,
        norm_before_gate: bool = False,
        dt_limit: Tuple[float, float] = (0.0, float("inf")),
        bias: bool = False,
        conv_bias: bool = True,
        dt_bias: bool = True,
        # Fused kernel and sharding options
        chunk_size: int = 256,
        layer_idx: Optional[int] = None,
        process_group: Optional[ProcessGroup] = None,
        sequence_parallel: bool = True,
        learnable_init_states: bool = False,
        device: Optional[DeviceLikeType] = None,
        dtype: Optional[torch.dtype] = None,
        #### MY OPTS ####
        attn_idx: Optional[int] = None,
        batch_first: bool = True,
        use_post_norm: bool = True,
        dropout: float = 0.1,
        num_cams: int = 6,
        q_zero_params: ZeroParams = {"dt": True},
        v_zero_params: ZeroParams = {"z": True},
        x_og_activation: bool = False,
        traversal_methods: Tuple[Union[Tuple[str, str], str], ...] = ("tl0cross",),
        average_method: Optional[AverageMethodType] = None,
        collect_method: CollectMethodType = "tbc",
        feature_levels: Union[Literal["all"], List[int]] = "all",
        patch_size: Tuple[int, int] = (4, 4),
        ref2int_convert_method: ConvertMethodType = "ceil",
        ref2int_offset: int = 1,
        init_cfg: List[Dict[str, Any]] = [
            dict(
                type="Kaiming",
                layer="Linear",
                mode="fan_out",
                nonlinearity="relu",
                distribution="normal",
                override=dict(
                    name="A_log",
                    type="Mamba2A",  # alternatively use HydraA
                    a=1,
                    b=16,
                ),
            ),
            dict(
                type="Kaiming",
                layer="Linear",
                mode="fan_out",
                nonlinearity="relu",
                distribution="normal",
                override=dict(
                    name="dt_bias",
                    type="Mamba2dt",
                    a=0.001,
                    b=0.1,
                    init_floor=1e-4,
                ),
            ),
        ],
        **kwargs,
    ):
        # handle deprecated kwargs
        if "average_slots" in kwargs and kwargs["average_slots"]:
            average_method = "slots"
        self.factory_kwargs: FactoryKwargs = {"device": device, "dtype": dtype}
        super().__init__(init_cfg=init_cfg)
        self.learnable_init_states = learnable_init_states
        # not used for now
        self.patch_size = patch_size
        assert isinstance(feature_levels, list) or feature_levels == "all"
        self.feature_levels: Union[Literal["all"], List[int]] = feature_levels
        self.batch_first = batch_first
        self.use_post_norm = use_post_norm
        self.num_cams = num_cams
        self.embed_dims = self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        # self.conv_init = conv_init
        self.traversal_methods = traversal_methods
        self.ntraversals = len(traversal_methods)
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = int(self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == int(self.expand * self.d_model)
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.attn_idx = attn_idx
        self.q_zero_params = q_zero_params
        self.v_zero_params = v_zero_params
        self.average_method = average_method
        self.p_drop = dropout
        self.ref2int_convert_method = ref2int_convert_method
        self.ref2int_offset = ref2int_offset

        self.traversal_constructor: Traverse2D = Traverse2D(
            self.traversal_methods, self.patch_size, reverse=True
        )
        self.x_og_act: Union[nn.Identity, nn.SiLU]
        if not x_og_activation:
            print_log(
                "x_og_activation is set to False, if queries are merged after the convolution"
                " this will result in 2 linear layers with no activation between them.",
                "current",
                logging.WARNING,
            )
            self.x_og_act = nn.Identity()
        else:
            self.x_og_act = nn.SiLU()

        self.dt_limit_kwargs = (
            {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        )
        self.act = nn.SiLU()
        if self.learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(
                    self.nheads, self.headdim, self.d_state, **self.factory_kwargs
                )
            )
            self.init_states._no_weight_decay = True  # type: ignore

        self.ref2int = Ref2Int(
            convert_method=ref2int_convert_method, offset=ref2int_offset
        )
        # NOTE: the order of these is important
        self._init_sizes()
        self._init_rmsnorms()
        self._init_inproj(bias)
        self._init_masks()
        self._init_outproj(bias)
        self._init_conv(conv_bias, d_conv)
        self._init_dt_bias(dt_bias)
        self._init_AD()

        match average_method:
            case "slots":
                self.post_average = average_slots
            case "cams":
                self.post_average = average_cams
            case None:
                self.post_average = lambda bev_mask, slots: slots
            case _:
                raise ValueError("Invalid average method. Check spelling.")

        self.collect_method = collect_method
        match collect_method:
            case "tbc":
                self.collect_fn = lambda x: list(chain.from_iterable(x))
                self.cat_fn = partial(torch.cat, dim=0)
            case "bct":
                self.collect_fn = lambda x: list(chain.from_iterable(zip(*x)))
                self.cat_fn = partial(torch.cat, dim=-2)
            case "btc":
                raise NotImplementedError()
            case _:
                raise ValueError("Must Include a valid collect method")

        self.roll_C: nn.Module

        # Union[Roll, nn.Identity]
        if "C" in self.v_zero_params and self.v_zero_params["C"]:
            self.roll_C = Roll(-1, 1)
        else:
            self.roll_C = nn.Identity()

    def init_weights(self):
        super().init_weights()
        ## TODO: Remove if init changes are finialized
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                raise ValueError(f"NaN values found in parameter '{name}'")

    def _init_sizes(self):
        self.d_x = self.d_z = self.d_inner
        self.d_C = self.d_B = self.ngroups * self.d_state
        self.d_dt = self.nheads

    def _init_AD(self):
        device = self.factory_kwargs["device"]
        d_out = max(self.d_x, self.d_z)
        self.A_log = nn.Parameter(torch.empty(self.d_dt, **self.factory_kwargs))
        self.A_log._no_weight_decay = True  # type:ignore
        # D "skip" parameter
        self.D = nn.Parameter(
            torch.ones(d_out if self.D_has_hdim else self.d_dt, device=device)
        )

        self.D._no_weight_decay = True  # type:ignore
        self.fc_D = nn.Linear(d_out, self.d_dt, bias=False, **self.factory_kwargs)

    def _init_rmsnorms(self):
        out_dim = max(self.d_z, self.d_x)
        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(
                out_dim,
                eps=1e-5,
                norm_before_gate=self.norm_before_gate,
                group_size=out_dim // self.ngroups,
                **self.factory_kwargs,
            )
        else:
            self.norm = lambda x, z: x
            self.d_z = 0

        if self.use_post_norm:
            self.post_norm = RMSNorm([self.d_model])
            self.dropout = None
        else:
            self.post_norm = None
            self.dropout = (
                nn.Dropout(self.p_drop) if self.p_drop != 0.0 else nn.Identity()
            )

    def _init_outproj(self, bias: bool):
        out_dim = max(self.d_z, self.d_x)
        if self.process_group is None:
            self.out_proj = nn.Linear(
                out_dim, self.d_model, bias=bias, **self.factory_kwargs
            )
        else:
            self.out_proj = RowParallelLinear(
                out_dim * self.world_size,
                self.d_model,
                bias=bias,
                process_group=self.process_group,
                sequence_parallel=self.sequence_parallel,
                **self.factory_kwargs,
            )

    def _init_inproj(self, bias: bool):
        # Order: [z, x, Bfw, Cfw, Bbw, Cbw, dtfw, dtbw]
        self.d_in_proj = self.d_z + self.d_x + (2 * (self.d_B + self.d_C + self.d_dt))
        if self.process_group is None:
            self.in_proj = nn.Linear(
                self.d_model, self.d_in_proj, bias=bias, **self.factory_kwargs
            )
        else:
            self.in_proj = ColumnParallelLinear(
                self.d_model,
                self.d_in_proj * self.world_size,
                bias=bias,
                process_group=self.process_group,
                sequence_parallel=self.sequence_parallel,
                **self.factory_kwargs,
            )

    def _init_masks(self):
        q_mask = self.construct_in_proj_mask(
            self.q_zero_params, torch.ones((self.d_in_proj,), dtype=torch.bool)
        )
        self.register_buffer("q_in_mask", q_mask)
        self.register_buffer(
            "q_in_mask_mask", torch.ones((self.d_in_proj,), dtype=torch.bool)
        )

        v_mask = self.construct_in_proj_mask(
            self.v_zero_params, torch.ones((self.d_in_proj,), dtype=torch.bool)
        )
        self.register_buffer("v_in_mask", v_mask)
        self.register_buffer(
            "v_in_mask_mask", torch.ones((self.d_in_proj,), dtype=torch.bool)
        )

        v_conv_mask = torch.zeros_like(self.v_in_mask[self.d_z :])
        v_conv_mask[: -2 * self.d_dt] = self.v_in_mask[self.d_z : -2 * self.d_dt]
        self.register_buffer("v_conv_mask", v_conv_mask)

    def construct_in_proj_mask(
        self, zero_params: ZeroParams, mask: torch.Tensor
    ) -> torch.Tensor:
        for param, zero in zero_params.items():
            if not zero:
                continue
            match param:
                case "z":
                    mask[: self.d_z] = False
                case "x":
                    mask[self.d_z : self.d_z + self.d_x] = False
                case "B":
                    mask[self.d_z + self.d_x : self.d_z + self.d_x + self.d_B] = False
                    mask[
                        self.d_z + self.d_x + self.d_B + self.d_C : self.d_z
                        + self.d_x
                        + 2 * self.d_B
                        + self.d_C
                    ] = False
                case "C":
                    mask[
                        self.d_z + self.d_x + self.d_B : self.d_z
                        + self.d_x
                        + self.d_B
                        + self.d_C
                    ] = False
                    mask[
                        self.d_z + self.d_x + 2 * self.d_B + self.d_C : self.d_z
                        + self.d_x
                        + 2 * self.d_B
                        + 2 * self.d_C
                    ] = False
                case "dt":
                    mask[-2 * self.d_dt :] = False
                case _:
                    raise ValueError("Invalid parameter matrix")

        return mask

    @abstractmethod
    def _init_conv(self, conv_bias: bool, d_conv: int):
        raise NotImplementedError("Must implement method in the subclass")

    def _init_dt_bias(self, dt_bias: bool):
        if not dt_bias:
            self.register_buffer(
                "dt_bias", torch.zeros(self.d_dt, **self.factory_kwargs)
            )
            return
        self.dt_bias = nn.Parameter(torch.empty(self.d_dt, **self.factory_kwargs))
        self.dt_bias._no_weight_decay = True  # type:ignore

    def _prep_inputs(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        reference_points_cam: torch.Tensor,
        bev_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        num_query=None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        List[int],
        int,
        int,
    ]:
        if identity is None:
            inp_res = query.clone()
        else:
            inp_res = identity
        if query_pos is not None:
            query = query + query_pos

        if num_query is None:
            batch_size, num_query, dim = query.shape
        else:
            batch_seqlen, dim = query.shape
            batch_size = batch_seqlen // num_query

        bev_mask = rearrange(bev_mask, "nc bs ... -> bs nc ...")
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)

        initial_states = (
            repeat(self.init_states, "... -> b ...", b=2 * batch_size)
            if self.learnable_init_states
            else None
        )

        reference_points_cam = rearrange(reference_points_cam, "nc bs ... -> bs nc ...")

        level_sizes = [int(H * W) for H, W in spatial_shapes]
        if self.feature_levels != "all":
            temp = value.split(level_sizes, dim=2)
            value = torch.cat(
                [temp[i] for i in self.feature_levels],
                dim=2,
            )
            spatial_shapes = spatial_shapes[self.feature_levels]
            level_sizes = [level_sizes[i] for i in self.feature_levels]

        return (
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
        )

    def _create_input_seqs(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        v_in_seq = value.new_zeros(
            (*value.shape[:-1], self.v_in_mask_mask.sum()), dtype=dtype
        )
        q_in_seq = query.new_zeros(
            (*query.shape[:-1], self.q_in_mask_mask.sum()), dtype=dtype
        )
        return q_in_seq, v_in_seq

    def _masked_in_proj(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        q_out: torch.Tensor,
        v_out: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        v_in_proj_w = self.in_proj.weight[self.v_in_mask]
        q_in_proj_w = self.in_proj.weight[self.q_in_mask]
        if self.in_proj.bias is not None:
            v_in_proj_b = self.in_proj.bias[self.v_in_mask]
            q_in_proj_b = self.in_proj.bias[self.q_in_mask]
        else:
            v_in_proj_b = q_in_proj_b = None

        v_out[..., self.v_in_mask[self.v_in_mask_mask]] = F.linear(
            value, v_in_proj_w, v_in_proj_b
        )
        q_out[..., self.q_in_mask[self.q_in_mask_mask]] = F.linear(
            query, q_in_proj_w, q_in_proj_b
        )
        return q_out, v_out

    def _split_xBCdt(
        self, xBCdt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xBC, dt = torch.split(
            xBCdt,
            [
                self.d_x + 2 * self.d_B + 2 * self.d_C,
                2 * self.d_dt,
            ],
            dim=-1,
        )

        dt = torch.cat(
            (dt[:, :, : self.d_dt], torch.flip(dt[:, :, self.d_dt :], (1,))),
            dim=0,
        )

        x, BC = torch.split(
            xBC,
            [
                self.d_x,
                2 * self.d_B + 2 * self.d_C,
            ],
            dim=-1,
        )
        x = torch.cat((x, torch.flip(x, (1,))), dim=0)
        BC = torch.cat(
            (
                BC[:, :, : self.d_B + self.d_C],
                torch.flip(BC[:, :, self.d_B + self.d_C :], (1,)),
            ),
            dim=0,
        )
        B, C = torch.split(BC, [self.d_B, self.d_C], dim=-1)
        C = self.roll_C(C)
        return x, B, C, dt

    def _inner_ssm(
        self,
        xBCdt: torch.Tensor,
        A: torch.Tensor,
        seq_idx: Optional[torch.Tensor],
        initial_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x, B, C, dt = self._split_xBCdt(xBCdt)
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", h=self.d_dt),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", n=self.d_state),
            rearrange(C, "b l (g n) -> b l g n", n=self.d_state),
            chunk_size=self.chunk_size,
            D=None,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
            dt_softplus=False,
            dt_bias=None,
            **self.dt_limit_kwargs,  # type:ignore
        )  # type:torch.Tensor

        return rearrange(y, "b l h p -> b l (h p)")

    def _add_fb_masked(
        self, y: torch.Tensor, x_og: torch.Tensor, mask: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        mask = ~mask
        y = torch.roll(y, shifts=1, dims=1)
        # TODO: Consider setting first element of all seqs to 0
        y[:, 0, :] = 0.0
        # NOTE: uncomment for reproducability
        # y = y[:, mask]
        # y_fw, y_bw = (
        #     y[0:batch_size],
        #     torch.flip(y[batch_size : 2 * batch_size], (1,)),
        # )
        # # batches are collapsed (unpadding)
        y = (
            y[0:batch_size][:, mask]
            + torch.flip(y[batch_size : 2 * batch_size], (1,))[:, mask]
        )
        y += x_og[:, mask] * repeat(
            F.linear(x_og[:, mask], self.fc_D.weight, bias=self.D),
            "b l h -> b l (h p)",
            p=self.headdim,
        )
        return y

    def _add_fb_masked_pregate(
        self,
        y: torch.Tensor,
        x_og: torch.Tensor,
        mask: torch.Tensor,
        extract_map: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        mask = ~mask
        y = torch.roll(y, shifts=1, dims=1)
        y[:, 0, :] = 0.0
        # batches are collapsed (unpadding)
        y = (
            y[0:batch_size][:, mask]
            + torch.flip(y[batch_size : 2 * batch_size], (1,))[:, mask]
        )
        x_og = x_og * repeat(
            F.linear(x_og, self.fc_D.weight, bias=self.D),
            "b l h -> b l (h p)",
            p=self.headdim,
        )

        return x_og.index_put_(
            [extract_map["bs"], extract_map["q"]], y.squeeze(0), accumulate=True
        )

    def _fused_outproj_res_norm_drop(
        self,
        y: torch.Tensor,
        inp_res: torch.Tensor,
    ) -> torch.Tensor:
        if self.post_norm is not None:
            return layer_norm_fn(
                self.out_proj(y),
                self.post_norm.weight,
                self.post_norm.bias,
                inp_res,
                dropout_p=self.p_drop,
                is_rms_norm=True,
            )  # type:ignore
        else:
            return self.dropout(self.out_proj(y)) + inp_res  # type:ignore

    def _traverse_image_features_contructor(
        self, vxBCdt: torch.Tensor, level_sizes: List[int], spatial_shapes: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[Tuple[int, int]],
    ]:
        value_level_list: List[torch.Tensor] = vxBCdt.split(level_sizes, dim=2)
        value_level_list = [
            rearrange(
                value_level, "bs num_cams (H W) dim -> bs num_cams H W dim", H=H_, W=W_
            )
            for (H_, W_), value_level in zip(spatial_shapes, value_level_list)
        ]

        # traversals of each image feature level
        # order: F_0 tr0, F_0 tr1, .... F_0 trN, F_1 tr0, ....
        values_traversals = []
        idx_maps = []
        out_shapes = []
        count = 1
        seq_lens = value_level_list[0].new_zeros(
            len(value_level_list) * self.ntraversals + 1, dtype=torch.int64
        )
        trav_mult = torch.arange(
            1, self.ntraversals + 1, dtype=seq_lens.dtype, device=seq_lens.device
        )

        for (H, W), value_level in zip(spatial_shapes, value_level_list):
            added_len = H * W
            vts, idxs, sizes = self.traversal_constructor.forward(
                value_level, H.item(), W.item()
            )
            values_traversals.append(vts)
            idx_maps.append(idxs)
            out_shapes.append(sizes)
            seq_lens[count : count + self.ntraversals] = (
                seq_lens[count - 1] + added_len * trav_mult
            )

            count += self.ntraversals
        values_traversals_ten = self.cat_fn(self.collect_fn(values_traversals))
        idx_maps_ten = torch.cat(self.collect_fn(idx_maps), dim=-1)
        out_shapes_lis = list(chain.from_iterable(self.collect_fn(out_shapes)))
        return (values_traversals_ten, idx_maps_ten, seq_lens, out_shapes_lis)

    def _traverse_image_features(
        self,
        vxBCdt: torch.Tensor,
        level_sizes: List[int],
        spatial_shapes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (values_traversals, idx_maps, seq_lens, out_shapes) = (
            self._traverse_image_features_contructor(
                vxBCdt, level_sizes, spatial_shapes
            )
        )
        return values_traversals, idx_maps, seq_lens

    def _bev_aware_merge(
        self,
        qxBCdt: torch.Tensor,
        vxBCdt: torch.Tensor,
        reference_points_cam: torch.Tensor,
        query_grid_hit_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
        value_map: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        extract_map, vmask, seq_lens_new = self._build_extractmap_mask(
            query_grid_hit_mask,
            reference_points_cam,
            spatial_shapes,
            value_map,
            seq_lens,
        )
        input_seq = qxBCdt.new_zeros(
            (
                1,
                int(seq_lens_new.sum().item()),
                self.d_x + 2 * (self.d_B + self.d_C + self.d_dt),
            )
        )
        vxBCdt = rearrange(vxBCdt, "bs ncntnv c -> (bs ncntnv) c")

        input_seq[:, vmask] = vxBCdt
        input_seq[:, ~vmask] = qxBCdt[extract_map["bs"], extract_map["q"]]

        return input_seq, extract_map, vmask

    def _iter_maps(
        self,
        all_inds_tr_batch_cam: Tuple[torch.Tensor, ...],
        int_ref_bylevel_defaulttr: List[Tuple[torch.Tensor, ...]],
        vmap_level_tr: Tuple[torch.Tensor, ...],
    ) -> Generator[
        Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, int, int], None, None
    ]:
        running_q_count = 0
        for init_refs, vtr in zip(
            int_ref_bylevel_defaulttr, batched(vmap_level_tr, n=self.ntraversals)
        ):
            # iterate over batch and camera
            for indices, refs in zip(all_inds_tr_batch_cam, init_refs):
                # iterate over num traversals
                for i, vmap in enumerate(vtr):
                    update_refs = vmap[refs]
                    sort_offsets, vmask, new_len = get_merge_mask_and_sort(
                        len(vmap),
                        update_refs,
                    )

                    yield i, indices, sort_offsets, vmask, new_len, running_q_count

                running_q_count += len(indices)

    def _build_extractmap_mask(
        self,
        bev_mask: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        value_map: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        bev_mask (bs, nc, q, z): Mask for the reference_points that fall in bounds
        reference_points (bs, nc, q, z, 2): Locations on the image corresponding to each query
        spatial_shapes (nl, 2): shapes of the image feature maps by level
        value_map (nl, (tr,H*W)): locations of each image feature after traversal and flattening
        ## GOAL ##
        # reorder query_ind so that the correct query is inserted into the sequence
        # update the insert position based on the traversal method

        """
        tr_bev_mask = bev_mask
        all_inds = torch.nonzero(tr_bev_mask)
        nquery_per_tr_batch_cam = reduce(
            tr_bev_mask.int(), "bs nc q z -> (bs nc)", "sum"
        ).tolist()
        int_ref_bylevel_defaulttr = [
            self.ref2int(reference_points[tr_bev_mask], H_, W_).split_with_sizes(
                nquery_per_tr_batch_cam, dim=-1
            )
            for H_, W_ in spatial_shapes
        ]
        # assert isinstance(int_ref_bylevel_defaulttr[1], torch.Tensor)
        all_inds_tr_batch_cam = all_inds.split_with_sizes(
            nquery_per_tr_batch_cam, dim=0
        )
        vmap_level_tr = value_map.split_with_sizes(seq_lens.diff().tolist(), dim=-1)

        # match the format of traversals for collect_fn
        update_inds: List[List[torch.Tensor]] = [[] for _ in range(self.ntraversals)]
        q_expand_map: List[List[torch.Tensor]] = [[] for _ in range(self.ntraversals)]
        seq_lens_new: List[List[int]] = [[] for _ in range(self.ntraversals)]
        vmasks: List[List[torch.Tensor]] = [[] for _ in range(self.ntraversals)]
        # q_expand_map = []
        # running_q_count = 0
        for (
            i,
            indices,
            sort_offsets,
            vmask,
            new_len,
            running_q_count,
        ) in self._iter_maps(
            all_inds_tr_batch_cam, int_ref_bylevel_defaulttr, vmap_level_tr
        ):
            update_inds[i].append(indices[sort_offsets])
            q_expand_map[i].append(sort_offsets + running_q_count)
            seq_lens_new[i].append(new_len)
            vmasks[i].append(vmask)

        all_inds = torch.cat(self.collect_fn(update_inds))
        sort_offsets_flat = torch.cat(self.collect_fn(q_expand_map))

        vmasks_tens = torch.cat(self.collect_fn(vmasks))
        input_seq_lens = torch.tensor(self.collect_fn(seq_lens_new))
        return (
            {
                "bs": all_inds[:, 0],
                "nc": all_inds[:, 1],
                "q": all_inds[:, 2],
                "z": all_inds[:, 3],
                "flat": sort_offsets_flat,
            },
            vmasks_tens,
            input_seq_lens,
        )

    def _construct_input_seq(
        self,
        qxBCdt: torch.Tensor,
        vxBCdt: torch.Tensor,
        reference_points_cam: torch.Tensor,
        spatial_shapes: torch.Tensor,
        query_grid_hit_mask: torch.Tensor,
        level_sizes: List[int],
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        vxBCdt, vmap, seq_lens = self._traverse_image_features(
            vxBCdt, level_sizes, spatial_shapes
        )
        vxBCdt = rearrange(vxBCdt, "bs nc nv c -> bs (nc nv) c")
        input_seq, extract_map, vmask = self._bev_aware_merge(
            qxBCdt,
            vxBCdt,
            reference_points_cam,
            query_grid_hit_mask,
            spatial_shapes,
            vmap,
            seq_lens,
        )

        return input_seq, extract_map, vmask
