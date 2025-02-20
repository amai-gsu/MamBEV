from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear
import torch
from torch import nn
from torch.distributed import ProcessGroup

from projects.MAMBEV.mambev.modules.hydra_blocks.base_hydra_block import (
    ConvertMethodType,
    FactoryKwargs,
)

from ....utils import RefPtsVis


class HasBEVAttrs(Protocol):
    pillar_points: int
    nlevels: int
    debug: bool
    debug_vis: RefPtsVis


class HasMiscAttrs(Protocol):
    ref2int: Callable[
        [torch.Tensor, Union[torch.Tensor, int], Union[torch.Tensor, int], bool],
        torch.Tensor,
    ]
    ref2int_offset: int
    ref2int_convert_method: ConvertMethodType
    layer_idx: Optional[int]
    attn_idx: Optional[int]


class HasSizeAttrs(Protocol):
    d_z: int
    d_x: int
    d_B: int
    d_C: int
    d_dt: int
    d_inner: int
    d_model: int
    d_conv: int
    d_in_proj: int
    num_cams: int
    headdim: int
    ntraversals: int


class HasConvAttrs(Protocol):
    v_conv_mask: torch.Tensor
    conv_mask: torch.Tensor
    conv_init: Optional[float]
    act: nn.SiLU


class HasDistributedAttrs(Protocol):
    world_size: int
    process_group: Optional[ProcessGroup]
    factory_kwargs: FactoryKwargs


class HasSkipConnectionAttrs(Protocol):
    fc_D: nn.Linear
    D: nn.Parameter


class Hasdt(Protocol):
    dt_bias: nn.Parameter


class HasLocationWeightAttrs(Protocol):
    xy_dim: Tuple[bool, bool]
