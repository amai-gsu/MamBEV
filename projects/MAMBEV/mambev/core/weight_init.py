import logging
import math
from mmengine import print_log
from mmengine.model import (
    BaseInit,
    constant_init,
    uniform_init,
    update_init_info,
)
import torch.nn as nn
import torch

from mmengine.registry import WEIGHT_INITIALIZERS


def _get_bases_name(m):
    return [b.__name__ for b in m.__class__.__bases__]


def hydra_A_init(m: nn.Parameter):
    with torch.no_grad():
        m.fill_(0)
        update = torch.full_like(m.data, 1.0, dtype=torch.float).log().to(m.data.dtype)
        m.data += update
    print_log(f"HydraA Result: {m.data}", "current", logging.DEBUG)


def mamba2_A_init(m: nn.Parameter, a: float, b: float):
    with torch.no_grad():
        m.fill_(0)
        update = torch.empty_like(m.data, dtype=torch.float).uniform_(a, b).log()
        m.data += update.to(m.data.dtype)

    print_log(f"Mamba2A Result: {m.data}", "current", logging.DEBUG)


# def s4d_A_init():
#     # S4D real initialization
#     A = repeat(
#         torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
#         "n -> d n",
#         d=self.d_inner,
#     ).contiguous()
#     A_log = torch.log(A)  # Keep A_log in fp32
#     self.A_log = nn.Parameter(A_log)
#


def dt_bias_init(m: nn.Parameter, dt_max: float, dt_min: float, dt_init_floor: float):
    with torch.no_grad():
        m.data.uniform_(0, 1)
        m.data *= math.log(dt_max) - math.log(dt_min)
        m.data += math.log(dt_min)
        m.data.exp_()
        m.data.clamp_min_(dt_init_floor)
        m.data += torch.log(-torch.expm1(-m.data))

    print_log(f"DT Bias init Result: {m.data}", "current", logging.DEBUG)


def grid_init(m: nn.Linear, nheads: int, nlevels: int, pillar_points: int):
    constant_init(m, 0.0)
    thetas = torch.arange(nheads, dtype=torch.float32) * (2.0 * math.pi / nheads)
    grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
    grid_init = (
        (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        .view(nheads, 1, 1, 2)
        .repeat(1, nlevels, pillar_points, 1)
    )
    for i in range(pillar_points):
        grid_init[:, :, i, :] *= i + 1
    with torch.no_grad():
        m.bias += grid_init.view(-1).to(m.bias.device, m.bias.dtype)


@WEIGHT_INITIALIZERS.register_module(name="Mamba2A")
class Mamba2AInit(BaseInit):
    r"""
    Args:
        a (float): The minimum cutoff value.
        b (float): The maximum cutoff value.
    """

    def __init__(self, a: float = 1, b: float = 16, **kwargs) -> None:
        super().__init__(**kwargs)
        assert a > 0 and b >= a
        self.lower = a
        self.upper = b

    def __call__(self, module: nn.Parameter) -> None:
        def init(m):
            if self.wholemodule:
                mamba2_A_init(m, self.lower, self.upper)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basesname)):
                    mamba2_A_init(m, self.lower, self.upper)

        init(module)
        # module.apply(init)
        if hasattr(module, "_params_init_info"):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f"{self.__class__.__name__}: lower={self.lower}, upper={self.upper}"
        return info


@WEIGHT_INITIALIZERS.register_module(name="HydraA")
class HydraAInit(BaseInit):
    r""" """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self, module: nn.Parameter) -> None:
        def init(m):
            if self.wholemodule:
                hydra_A_init(m)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basesname)):
                    hydra_A_init(m)

        init(module)
        # module.apply(init)
        if hasattr(module, "_params_init_info"):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f"{self.__class__.__name__}"
        return info


@WEIGHT_INITIALIZERS.register_module(name="Mamba2dt")
class Mamba2dtInit(BaseInit):
    r"""
    Args:
        a (float): The minimum cutoff value.
        b (float): The maximum cutoff value.
        init_floor (float): Minimum value of dt
    """

    def __init__(self, a: float, b: float, init_floor: float, **kwargs) -> None:
        super().__init__(**kwargs)
        # assert a > 0 and b >= a
        self.lower = a
        self.upper = b
        self.floor = init_floor

    def __call__(self, module: nn.Parameter) -> None:
        def init(m):
            if self.wholemodule:
                dt_bias_init(m, self.lower, self.upper, self.floor)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basesname)):
                    dt_bias_init(m, self.lower, self.upper, self.floor)

        init(module)
        # module.apply(init)
        if hasattr(module, "_params_init_info"):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f"{self.__class__.__name__}: lower={self.lower}, upper={self.upper},"
        f" floor={self.floor}"
        return info


@WEIGHT_INITIALIZERS.register_module(name="Grid")
class GridInit(BaseInit):
    r"""
    Args:
        nheads (int): Number of deformable attn heads
        nlevels (int): Number of feature map levels
        pillar_points (int): Number of reference points per query
    """

    def __init__(self, nheads: int, nlevels: int, pillar_points: int, **kwargs) -> None:
        super().__init__(**kwargs)
        # assert a > 0 and b >= a
        self.nheads = nheads
        self.nlevels = nlevels
        self.pillar_points = pillar_points

    def __call__(self, module: nn.Module) -> None:
        def init(m):
            if self.wholemodule:
                grid_init(m, self.nheads, self.nlevels, self.pillar_points)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basesname)):
                    grid_init(m, self.nheads, self.nlevels, self.pillar_points)

        module.apply(init)
        if hasattr(module, "_params_init_info"):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = (
            f"{self.__class__.__name__}: nheads={self.nheads}, nlevels={self.nlevels},"
            f" pillar_points={self.pillar_points}"
        )
        return info


@WEIGHT_INITIALIZERS.register_module(name="Dummy")
class DummyInit:
    r"""
    Does nothing
    Args:
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # assert a > 0 and b >= a

    def __call__(self, module: nn.Module) -> None:
        pass
