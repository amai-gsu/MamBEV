from typing import Any, List

import torch
from einops.layers.torch import Rearrange
from torch import nn


class Rot90(nn.Module):
    """
    Layer behaves identically to nn.functional.rot90

    """

    def __init__(self, rotations: int, dims: List[int]) -> None:
        super().__init__()
        self.rotations = rotations
        self.dims = dims

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return torch.rot90(seq, self.rotations, self.dims)


class SkipFlip(nn.Module):
    def __init__(
        self, start: int = 1, step: int = 2, dims: List[int] = [-2], *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.start = start
        self.step = step
        self.dims = dims

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        seq[..., self.start :: self.step, :, :] = torch.flip(
            seq[..., self.start :: self.step, :, :].clone(), dims=self.dims
        )
        return seq


class Roll(nn.Module):
    def __init__(self, shifts: int = -1, dims: int = 1) -> None:
        super().__init__()
        self.shifts = shifts
        self.dims = dims

    def forward(self, input):
        return torch.roll(input, shifts=self.shifts, dims=self.dims)


class ContiguousRearrange(Rearrange):
    def __init__(self, pattern: str, **axes_lengths: Any) -> None:
        super().__init__(pattern, **axes_lengths)

    def forward(self, input):
        out = super().forward(input)
        return out.contiguous()
        # return out.clone(memory_format=torch.contiguous_format)
