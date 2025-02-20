# Jack Morris
# import math
from typing import (
    Any,
    Literal,
    Tuple,
    TypeAlias,
    Union,
)

import torch

ConvertMethodType: TypeAlias = Literal["floor", "ceil", "round"]


class Ref2Int:
    def __init__(self, convert_method: ConvertMethodType, offset: int = 0) -> None:
        self.offset = offset
        match convert_method:
            case "floor":
                self.convert_func = torch.floor
            case "ceil":
                self.convert_func = torch.ceil
            case "round":
                self.convert_func = torch.round
            case _:
                raise ValueError("Convert method must be one of floor ceil or round")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.ref2int(*args, **kwds)

    def _ref2int(
        self,
        ref: torch.Tensor,
        H: Union[torch.Tensor, int],
        W: Union[torch.Tensor, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        col = self.convert_func(ref[..., 0].detach() * (W - self.offset)).long()
        row = self.convert_func(ref[..., 1].detach() * (H - self.offset)).long()
        return col, row

    def ref2int_sep(
        self,
        ref: torch.Tensor,
        H: Union[torch.Tensor, int],
        W: Union[torch.Tensor, int],
    ) -> torch.Tensor:
        col, row = self._ref2int(ref, H, W)
        # return pack([col, row], "bs nc nq z *")[0]
        return torch.stack([col, row], dim=-1)

    def ref2int(
        self,
        ref: torch.Tensor,
        H: Union[torch.Tensor, int],
        W: Union[torch.Tensor, int],
    ) -> torch.Tensor:
        col, row = self._ref2int(ref, H, W)
        return row * W + col
