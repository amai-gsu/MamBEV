from itertools import islice  # , batched
from typing import Iterable, List, Optional, Sequence, Tuple, Union, TypeAlias, Literal

import torch
from einops import rearrange, pack
from einops.layers.torch import Rearrange
from torch import nn
from torch.types import Number

from .tensor_reshape import (
    ContiguousRearrange,
    Rot90,
    SkipFlip,
)


ConvertMethodType: TypeAlias = Literal["floor", "ceil", "round"]


def ref2int_factory(convert_method: ConvertMethodType, offset: int = 0):
    match convert_method:
        case "floor":
            convert_func = torch.floor
        case "ceil":
            convert_func = torch.ceil
        case "round":
            convert_func = torch.round
        case _:
            raise ValueError("Convert method must be one of floor ceil or round")

    def ref2int(
        ref: torch.Tensor, H: torch.Tensor, W: torch.Tensor, separate: bool = False
    ) -> torch.Tensor:
        col = convert_func(ref[..., 0] * (W - offset)).long()
        row = convert_func(ref[..., 1] * (H - offset)).long()
        if separate:
            return pack([col, row], "bs nc nq z *")[0]
        else:
            return row * W + col

    return ref2int


def _parse_sequential(sequential: nn.Sequential) -> Tuple[int, ...]:
    ops = []
    for m in sequential:
        match m.__class__.__name__:
            case "Rot90":
                ops.append(m.rotations)  # can be 0-3
            case "Rearrange":
                pass
            case "ContiguousRearrange":
                ops.append(4)
            case "SkipFlip":
                ops.append(5)
            case _:
                raise ValueError()
    return tuple(ops)


def batched(iterable: Iterable, n: int):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def _get_bools(flatten_method: str, reverse: bool = False):
    if reverse:
        corner_map = {"tl": 0, "tr": 1, "br": 2, "bl": 3}
    else:
        corner_map = {"tl": 0, "tr": 3, "br": 2, "bl": 1}
    start_corner = corner_map[flatten_method[:2]]
    rotations = int(flatten_method[2])
    assert rotations in (0, 1), "Rotation must be either 0 or 1"
    row_prim = rotations == 0
    permute_HW = row_prim + start_corner % 2 == 0
    is_snake = flatten_method[3:] == "snake"
    return start_corner, permute_HW, is_snake


def _add_traversal_ops(sequential: nn.Sequential, flatten_method: str):
    start_corner, permute_HW, is_snake = _get_bools(flatten_method)
    sequential.append(Rot90(rotations=start_corner, dims=[-2, -3]))
    if permute_HW:
        sequential.append(ContiguousRearrange("... H W C -> ... W H C"))
    if is_snake:
        sequential.append(SkipFlip(start=1, step=2, dims=[-2]))


def traversal_factory(
    sequential: nn.Sequential,
    flatten_method: Union[str, Tuple[str, str]],
    patch_size: Tuple[int, int] = (4, 4),
):
    if isinstance(flatten_method, tuple):
        hp, wp = patch_size
        sequential.append(
            Rearrange("... (H h2) (W w2) c ->  ... H W h2 w2 c", h2=hp, w2=wp)
        )
        _add_traversal_ops(sequential, flatten_method[0])
        sequential.append(Rearrange("... H W h2 w2 c -> ... h2 w2 H W c"))
        _add_traversal_ops(sequential, flatten_method[1])
        sequential.append(Rearrange("... h2 w2 H W c -> ... (H h2) (W w2) c"))
    else:
        _add_traversal_ops(sequential, flatten_method)

    return sequential


def _add_reverse_traversal_ops(sequential: nn.Sequential, flatten_method: str):
    start_corner, permute_HW, is_snake = _get_bools(flatten_method, reverse=True)
    if is_snake:
        sequential.append(SkipFlip(start=1, step=2, dims=[-2]))
    if permute_HW:
        sequential.append(ContiguousRearrange("... H W C -> ... W H C"))
    sequential.append(Rot90(rotations=start_corner, dims=[-2, -3]))


def reverse_traversal_factory(
    sequential: torch.nn.Sequential,
    flatten_method: Union[str, Tuple[str, str]],
    patch_size: Tuple[int, int] = (4, 4),
):
    if isinstance(flatten_method, tuple):
        hp, wp = patch_size
        sequential.append(
            Rearrange("... (H h2) (W w2) c ->  ... h2 w2 H W c", h2=hp, w2=wp)
        )
        _add_reverse_traversal_ops(sequential, flatten_method[1])
        sequential.append(Rearrange("... h2 w2 H W c -> ... H W h2 w2 c"))
        _add_reverse_traversal_ops(sequential, flatten_method[0])
        sequential.append(Rearrange("... H W h2 w2 c -> ... (H h2) (W w2) c"))
    else:
        _add_reverse_traversal_ops(sequential, flatten_method)

    return sequential


class Traverse2D(nn.Module):
    def __init__(
        self,
        flatten_methods: Sequence[Union[str, Tuple[str, str]]],
        patch_size: Tuple[int, int],
        reverse: bool,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.flatten_methods = flatten_methods
        self.patch_size = patch_size
        self.reverse = reverse
        self.flatten = Rearrange("... H W C -> ... (H W) C")
        self.flatten_pipes = [nn.Sequential() for _ in self.flatten_methods]
        for i, method in enumerate(self.flatten_methods):
            traversal_factory(self.flatten_pipes[i], method, patch_size)
        if self.reverse:
            self.reverse_pipes = [nn.Sequential() for _ in self.flatten_methods]

            for i, method in enumerate(self.flatten_methods):
                reverse_traversal_factory(self.reverse_pipes[i], method, patch_size)

    def forward_reverse(
        self,
        seq: Sequence[torch.Tensor],
        shapes: Optional[List[Tuple[int, int]]] = None,
    ) -> List[torch.Tensor]:
        if shapes is not None:
            seq = [
                rearrange(
                    s, "bs num_cams (H W) dim -> bs num_cams H W dim", H=H_, W=W_
                ).clone()
                for (H_, W_), s in zip(shapes, seq)
            ]

        assert len(seq) % len(self.flatten_methods) == 0
        output = []

        for traversals_of_level in batched(seq, len(self.flatten_methods)):
            for tr, pipe in zip(traversals_of_level, self.reverse_pipes):
                output.append(pipe(tr))

        return output

    def forward(
        self,
        seq: Union[List[torch.Tensor], torch.Tensor],
        H: Optional[int | Number] = None,
        W: Optional[int | Number] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple[int, int]]]:
        if isinstance(seq, List):
            first_shape = seq[0].shape
            assert len(self.flatten_methods) == len(
                seq
            ), "Must have an equal number of inputs and flatten methods"
            assert all(
                tensor.shape == first_shape for tensor in seq
            ), "Not all tensors have the same shape."

        else:
            seq = [seq.clone() for _ in range(len(self.flatten_methods))]
        device = seq[0].device

        if H is None:
            H = seq[0].size(-3)

        if W is None:
            W = seq[0].size(-2)
        x = torch.arange(H, dtype=torch.long, device=device)
        y = torch.arange(W, dtype=torch.long, device=device)
        row, col = torch.meshgrid(x, y, indexing="ij")
        idx = row * W + col

        # NOTE: May not need to broadcast until the end
        idx = idx.unsqueeze(-1)

        output_seqs = []
        output_indices = []
        output_sizes = []
        for sq, pipe in zip(seq, self.flatten_pipes):
            s1, s2 = pipe(sq), pipe(idx.clone())

            H_out, W_out = s2.shape[-3:-1]
            s1 = self.flatten(s1)

            flatten_indices = self.flatten(s2).squeeze(-1)
            unflatten_indices = self.flatten(torch.clone(idx)).squeeze(-1)

            # flatten_idx[i] = original loc of vector at position i in final traversal
            # unflatten_idx[i] = traversal loc of vector with original position i
            unflatten_indices[flatten_indices] = unflatten_indices.clone()
            output_seqs.append(s1)
            output_indices.append(unflatten_indices)
            output_sizes.append((H_out, W_out))

        return output_seqs, output_indices, output_sizes
