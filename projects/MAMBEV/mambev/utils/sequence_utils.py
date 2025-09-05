from functools import lru_cache
import torch
from typing import Any, Sequence, Tuple, Union, Optional, List

from einops import pack, rearrange, repeat
from einops.layers.torch import Rearrange
from torch._prims_common import DeviceLikeType

# from torch.nn.utils.rnn import pad_sequence
from torch.types import Number
from mmengine.registry import TASK_UTILS


def seq_len_to_seq_idx(
    seq_len: Sequence[int], device: Optional[DeviceLikeType] = None
) -> torch.Tensor:
    """
    Convert a sequence of integers to an index tensor of length == sum(seq_len)
    """
    seq_idx = torch.cat(
        [torch.arange(ll, device=device, dtype=torch.int32) for ll in seq_len],
        dim=0,
    )
    return seq_idx.unsqueeze(0)


def seq_len_to_mask(
    seq_len: Sequence[int], device: Optional[DeviceLikeType] = None
) -> torch.Tensor:
    longest = max(seq_len)
    batch = len(seq_len)
    base = torch.zeros((batch, longest), device=device, dtype=torch.bool)
    for i, sl in enumerate(seq_len):
        base[i][:sl] = True
    return base


def seq_len_to_cu_seqlens(
    seq_len: Sequence[int], device: Optional[DeviceLikeType] = None
) -> torch.Tensor:
    """
    Convert a sequence of integers to a tensor of cumulative sequence lengths starting from 0
    """
    seq_idx = torch.cat(
        [torch.arange(ll, device=device, dtype=torch.int32) for ll in seq_len],
        dim=0,
    )
    return seq_idx.unsqueeze(0)


def seq_len_to_seq_idx_alt(
    seq_len: Sequence[int], device: Optional[DeviceLikeType] = None
) -> torch.Tensor:
    # WARNING: Do not use
    seq_idx = torch.cat(
        [
            torch.ones((1, ll), device=device, dtype=torch.int32) * i
            for i, ll in enumerate(seq_len)
        ],
        dim=1,
    )
    return seq_idx


def flip_hori(col: torch.Tensor):
    return torch.flip(col, dims=(1,)).as_strided(col.size(), col.stride())


def flip_vert(row: torch.Tensor):
    return torch.flip(row, dims=(0,)).as_strided(row.size(), row.stride())


def even_flip(col: torch.Tensor):
    col = col.contiguous()
    H, W = col.shape
    reverse_idx = torch.arange(W - 1, -1, -1, device=col.device)
    mask = torch.arange(H, device=col.device) % 2 == 1
    col[mask] = col[mask][:, reverse_idx]
    return col.clone(memory_format=torch.contiguous_format)


def odd_flip(row: torch.Tensor):
    row = row.contiguous()
    H, W = row.shape
    reverse_idx = torch.arange(H - 1, -1, -1, device=row.device)
    mask = torch.arange(W, device=row.device) % 2 == 1
    row[:, mask] = row[:, mask][reverse_idx]
    return row.clone(memory_format=torch.contiguous_format)


def _batch_seq_flatten(
    seq: torch.Tensor,
    H: int,
    W: int,
    flatten_method: Union[str, Tuple[str, str]],
    patch_size: Tuple[int, int] = (4, 4),
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    device = seq.device
    corner_map = {"tl": 0, "tr": 3, "br": 2, "bl": 1}

    # NOTE: Ref pts must be a 1d sequence of indices that fall within the bounds of
    # [0, H*W] for this to be effective
    def _traverse_seq(seq: torch.Tensor, idx: torch.Tensor, flatten_method: str):
        start_corner = corner_map[flatten_method[:2]]
        # rotate so that desired corner is in top left
        seq = torch.rot90(seq, start_corner, dims=[-2, -3])
        idx = torch.rot90(idx, start_corner, dims=[-2, -3])

        rotations = int(flatten_method[2])
        row_prim = rotations == 0

        if row_prim + start_corner % 2 == 0:
            seq = rearrange(seq, "... H W C -> ... W H C").contiguous()
            idx = rearrange(idx, "... H W C -> ... W H C").contiguous()

        if flatten_method[3:] == "snake":
            seq[..., 1::2, :, :] = torch.flip(seq[..., 1::2, :, :].clone(), dims=[-2])
            idx[..., 1::2, :, :] = torch.flip(idx[..., 1::2, :, :].clone(), dims=[-2])

        return seq, idx

    # og_order = rearrange(seq, "... H W C ->... (H W) C")
    x = torch.arange(H, dtype=torch.long, device=device)
    y = torch.arange(W, dtype=torch.long, device=device)
    row, col = torch.meshgrid(x, y, indexing="ij")
    idx = row * W + col

    # NOTE: May not need to broadcast until the end
    unflatten_indices = torch.clone(idx)
    idx = idx.unsqueeze(-1)

    if isinstance(flatten_method, tuple):
        # flatten_indices = flatten_input_local_scan(H, W, flatten_method, patch_size)

        hp, wp = patch_size
        seq = rearrange(seq, "... (H h2) (W w2) c ->  ... H W h2 w2 c", h2=hp, w2=wp)
        idx = rearrange(idx, "... (H h2) (W w2) c ->  ... H W h2 w2 c", h2=hp, w2=wp)
        seq, idx = _traverse_seq(seq, idx, flatten_method[0])
        seq = rearrange(seq, "... H W h2 w2 c -> ... h2 w2 H W c")
        idx = rearrange(idx, "... H W h2 w2 c -> ... h2 w2 H W c")
        seq, idx = _traverse_seq(seq, idx, flatten_method[1])
        seq = rearrange(seq, "... h2 w2 H W c -> ... (H h2) (W w2) c")
        idx = rearrange(idx, "... h2 w2 H W c -> ... (H h2) (W w2) c")

    else:
        seq, idx = _traverse_seq(seq, idx, flatten_method)
    seq = rearrange(seq, "... H W C -> ... (H W) C")

    # print("FLAT SHAPE:", seq.shape)
    H_out, W_out = idx.shape[-3:-1]
    flatten_indices = rearrange(idx, "... H W C -> ... (H W) C")
    flatten_indices = flatten_indices.squeeze(-1)
    unflatten_indices = rearrange(unflatten_indices, "H W  -> (H W)")

    # flatten_idx[i] = original loc of vector at position i in final traversal
    # unflatten_idx[i] = traversal loc of vector with original position i
    unflatten_indices[flatten_indices] = unflatten_indices.clone()

    return seq, unflatten_indices, H_out, W_out


def batch_seq_flatten(
    seq: torch.Tensor,
    H: int,
    W: int,
    flatten_method: Union[str, Tuple[str, str]],
    patch_size: Tuple[int, int] = (4, 4),
) -> Tuple[torch.Tensor, ...]:
    """
    Batch seq flatten without including original shape in output
    for compatability with older versions
    """
    return _batch_seq_flatten(seq, H, W, flatten_method, patch_size)[:2]


def batch_seq_flatten_ex(
    seq: torch.Tensor,
    H: int,
    W: int,
    flatten_method: Union[str, Tuple[str, str]],
    patch_size: Tuple[int, int] = (4, 4),
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    return _batch_seq_flatten(seq, H, W, flatten_method, patch_size)


def reverse_batch_seq_flatten(
    seq: torch.Tensor,
    H: int,
    W: int,
    flatten_method: Union[str, Tuple[str, str]],
    patch_size: Tuple[int, int] = (4, 4),
) -> torch.Tensor:
    corner_map = {"tl": 4 - 0, "tr": 4 - 3, "br": 4 - 2, "bl": 4 - 1}

    # NOTE: Ref pts must be a 1d sequence of indices that fall within the bounds of
    # [0, H*W] for this to be effective
    def _traverse_seq(seq: torch.Tensor, flatten_method: str):
        start_corner = corner_map[flatten_method[:2]]
        # rotate so that desired corner is in top left
        rotations = int(flatten_method[2])
        row_prim = rotations == 0

        if flatten_method[3:] == "snake":
            seq[..., 1::2, :, :] = torch.flip(seq[..., 1::2, :, :].clone(), dims=[-2])

        if row_prim + start_corner % 2 == 0:
            seq = rearrange(seq, "... H W C -> ... W H C").contiguous()

        seq = torch.rot90(seq, start_corner, dims=[-2, -3])

        return seq

    seq = rearrange(seq, "... (H W) C -> ... H W C", H=H, W=W)

    if isinstance(flatten_method, tuple):
        hp, wp = patch_size
        seq = rearrange(seq, "... (H h2) (W w2) c ->  ... h2 w2 H W  c", h2=hp, w2=wp)
        seq = _traverse_seq(seq, flatten_method[1])
        seq = rearrange(seq, "... h2 w2 H W c -> ... H W h2 w2 c")
        seq = _traverse_seq(seq, flatten_method[0])
        seq = rearrange(seq, "... H W h2 w2 c -> ... (H h2) (W w2) c")

    else:
        seq = _traverse_seq(seq, flatten_method)
    return seq


def flatten_input_seq(
    flatten_method: str,
    H: torch.Tensor,
    W: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    # WARNING: VERY FRAGILE, DO NOT TOUCH
    """
    Get indices to flatten feature map of shape H, W, ... into shape
    (H W) ... and adjust reference points to match

    Args:
        H (Union[int, torch.Tensor]): Number of rows in feature map
        W (Union[int, torch.Tensor]): number of cols in feature map
        flatten_method (FlattenMethod): method used to flatten feature map of the format
        (t | b)(l | r)(0 - 3)(cross | snake)

    Raises:
        ValueError: Invalid flatten method

    Returns:
        Tuple[torch.Tensor, ...]: Row indices, Col indices
    """
    device = H.device
    H_num = H.item()
    W_num = W.item()
    torch._constrain_as_size(H_num, 1, 128)
    torch._constrain_as_size(W_num, 1, 256)

    left_90 = torch.tensor([[0, -1], [1, 0]], dtype=torch.float32, device=device)
    rotations = int(flatten_method[2])
    rmat = left_90**rotations
    start = torch.tensor(
        (float(flatten_method[1] == "r") - 0.5, float(flatten_method[0] == "b") - 0.5),
        dtype=torch.float32,
        device=device,
    )
    start_hori, start_vert = (torch.matmul(rmat, start) + 0.5).round().to(torch.bool)

    def arange_reverse_H():
        return torch.flip(
            torch.arange(H_num, device=device, dtype=torch.int64), dims=(0,)
        )

    def arange_H():
        return torch.arange(H_num, device=device, dtype=torch.int64)

    def arange_reverse_W():
        return torch.flip(
            torch.arange(W_num, device=device, dtype=torch.int64), dims=(0,)
        )

    def arange_W():
        return torch.arange(W_num, device=device, dtype=torch.int64)

    x = torch.cond(start_hori, arange_reverse_H, arange_H, tuple())
    y = torch.cond(start_vert, arange_reverse_W, arange_W, tuple())

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    row, col = torch.meshgrid(x, y, indexing="ij")
    col = torch.cond(
        flatten_method[3:] == "snake" and rotations % 2 == 0,
        even_flip,
        lambda x: x.contiguous(),
        (col,),
    )
    row = torch.cond(
        flatten_method[3:] == "snake" and rotations % 2 == 1,
        odd_flip,
        lambda x: x.contiguous(),
        (row,),
    )
    assert isinstance(row, torch.Tensor) and isinstance(col, torch.Tensor)
    row, col = torch.rot90(row, k=rotations), torch.rot90(col, k=rotations)

    return rearrange(row, "H W -> (H W)"), rearrange(col, "H W -> (H W)")


@lru_cache(maxsize=32, typed=True)
def flatten_input_seq_cpu(
    flatten_method: str,
    H: torch.Tensor,
    W: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """
    Get indices to flatten feature map of shape H, W, ... into shape
    (H W) ... and adjust reference points to match

    Args:
        H (Union[int, torch.Tensor]): Number of rows in feature map
        W (Union[int, torch.Tensor]): number of cols in feature map
        flatten_method (FlattenMethod): method used to flatten feature map of the format
        (t | b)(l | r)(0 - 3)(cross | snake)

    Raises:
        ValueError: Invalid flatten method

    Returns:
        Tuple[torch.Tensor, ...]: Row indices, Col indices
    """
    device = "cpu"
    H_num = H.item()
    W_num = W.item()

    left_90 = torch.tensor([[0, -1], [1, 0]], dtype=torch.float32, device=device)
    rotations = int(flatten_method[2]) % 4
    rmat = left_90**rotations
    start = torch.tensor(
        (float(flatten_method[1] == "r") - 0.5, float(flatten_method[0] == "b") - 0.5),
        dtype=torch.float32,
        device=device,
    )
    start_hori, start_vert = (torch.matmul(rmat, start) + 0.5).round().to(torch.bool)
    if start_vert:
        x = torch.arange(H_num - 1, -1, -1, device=device, dtype=torch.int64)
    else:
        x = torch.arange(H_num, device=device, dtype=torch.int64)
    if start_hori:
        y = torch.arange(W_num - 1, -1, -1, device=device, dtype=torch.int64)
    else:
        y = torch.arange(W_num, device=device, dtype=torch.int64)
    row, col = torch.meshgrid(x, y, indexing="ij")

    if flatten_method[3:] == "snake":
        if rotations % 2 == 0:
            col = even_flip(col)
        else:
            row = odd_flip(row)

    assert isinstance(row, torch.Tensor) and isinstance(col, torch.Tensor)
    row, col = torch.rot90(row, k=rotations), torch.rot90(col, k=rotations)

    return rearrange(row, "H W -> (H W)"), rearrange(col, "H W -> (H W)")

def _merge_multilevel_values_flattened(
    lowest_level: torch.Tensor,
    other_levels: torch.Tensor,
    insert_points: torch.Tensor,
):
    """
    Merge 2 sequences using points p as a list of insertion points
    at which to add elements from S to V
    """

    nt, hw = insert_points.shape
    bs, ncnt, lhw, c = lowest_level.shape
    bs, nc, hw, c = other_levels.shape

    new_length = lhw + hw
    sorted_inserts, offset = torch.sort(insert_points, stable=True)

    # check_insert_bounds(insert_points, new_length)
    # insert_points[:, offset] = insert_points[:, offset]  + torch.arange(hw, device=insert_points.device, dtype=insert_points.dtype).unsqueeze(0)
    for i in range(nt):
        insert_points[i].index_add_(
            -1,
            offset[i],
            torch.arange(hw, device=insert_points.device, dtype=insert_points.dtype),
        )

    # Create a mask to keep track of insert positions
    lowest_level = lowest_level.to(other_levels.dtype)
    insert_points = insert_points.contiguous()
    mask = torch.ones((nt, new_length), dtype=torch.bool, device=other_levels.device)
    result = lowest_level.new_zeros(
        (bs, ncnt, new_length, c),
    )

    # Mark the insertion points in the mask
    for i in range(len(insert_points)):
        mask[i, insert_points[i]] = False
    mask = rearrange(mask, "nt hw -> (nt hw)")
    value_insert_points = mask.nonzero()
    value_insert_points = rearrange(
        value_insert_points.squeeze(-1), "(nt hw)->nt hw", nt=nt
    )
    mask = rearrange(mask, "(nt hw)->nt hw", nt=nt)
    for ci in range(nc):
        for trav in range(nt):
            result[:, ci * nt + trav, mask[trav]] = lowest_level[:, ci * nt + trav]
            result[:, ci * nt + trav, insert_points[trav]] = other_levels[:, ci]

    return result, insert_points, mask


def merge_multilevel_values_flattened(
    lowest_level: torch.Tensor,
    other_levels: torch.Tensor,
    insert_points: torch.Tensor,
):
    """
    Merge 2 sequences using points p as a list of insertion points
    at which to add elements from S to V
    """

    nt, hw = insert_points.shape
    bs, ncnt, lhw, c = lowest_level.shape
    bs, nc, hw, c = other_levels.shape

    new_length = lhw + hw
    sorted_inserts, offset = torch.sort(insert_points, stable=True)

    # check_insert_bounds(insert_points, new_length)
    # insert_points[:, offset] = insert_points[:, offset]  + torch.arange(hw, device=insert_points.device, dtype=insert_points.dtype).unsqueeze(0)
    for i in range(nt):
        insert_points[i].index_add_(
            -1,
            offset[i],
            torch.arange(hw, device=insert_points.device, dtype=insert_points.dtype),
        )

    # Create a mask to keep track of insert positions
    lowest_level = lowest_level.to(other_levels.dtype)
    insert_points = insert_points.contiguous()
    mask = torch.ones((nt, new_length), dtype=torch.bool, device=other_levels.device)
    result = lowest_level.new_zeros(
        (bs, ncnt, new_length, c),
    )

    # Mark the insertion points in the mask
    for i in range(len(insert_points)):
        mask[i, insert_points[i]] = False
    mask = rearrange(mask, "nt hw -> (nt hw)")
    value_insert_points = mask.nonzero()
    value_insert_points = rearrange(
        value_insert_points.squeeze(-1), "(nt hw)->nt hw", nt=nt
    )
    mask = rearrange(mask, "(nt hw)->nt hw", nt=nt)
    for ci in range(nc):
        for trav in range(nt):
            result[:, ci * nt + trav, mask[trav]] = lowest_level[:, ci * nt + trav]
            result[:, ci * nt + trav, insert_points[trav]] = other_levels[:, ci]

    return result, insert_points, mask


def flatten_input_seq_with_default(
    H: torch.Tensor,
    W: torch.Tensor,
    flatten_method: str,
    unflatten_method: str,
) -> Tuple[torch.Tensor, ...]:
    """
    Get indices to flatten a tensor with dimensions ... H W ... and indices
    to return the tensor to the shape ... (H W) ... with a default traversal
    Arguments:
        H (int|torch.Tensor): Height
        W (int|torch.Tensor): Width
        flatten_method (str): Flatten method string
        unflatten_method (str): Unflatten method string
    Returns:
        Flatten Row Index, Flatten Column Index, Unflatten Index

    usage:
    sample = torch.arange(12).reshape(3,4)
    H, W = sample.shape
    r, c, unf = flatten_input_seq_with_default(H, W, "tl0snake", "tl0cross")
    sample[r, c]
    >> torch.Tensor([0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11])
    sample[r, c][unf]
    >> torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    """
    flat_row, flat_col = flatten_input_seq(H=H, W=W, flatten_method=flatten_method)

    unflatten_indices = flat_row * W + flat_col
    _, unflatten_indices = torch.sort(unflatten_indices, dim=0)
    unflat_row, unflat_col = flatten_input_seq(
        H=H, W=W, flatten_method=unflatten_method
    )
    final_indices = unflat_row * W + unflat_col
    return flat_row, flat_col, unflatten_indices[final_indices]


def flatten_input_local_scan(
    H: torch.Tensor,
    W: torch.Tensor,
    traversal_method: Tuple[str, str],
    patch_size: Tuple[int, int],
):
    # patch size for lowest level
    hp, wp = patch_size
    device = H.device

    def get_patches(H: Number, W: Number):
        grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        grid = pack(grid, "H W *")[0]
        # NOTE: Dim 0 is now the patch number, Dim 2 is the flattened patch column
        grid = rearrange(grid, "(H h2) (W w2) c ->  (H W) (h2 w2)  c", h2=hp, w2=wp)
        grid = grid[..., 0] * W + grid[..., 1]
        return grid.squeeze(-1)

    # number of patches
    ho, wo = H // hp, W // wp
    flat_patch_ind = get_patches(H.item(), W.item())
    # num_patches, numel_per_patch = flat_patch_ind.shape
    trout, trin = traversal_method
    tror, troc = flatten_input_seq(trout, ho, wo)
    tro = tror * wo + troc

    trr, trc = flatten_input_seq(
        trin, torch.tensor(hp, device=device), torch.tensor(wp, device=device)
    )
    tr = trr * wp + trc
    flat_patch_ind = flat_patch_ind[tro][:, tr].flatten()
    return flat_patch_ind


def flatten_input_and_int_ref_pts(
    H: torch.Tensor,
    W: torch.Tensor,
    ref_pts: torch.Tensor,
    flatten_method: Union[str, Tuple[str, str]],
    patch_size: Tuple[int, int] = (4, 4),
) -> Tuple[torch.Tensor, ...]:
    # NOTE: Ref pts must be a 1d sequence of indices that fall within the bounds of
    # [0, H*W] for this to be effective
    if isinstance(flatten_method, tuple):
        unflatten_indices = flatten_input_local_scan(H, W, flatten_method, patch_size)
        flat_row = torch.div(unflatten_indices, W, rounding_mode="floor")
        flat_col = torch.remainder(unflatten_indices, W)

    else:
        flat_row, flat_col = flatten_input_seq(H=H, W=W, flatten_method=flatten_method)
        unflatten_indices = flat_row * W + flat_col
    _, unflatten_indices = torch.sort(unflatten_indices, dim=0)
    return flat_row, flat_col, unflatten_indices[ref_pts]


@lru_cache(maxsize=32, typed=True)
def flatten_input_seq_cpu_old(
    flatten_method: str,
    H: torch.Tensor,
    W: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """
    Get indices to flatten feature map of shape H, W, ... into shape
    (H W) ... and adjust reference points to match

    Args:
        flatten_method (FlattenMethod): method used to flatten feature map of the format
        (t | b)(l | r)(0 - 3)(cross | snake)
        H (Union[int, torch.Tensor]): Number of rows in feature map
        W (Union[int, torch.Tensor]): number of cols in feature map

    Raises:
        ValueError: Invalid flatten method

    Returns:
        Tuple[torch.Tensor, ...]: Row indices, Col indices
    """
    # center reference points around 0 for simpler transformations

    device = "cpu"
    H, W = H.item(), W.item()  # type:ignore
    # ref_points -= ref_mean #/ ref_std
    row = repeat(torch.arange(H, device=device), "H -> H W", W=W)  # type: ignore
    col = repeat(torch.arange(W, device=device), "W -> H W", H=H)  # type: ignore

    left_90 = torch.tensor([[0, -1], [1, 0]], dtype=torch.float32, device=device)
    # start_reverse = flatten_method[2] == "r"
    rotations = int(flatten_method[2])
    rmat = left_90**rotations
    start = torch.tensor(
        (float(flatten_method[1] == "r") - 0.5, float(flatten_method[0] == "b") - 0.5),
        dtype=torch.float32,
        device=device,
    )
    start_hori, start_vert = (torch.matmul(rmat, start) + 0.5).round().to(torch.bool)

    if start_hori:
        col = torch.flip(col, dims=(1,))
    if start_vert:
        row = torch.flip(row, dims=(0,))

    match flatten_method[3:]:
        case "cross":
            pass
        case "snake":
            if rotations % 2 == 0:
                # reverse odd rows
                col = col.clone()
                col[1::2] = torch.flip(col[1::2], dims=(1,))
            else:
                # reverse odd columns
                row = row.clone()
                row[:, 1::2] = torch.flip(row[:, 1::2], dims=(0,))
        case _:
            raise ValueError(f"{flatten_method[4:]} is an invalid flatten method ")

    # TODO: Make sure this works as inteded for reference points
    if rotations:
        row, col = torch.rot90(row, k=rotations), torch.rot90(col, k=rotations)

    return rearrange(row, "H W -> (H W)"), rearrange(col, "H W -> (H W)")


def _merge_mask(
    num_values: int,
    insert_points: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Merge 2 sequences using points p as a list of insertion points
    at which to add elements from S to V
    """

    num_insert = insert_points.shape[0]
    new_length = num_values + num_insert
    sorted_inserts, offset = torch.sort(insert_points)
    insert_points.index_add_(
        0,
        offset,
        torch.arange(
            num_insert, device=insert_points.device, dtype=insert_points.dtype
        ),
    )

    # Create a mask to keep track of insert positions
    mask = torch.ones(new_length, dtype=torch.bool, device=insert_points.device)
    # max_insert = torch.max(insert_points)
    # assert max_insert < new_length, f"Insert points OOB, {max_insert} >= { new_length }"
    mask[insert_points] = False

    return offset, insert_points, mask, new_length


def get_merge_mask(
    num_values: int,
    insert_points: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Merge 2 sequences using points p as a list of insertion points
    at which to add elements from S to V
    """
    return _merge_mask(num_values, insert_points)[1:]


def get_merge_mask_and_sort(
    num_values: int,
    insert_points: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Merge 2 sequences using points p as a list of insertion points
    at which to add elements from S to V
    """
    offset, insert_points, mask, new_length = _merge_mask(num_values, insert_points)
    return offset, mask, new_length


def _merge_query_and_value_flattened(
    values: torch.Tensor,
    queries: torch.Tensor,
    insert_points: torch.Tensor,
):
    """
    Merge 2 sequences using points p as a list of insertion points
    at which to add elements from S to V
    """

    # num_insert = insert_points.shape[0]
    num_values, dim = values.shape
    insert_points, mask, new_length = get_merge_mask(num_values, insert_points)
    if values.dtype != queries.dtype:
        values = values.to(queries.dtype)
    result = torch.zeros((new_length, dim), dtype=values.dtype, device=values.device)
    # Mark the insertion points in the mask
    result[~mask] = queries
    result[mask] = values

    return result, insert_points, mask


def merge_query_and_value_flattened(
    values: torch.Tensor,
    queries: torch.Tensor,
    insert_points: torch.Tensor,
):
    """
    Merge 2 sequences using points insert points as a list of insertion points
    at which to add elements from queries to values.
    Returns merged sequence and indices of queries insert points
    """
    return _merge_query_and_value_flattened(values, queries, insert_points)[:-1]


def merge_query_and_value_flattened_return_vmask(
    values: torch.Tensor,
    queries: torch.Tensor,
    insert_points: torch.Tensor,
):
    """
    Merge 2 sequences using points insert points as a list of insertion points
    at which to add elements from queries to values.
    Also returns mask where true indicates a value appears at that location in
    the final sequence.
    """
    return _merge_query_and_value_flattened(values, queries, insert_points)


# @torch.compile(mode="max-autotune", dynamic=True)
def mask_queries_refs(
    query: torch.Tensor,
    ref_hits: torch.Tensor,
    cam_ind: torch.Tensor,
    H_: Union[int, torch.Tensor],
    W_: Union[int, torch.Tensor],
    z_levels: int,
) -> Tuple[torch.Tensor, ...]:
    """
    Extract all queries whose corresponding reference points
    fall within H x W (are valid), also return the valid reference points
    """
    initial_ref_locs = (
        (ref_hits[..., 0] * W_).floor() + (ref_hits[..., 1] * H_).floor() * W_
    ).long()
    valid_refs = (initial_ref_locs >= 0) & (initial_ref_locs < H_ * W_)
    query_level_cam_ind = repeat(cam_ind, "nq -> nq z_levels", z_levels=z_levels)[
        valid_refs
    ]
    query_level_cam = repeat(
        query,
        "nq c -> nq z_levels c",
        z_levels=z_levels,
    )[valid_refs]
    return query_level_cam, query_level_cam_ind, valid_refs, initial_ref_locs


def apply_bev_mask(
    query: torch.Tensor,
    ref_hits: torch.Tensor,
    bev_mask: torch.Tensor,
    H_: Union[int, torch.Tensor],
    W_: Union[int, torch.Tensor],
) -> Tuple[torch.Tensor, ...]:
    """
    Extract all queries whose corresponding reference points
    fall within H x W (are valid), also return the valid reference points
    """

    initial_ref_locs = (
        (ref_hits[bev_mask][..., 0] * (W_)).floor()
        + (ref_hits[bev_mask][..., 1] * (H_)).floor() * W_
    ).long()

    # initial_ref_locs = (
    #     (ref_hits[bev_mask][..., 0] * ( W_ - 1 )).floor()
    #     + (ref_hits[bev_mask][..., 1] * ( H_ - 1 )).floor() * W_
    # ).long()
    count = bev_mask.sum(-1)
    query_level_cam_ind = bev_mask.nonzero()[:, 0]
    query_level_cam = query[query_level_cam_ind]
    return query_level_cam, query_level_cam_ind, initial_ref_locs, count.squeeze(0)


def index_add_to_residual(
    seq: torch.Tensor,
    queries: torch.Tensor,
    extract_indices: torch.Tensor,
):
    """
    Indexing done on the second dim of sequence and queries
    """
    # print_log(
    #     f"Query shape: {queries.shape}, Indices shape: {extract_indices.shape}",
    #     "current",
    #     logging.DEBUG,
    # )

    seq = seq.to(queries.dtype)
    queries.index_add_(1, extract_indices[1], seq[:, extract_indices[0]])


def index_add_to_residual_v2(
    seq: torch.Tensor,
    queries: torch.Tensor,
    extract_indices: torch.Tensor,
):
    """
    Indexing done on the first dim of sequence and Queries
    """
    seq = seq.to(queries.dtype)
    queries.index_add_(0, extract_indices[1], seq[extract_indices[0]])


def put_add_to_residual(
    seq: torch.Tensor,
    queries: torch.Tensor,
    extract_indices: torch.Tensor,
):
    """
    Extract indices must have 2 rows where all of
    row 0 have shape integer values < len(seq) and all of
    row 1 have shape integer values < len(queries)
    """
    seq = seq.to(queries.dtype)
    queries.index_put_((extract_indices[1],), seq[extract_indices[0]], accumulate=True)


def put_add_to_residual_v2(
    seq: torch.Tensor,
    queries: torch.Tensor,
    extract_indices: torch.Tensor,
):
    """
    Requires Seq to be equal in length to extract indices @ dim[1]
    Extract indices may still have 2 rows
    """
    seq = seq.to(queries.dtype)
    queries.index_put_((extract_indices[-1],), seq, accumulate=True)


def print_query_grid(H: int, W: int, extract_indices: torch.Tensor):
    grid = torch.zeros((H * W))
    empty = "░░"
    filled = "▓▓"
    queries = extract_indices.unique()
    grid[queries] = 1
    grid = rearrange(grid, "(H W) -> H W", H=H)
    input_arr = []
    for r in range(H):
        for c in range(W):
            entry = empty if grid[r, c] == 0 else filled
            input_arr.append(entry)
        input_arr.append("\n")
    final_string = "".join(input_arr)
    print(final_string)
    return final_string


def compare_tensors(
    seq: torch.Tensor, queries: torch.Tensor, extract_indices: torch.Tensor
):
    print("Queries shape:", queries.shape)
    print("Extract indices shape:", extract_indices.shape)
    print("Sequence Shape:", seq.shape)

    print("Duplicated Queries shape:", queries[extract_indices[1]].shape)
    q_seq = seq[extract_indices[0]]
    print("Extracted shape:", q_seq.shape)
    print()
    errors = 0
    passes = 0
    emin = float("inf")
    emax = 0
    print("Hit Query locations in grid for camera")

    try:
        assert torch.allclose(q_seq, queries[extract_indices[1]])
        print(q_seq[:3, :10])
        print(queries[extract_indices[1]][:3, :10])
        print("********Passed All Close Assertion********")
    except AssertionError:
        for i, (output, query) in enumerate(zip(q_seq, queries[extract_indices[1]])):
            try:
                assert torch.allclose(query, output)
                passes += 1

            except AssertionError:
                errors += 1
                emin = min(i, emin)
                emax = max(i, emax)
                # print(f"Query[{i}] ~= Output[{i}]")
                # print(f"{query[:5]} ~= {output[:5]}")
            print(
                f"Passed: {passes:,}, Errors: {errors:,}, Index Range: {emin, emax}",
                end="\r",
            )
        print(
            f"Passed: {passes:,}, Errors: {errors:,}, Index Range: {emin, emax}",
            end="\n",
        )

    print()


def compare_output_tensors(gt: torch.Tensor, test: torch.Tensor):
    gt = gt.float()
    test = test.float()
    emin = float("inf")
    emax = 0
    errors = 0
    passes = 0
    try:
        assert torch.allclose(torch.zeros_like(gt), test % gt)
    except AssertionError:
        for i, (truth, tt) in enumerate(zip(gt, test)):
            try:
                assert torch.allclose(
                    torch.zeros_like(tt, dtype=torch.float32), tt % truth
                )
                passes += 1

            except AssertionError:
                # print(f"test[{i}] ~= gt[{i}]")
                # print(f"{ ( tt % truth ).sum() } ~= 0")
                errors += 1
                emin = min(i, emin)
                emax = max(i, emax)
        print(f"Error index range: {emin}, {emax}")
        print(f"Num errors: {errors}")


def extract_slow(
    seq: torch.Tensor, queries: torch.Tensor, extract_indices: torch.Tensor
):
    q_seq = seq[extract_indices[0]]
    for i, out in zip(extract_indices[1], q_seq):
        queries[i] += out
    return queries


def interleaved_feature_1_scale(base, high_level):
    # chunk_base =rearrange(base, "(h h1) (w w1) ... -> (h1 w1) (h w) ...", h=2, w=2)
    chunk_base = rearrange(base, "(h h2) (w w2) ... -> (h w) (h2 w2) ...", h2=2, w2=2)
    flatten_high_level = rearrange(high_level, "H W C -> (H W) 1 C")
    final_chuck, _ = pack((chunk_base, flatten_high_level), "g * C")
    final_chuck = rearrange(final_chuck, "g L C -> (g L) C")

    return final_chuck


def interleaved_feature_2_scale(base, high_level, higher_level):
    interleaved_feature_1 = interleaved_feature_1_scale(base, high_level)
    interleaved_feature_1 = rearrange(
        interleaved_feature_1,
        "(H W g) C -> H W g C",
        H=high_level.shape[0],
        W=high_level.shape[1],
    )

    chunk_base = rearrange(
        interleaved_feature_1, "(h h2) (w w2) ... -> (h w) (h2 w2) ...", h2=2, w2=2
    )
    flatten_high_level = rearrange(higher_level, "H W C -> (H W) 1 C")
    final_chuck, _ = pack((chunk_base, flatten_high_level), "g * C")
    final_chuck = rearrange(final_chuck, "g L C -> (g L) C")

    return final_chuck


def merge_query_and_interleaved_value_flattened(
    values: torch.Tensor,
    queries: torch.Tensor,
    insert_points: torch.Tensor,
):
    """
    Merge 2 sequences using points p as a list of insertion points
    at which to add elements from S to V
    """
    _, group, _ = values.shape
    insert_points = insert_points * group
    num_insert = insert_points.shape[0]

    values = rearrange(values, "num_values group C -> (num_values group) C")
    num_values, dim = values.shape

    new_length = num_values + num_insert
    sorted_inserts, offset = torch.sort(insert_points)

    # check_insert_bounds(insert_points, new_length)
    # insert_points[offset] = insert_points[offset] + .to(insert_points.dtype)
    insert_points.index_add_(
        0,
        offset,
        torch.arange(
            num_insert, device=insert_points.device, dtype=insert_points.dtype
        ),
    )

    # Create a mask to keep track of insert positions
    mask = torch.ones(new_length, dtype=torch.bool, device=insert_points.device)
    if values.dtype != queries.dtype:
        values = values.to(queries.dtype)
    result = torch.zeros((new_length, dim), dtype=values.dtype, device=values.device)
    # Mark the insertion points in the mask
    mask[insert_points] = False
    result[~mask] = queries
    result[mask] = values

    return result, insert_points


def mask_queries_refs_with_return_low_and_high(
    query: torch.Tensor,
    ref_hits: torch.Tensor,
    cam_ind: torch.Tensor,
    H_l: Union[int, torch.Tensor],
    W_l: Union[int, torch.Tensor],
    H_h: Union[int, torch.Tensor],
    W_h: Union[int, torch.Tensor],
    z_levels: int,
) -> Tuple[torch.Tensor, ...]:
    initial_ref_locs_l = (
        (ref_hits[..., 0] * W_l).floor() + (ref_hits[..., 1] * H_l).floor() * W_l
    ).to(torch.int)
    initial_ref_locs_h = (
        (ref_hits[..., 0] * W_h).floor() + (ref_hits[..., 1] * H_h).floor() * W_h
    ).to(torch.int)

    valid_refs = (
        (initial_ref_locs_l >= 0) & (initial_ref_locs_l < H_l * W_l)
    ).nonzero()
    query_level_cam_ind = repeat(cam_ind, "nq -> nq z_levels", z_levels=z_levels)[
        valid_refs[:, 0], valid_refs[:, 1]
    ]
    query_level_cam = repeat(
        query,
        "nq c -> nq z_levels c",
        z_levels=z_levels,
    )
    query_level_cam = query_level_cam[valid_refs[:, 0], valid_refs[:, 1]]
    return (
        query_level_cam,
        query_level_cam_ind,
        valid_refs,
        initial_ref_locs_l,
        initial_ref_locs_h,
    )
