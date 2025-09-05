from typing import List, Sequence, Tuple

import torch
from einops import pack, rearrange, repeat
from mmengine.registry import TASK_UTILS
from projects.MAMBEV.mambev.utils.sequence_utils import (
    _merge_multilevel_values_flattened,
    flatten_input_seq,
    flatten_input_seq_cpu,
    flatten_input_seq_cpu_old,
)
from torch.types import Number


@TASK_UTILS.register_module()
class GlobalTraversalConstructor:
    """
    dict(
        type="TraversalConstructor",
        traversals=["tl0snake","tl1snake"],
        device="cpu",
        with_default=False,
        default="tl0cross",
        batch_type: str = "cam",
        interleave: bool = True,
    )
    """

    def __init__(
        self,
        traversals: Sequence[str],
        device: str = "cpu",
        with_default: bool = False,
        default: str = "tl0cross",
        batch_type: str = "cam",
        interleave: bool = True,
        patch_interleave: bool = True,
        traversals_inner: Sequence[str] = ("tl0cross",),
        patch_size: Tuple[int, int] = (16, 16),
    ) -> None:
        self.device = device
        self.traversals = traversals
        self.traversals_inner = traversals_inner
        self.ntrav = len(self.traversals)
        self.batch_type = batch_type
        self.interleave = interleave
        self.patch_size = torch.tensor(patch_size)
        self.patch_interleave = patch_interleave
        if self.device == "cpu":
            self.index_gen_func = flatten_input_seq_cpu
        elif self.device == "cpu_old":
            self.index_gen_func = flatten_input_seq_cpu_old
        else:
            self.index_gen_func = flatten_input_seq

        self.with_default = with_default
        self.default_traversal = default
        if with_default:
            self.call_func = self.flatten_input_seq_with_default
        else:
            self.call_func = self.index_gen_func

    def __call__(
        self,
        value_level_list: List[torch.Tensor],
        spatial_shapes: torch.Tensor,
    ):
        return self.construct_traversal_sequences(
            value_level_list=value_level_list, spatial_shapes=spatial_shapes
        )

    def get_patches(self, H: Number, W: Number):
        grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        # print_log(f"Level Shape: {H}, {W}", logger="current", level=logging.DEBUG)
        hp, wp = self.patch_size
        # print_log(f"Patch Shape: {hp}, {wp}", logger="current", level=logging.DEBUG)
        # print_log(f"Mod Shapes: {H%hp}, {W%wp}", logger="current", level=logging.DEBUG)
        grid = pack(grid, "H W *")[0]
        # NOTE: Dim 0 is now the patch number, Dim 2 is the flattened patch column
        grid = rearrange(grid, "(H h2) (W w2) c ->  (H W) (h2 w2)  c", h2=hp, w2=wp)
        grid = grid[..., 0] * W + grid[..., 1]
        return grid.squeeze(-1)

    def interleave_multilevel_patched(
        self,
        value_level_list: List[torch.Tensor],
        spatial_shapes: torch.Tensor,
    ):
        low_level = value_level_list.pop(0)
        out_shape = spatial_shapes[0:1]
        H, W = spatial_shapes[0]
        spatial_shapes = spatial_shapes[1:]

        # patch size for lowest level
        hp, wp = self.patch_size

        # number of patches
        ho, wo = H // hp, W // wp
        flat_patch_ind = self.get_patches(H.item(), W.item()).to(low_level.device)
        # num_patches, numel_per_patch = flat_patch_ind.shape
        patch_indices = []
        for trout in self.traversals:
            tror, troc = self.index_gen_func(trout, ho, wo)
            tro = tror * wo + troc
            for trin in self.traversals_inner:
                trr, trc = self.index_gen_func(trin, hp, wp)
                tr = trr * wp + trc
                patch_indices.append(flat_patch_ind[tro][:, tr].flatten())

        patch_indices = torch.vstack(patch_indices)
        low_level = rearrange(
            low_level[:, :, patch_indices],
            "bs nc nt hw c -> bs (nc nt) hw c",
        )
        # find reference point locations of high level on low level traversal
        insert_locs = []
        for lH, lW in spatial_shapes:
            # level index -> normalized index -> low level index -> low level index post traversal
            insert_locs.append(
                patch_indices[:, self.shape_to_reference_points(lH, lW, H, W)]
            )

        # concat low level into one seq
        values, _ = pack(value_level_list, "bs nc * c")
        insert_locs, _ = pack(insert_locs, "nt *")
        _, ind = torch.sort(insert_locs)
        # merge sequences
        batch_values, _, vmask = _merge_multilevel_values_flattened(
            low_level, values, insert_locs
        )
        # map low level value indices from traversal locations -> post merge location
        batch_indices = []
        # map low level value indices from traversal locations -> post merge location
        for i, trav in enumerate(vmask):
            batch_indices.append(trav.nonzero()[patch_indices[i]].squeeze())
        # batch_indices = vmask.nonzero()[level_indices]
        batch_indices = torch.vstack(batch_indices)
        seq_lens = (
            batch_values.new_ones((batch_values.shape[2], 1), dtype=torch.int64)
            * batch_values.shape[-2]
        )

        return batch_values, batch_indices, seq_lens, out_shape

    def interleave_multilevel_naive(
        self,
        value_level_list: List[torch.Tensor],
        spatial_shapes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Organize multilevel image feature values into a single sequence
        using a naive interleaving strategy and mutliple traversals

        returns:
            values (torch.Tensor):
            indices (torch.Tensor): indices[t, i] = value_level_list[i] new index for traversal t
            seq_lens (torch.Tensor): Length of each sequence in the output sequences
            out_shape (torch.tensor): Spatial shapes of the output sequences
        """
        # do traversal on low level
        # bs, nc = value_level_list[0].shape[:2]
        # Add extra rows for each additional traversal
        low_level = value_level_list.pop(0)
        out_shape = spatial_shapes[0:1]
        lH, lW = spatial_shapes[0]
        spatial_shapes = spatial_shapes[1:]
        level_indices = []
        for traversal in self.traversals:
            level_indices.append(self.reorder_flat_seq(traversal, lH, lW))
        level_indices = torch.vstack(level_indices)
        low_level = rearrange(
            low_level[:, :, level_indices],
            "bs nc nt hw c -> bs (nc nt) hw c",
        )
        # find reference point locations of high level on low level traversal
        insert_locs = []
        for H, W in spatial_shapes:
            # level index -> normalized index -> low level index -> low level index post traversal
            insert_locs.append(
                level_indices[:, self.shape_to_reference_points(H, W, lH, lW)]
            )

        # concat low level into one seq
        values, _ = pack(value_level_list, "bs nc * c")
        insert_locs, _ = pack(insert_locs, "nt *")
        # merge sequences
        batch_values, _, vmask = _merge_multilevel_values_flattened(
            low_level, values, insert_locs
        )
        batch_indices = []
        # map low level value indices from traversal locations -> post merge location
        for i, trav in enumerate(vmask):
            batch_indices.append(trav.nonzero()[level_indices[i]].squeeze())
        batch_indices = torch.vstack(batch_indices)
        seq_lens = (
            batch_values.new_ones((batch_values.shape[2], 1), dtype=torch.int64)
            * batch_values.shape[-2]
        )

        return batch_values, batch_indices, seq_lens, out_shape

    def rescale_indices(
        self,
        step_H: int,
        step_W: int,
        H: Number,
        W: Number,
    ):
        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        ref_row, ref_col = torch.meshgrid(
            torch.linspace(0, H - 1, step_H, dtype=torch.long),
            torch.linspace(0, W - 1, step_W, dtype=torch.long),
            indexing="ij",
        )
        # ref_row = ref_row.reshape(-1)[None] / H
        # ref_col = ref_col.reshape(-1)[None] / W
        return ref_row, ref_col

    def shape_to_reference_points(
        self,
        H: torch.Tensor | Number,
        W: torch.Tensor | Number,
        lH: torch.Tensor | Number,
        lW: torch.Tensor | Number,
    ):
        if isinstance(H, torch.Tensor):
            H = H.item()
        if isinstance(W, torch.Tensor):
            W = W.item()
        if isinstance(lH, torch.Tensor):
            lH = lH.item()
        if isinstance(lW, torch.Tensor):
            lW = lW.item()
        row, col = self.rescale_indices(int(H), int(W), lH, lW)

        row = rearrange(row, "H W -> (H W)")
        col = rearrange(col, "H W -> (H W)")
        # print(f"{H,W} -> {len(row)} == {H*W}")
        return row * lW + col

    @torch.compile(dynamic=True, backend="eager")
    def construct_traversal_sequences(
        self,
        value_level_list: List[torch.Tensor],
        spatial_shapes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            Image Feature Values in flattened traversal orders
            Indices of where Indices[i] is the location of Value[i] in the
                traversal order for that sequence
            Sequence Lengths of each sub-sequence in a given batched sequence
                whose sum is <= len of the sequence (less than implies padding)
            Spatial shapes of the levels in the output sequence
        """

        assert len(value_level_list) == len(spatial_shapes)
        # TODO: Add support for patched sequence interleaving
        if self.interleave and len(value_level_list) > 1:
            if self.patch_interleave:
                return self.interleave_multilevel_patched(
                    value_level_list, spatial_shapes
                )
            else:
                return self.interleave_multilevel_naive(
                    value_level_list, spatial_shapes
                )

        bs, nc = value_level_list[0].shape[:2]
        # Add extra rows for each additional traversal
        value_level_list = [
            repeat(v, "bs nc hw c -> bs (nc repeat) hw c", repeat=self.ntrav)
            for v in value_level_list
        ]
        indices = []
        for lvl_idx, (H, W) in enumerate(spatial_shapes):
            level_indices = []
            for traversal in self.traversals:
                # these indices can index a single sequence
                level_indices.append(self.reorder_flat_seq(traversal, H, W))
            level_indices = torch.vstack(level_indices)
            indices.append(level_indices)
            value_level_list[lvl_idx] = rearrange(
                value_level_list[lvl_idx][:, :, level_indices],
                "bs nc nt hw c -> bs (nc nt) hw c",
            )

        seq_lens = torch.tensor([v.shape[-2] for v in value_level_list])

        match self.batch_type:
            case "cam":
                batch_values, _ = pack(value_level_list, "bs ncnt * c")
                for i in range(1, len(seq_lens)):
                    indices[i] += seq_lens[i - 1]
                batch_indices, _ = pack(indices, "nt *")
                seq_lens = repeat(seq_lens, "k -> ncnt k", ncnt=nc * self.ntrav)
            #
            # case "level":
            #     print_log(
            #         "Batching by level vastly increases the amount of padding in sequences.",
            #         "current",
            #         logging.WARN,
            #     )
            #     value_level_list = [
            #         rearrange(
            #             value, "bs (nc nt) hw c ->  (nc hw) nt bs c", nt=self.ntrav
            #         )
            #         for value in value_level_list
            #     ]
            #
            #     batch_values = pad_sequence(value_level_list, False, 0.0)
            #     batch_values = rearrange(
            #         batch_values, "nchw nl nt bs c -> bs (nl nt) nchw c"
            #     )
            #
            #     seq_lens = repeat(
            #         seq_lens.unsqueeze((-1)),
            #         "nl p -> bs (nl nt) (p nc) ",
            #         nc=nc,
            #         nt=self.ntrav,
            #         bs=bs,
            #     )
            #     cu_seqlens = seq_lens.cumsum(2)
            #
            #     new_indices = []
            #     for i in range(1, len(seq_lens)):
            #         new_indices.append(indices[i] + seq_lens[i - 1])
            #         indices[i] += seq_lens[i - 1]
            #     batch_indices, _ = pack(indices, "nt *")
            # case "level_cam":
            #     print_log(
            #         "Batching by level and cam vastly increases the amount of padding in sequences.",
            #         "current",
            #         logging.WARN,
            #     )
            #
            #     batch_values = pad_sequence(value_level_list, True, 0.0)
            #
            #     # level1cam1trav1, level1cam1trav2, level1cam2trav1, level1cam1trav2, ... levelKcamJtravI
            #     batch_values = rearrange(
            #         batch_values,
            #         "nl sl bs (nc nt) c -> bs (nl nc nt) sl c",
            #         nt=self.ntrav,
            #     )
            #     # repeat the traversals for each level num cam times
            #     seq_lens = repeat(
            #         seq_lens, "k -> bs (k nt nc) ", bs=bs, nt=self.ntrav, nc=nc
            #     )
            #     batch_indices = [
            #         repeat(ind, "nt hw -> bs (nt nc) hw ") for ind in indices
            #     ]
            # no changes to indices since every sequence remains same length

            # case "all":
            #     batch_values, _ = pack(value_level_list, "bs ncnt * c")
            #     batch_values = rearrange(batch_values, "bs (ncnt nlhw) c")
            #
            case _:
                raise ValueError("Invalid batch type")
        return batch_values, batch_indices, seq_lens, spatial_shapes

    def reorder_flat_seq(
        self,
        flatten_method: str,
        H: torch.Tensor,
        W: torch.Tensor,
    ) -> torch.Tensor:
        """NOTE Flat here means tl0cross"""
        unflat_row, unflat_col = self.index_gen_func(
            H=H, W=W, flatten_method=flatten_method
        )
        final_indices = unflat_row * W.item() + unflat_col
        return final_indices.to(H.device)

    def flatten_input_seq_with_default(
        self,
        flatten_method: str,
        H: torch.Tensor,
        W: torch.Tensor,
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
        flat_row, flat_col = self.index_gen_func(
            H=H, W=W, flatten_method=flatten_method
        )

        unflatten_indices = flat_row * W.item() + flat_col
        _, unflatten_indices = torch.sort(unflatten_indices, dim=0)
        unflat_row, unflat_col = self.index_gen_func(
            H=H, W=W, flatten_method=self.default_traversal
        )
        final_indices = unflat_row * W.item() + unflat_col
        return (
            flat_row.to(H.device),
            flat_col.to(H.device),
            unflatten_indices[final_indices].to(H.device),
        )

    def flatten_input_and_int_ref_pts(
        self,
        flatten_method: str,
        H: torch.Tensor,
        W: torch.Tensor,
        ref_pts: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # NOTE: Ref pts must be a 1d sequence of indices that fall within the bounds of
        # [0, H*W] for this to be effective
        flat_row, flat_col = self.index_gen_func(
            H=H, W=W, flatten_method=flatten_method
        )

        unflatten_indices = flat_row * W.item() + flat_col
        _, unflatten_indices = torch.sort(unflatten_indices, dim=0)
        return (
            flat_row.to(H.device),
            flat_col.to(H.device),
            unflatten_indices[ref_pts].to(H.device),
        )
